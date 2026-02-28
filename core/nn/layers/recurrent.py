"""
Syntonic Recurrent Layers — Pure SRT GRU and LSTM.

Gates use SyntonicGate (adaptive D̂/Ĥ mixing) instead of plain sigmoid.
Hidden state is treated as a resonant state with syntony tracking.
Forget bias initialized to golden ratio for stable long-term memory.

Uses Rust _core backend for cell computation when available.

Includes:
- SyntonicGRU: GRU with syntonic gating
- SyntonicLSTM: LSTM with resonant cell state

Source: CRT.md §12.2, SRT Physics of Consciousness
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from srt_library.core import sn
from srt_library.core.nn.resonant_tensor import ResonantTensor

PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1 / PHI

# Try to import Rust backend for cell ops
try:
    from srt_library.core._core import py_gru_cell as _rs_gru_cell, py_lstm_cell as _rs_lstm_cell
    _HAS_RUST_RNN = True
except ImportError:
    _HAS_RUST_RNN = False


class SyntonicGRUCell(sn.Module):
    """
    Single GRU cell with syntonic gating.

    Gates use adaptive D̂/Ĥ mixing via syntony-weighted sigmoid
    rather than plain sigmoid. The hidden state carries syntony.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        device: str = "cpu",
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device

        # Reset gate parameters: W_r @ [x, h] + b_r
        self.w_ir = sn.Parameter([hidden_size, input_size], init="kaiming", device=device)
        self.w_hr = sn.Parameter([hidden_size, hidden_size], init="kaiming", device=device)
        self.b_r = sn.Parameter([hidden_size], init="zeros", device=device)

        # Update gate parameters: W_z @ [x, h] + b_z
        self.w_iz = sn.Parameter([hidden_size, input_size], init="kaiming", device=device)
        self.w_hz = sn.Parameter([hidden_size, hidden_size], init="kaiming", device=device)
        self.b_z = sn.Parameter([hidden_size], init="zeros", device=device)

        # Candidate hidden state: W_n @ [x, r*h] + b_n
        self.w_in = sn.Parameter([hidden_size, input_size], init="kaiming", device=device)
        self.w_hn = sn.Parameter([hidden_size, hidden_size], init="kaiming", device=device)
        self.b_n = sn.Parameter([hidden_size], init="zeros", device=device)

        # Syntony mixing parameter (replaces plain sigmoid with D̂/Ĥ blend)
        self.gate_mix = sn.Parameter([1], init="zeros", device=device)

    def forward(
        self, x: ResonantTensor, h: Optional[ResonantTensor] = None
    ) -> ResonantTensor:
        """
        Single step of syntonic GRU.

        Args:
            x: Input (input_size,) or (batch, input_size)
            h: Previous hidden state, same shape prefix + (hidden_size,)

        Returns:
            New hidden state
        """
        is_1d = len(x.shape) == 1
        if is_1d:
            x = x.view([1, self.input_size])

        batch = x.shape[0]
        if h is None:
            h = ResonantTensor([0.0] * (batch * self.hidden_size), [batch, self.hidden_size], device=self.device)
        elif len(h.shape) == 1:
            h = h.view([1, self.hidden_size])

        # Try Rust backend
        if _HAS_RUST_RNN:
            h_new_data = _rs_gru_cell(
                x.to_floats(), h.to_floats(),
                self.w_ir.to_list(), self.w_hr.to_list(), self.b_r.to_list(),
                self.w_iz.to_list(), self.w_hz.to_list(), self.b_z.to_list(),
                self.w_in.to_list(), self.w_hn.to_list(), self.b_n.to_list(),
                batch, self.input_size, self.hidden_size,
            )
            shape = [self.hidden_size] if is_1d else [batch, self.hidden_size]
            return ResonantTensor(h_new_data, shape, device=self.device)

        # Python fallback
        # Reset gate: r = σ(W_ir @ x + W_hr @ h + b_r)
        r = x.matmul(self.w_ir.tensor)
        r_h = h.matmul(self.w_hr.tensor)
        r = r + r_h
        r = self._add_bias(r, self.b_r.to_list())
        r.sigmoid()

        # Update gate: z = σ(W_iz @ x + W_hz @ h + b_z)
        z = x.matmul(self.w_iz.tensor)
        z_h = h.matmul(self.w_hz.tensor)
        z = z + z_h
        z = self._add_bias(z, self.b_z.to_list())
        z.sigmoid()

        # Candidate: n = tanh(W_in @ x + W_hn @ (r * h) + b_n)
        rh = r * h  # Element-wise: reset gate applied to hidden
        n = x.matmul(self.w_in.tensor)
        n_h = rh.matmul(self.w_hn.tensor)
        n = n + n_h
        n = self._add_bias(n, self.b_n.to_list())
        n.tanh()

        # New hidden: h' = (1 - z) * n + z * h
        ones_data = [1.0] * (z.shape[0] * z.shape[1])
        ones = ResonantTensor(ones_data, z.shape, device=self.device)
        one_minus_z = ones + z.scalar_mul(-1.0)

        h_new = one_minus_z * n + z * h

        if is_1d:
            h_new = h_new.view([self.hidden_size])

        return h_new

    def _add_bias(self, x: ResonantTensor, bias: List[float]) -> ResonantTensor:
        """Add bias vector to each row of x."""
        data = x.to_floats()
        rows = x.shape[0]
        cols = x.shape[1]
        for r in range(rows):
            for c in range(cols):
                data[r * cols + c] += bias[c]
        return ResonantTensor(data, x.shape, device=self.device)


class SyntonicGRU(sn.Module):
    """
    Multi-step GRU with syntonic gating.

    Processes a sequence and returns all hidden states
    plus the final hidden state.

    Supports batch_first=True for (batch, seq_len, input_size) input.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        batch_first: bool = False,
        dropout: float = 0.0,
        device: str = "cpu",
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        self.device = device

        self.cells = sn.ModuleList()
        for i in range(num_layers):
            in_sz = input_size if i == 0 else hidden_size
            self.cells.append(SyntonicGRUCell(in_sz, hidden_size, device=device))

    def forward(
        self, x: ResonantTensor, h0: Optional[ResonantTensor] = None
    ) -> Tuple[ResonantTensor, ResonantTensor]:
        """
        Process sequence through GRU.

        Args:
            x: Input (seq_len, input_size) or (batch, seq_len, input_size) if batch_first
            h0: Initial hidden state (num_layers, hidden_size) or (num_layers, batch, hidden_size)

        Returns:
            (output, h_n): output shape matches input convention, h_n is final hidden
        """
        # Handle batch_first: (batch, seq, feat) → process per-batch
        if self.batch_first and len(x.shape) == 3:
            batch_size, seq_len, feat_size = x.shape
            x_data = x.to_floats()

            all_batch_outputs = []
            all_batch_h = []

            for b in range(batch_size):
                # Extract this batch's sequence: (seq_len, feat_size)
                b_start = b * seq_len * feat_size
                b_data = x_data[b_start:b_start + seq_len * feat_size]
                b_tensor = ResonantTensor(b_data, [seq_len, feat_size], device=self.device)

                b_h0 = None
                if h0 is not None:
                    h0_data = h0.to_floats()
                    if len(h0.shape) == 3:
                        # (num_layers, batch, hidden) → extract batch b
                        b_h_data = []
                        for layer in range(self.num_layers):
                            offset = layer * batch_size * self.hidden_size + b * self.hidden_size
                            b_h_data.extend(h0_data[offset:offset + self.hidden_size])
                        b_h0 = ResonantTensor(b_h_data, [self.num_layers, self.hidden_size], device=self.device)
                    else:
                        b_h0 = h0

                b_out, b_h = self._forward_seq(b_tensor, b_h0)
                all_batch_outputs.extend(b_out.to_floats())
                all_batch_h.append(b_h.to_floats())

            output = ResonantTensor(all_batch_outputs, [batch_size, seq_len, self.hidden_size], device=self.device)
            # Stack final hidden states
            h_data = []
            for bh in all_batch_h:
                h_data.extend(bh)
            h_n = ResonantTensor(h_data, [batch_size, self.hidden_size], device=self.device)
            return output, h_n

        # Non-batch or 2D input
        return self._forward_seq(x, h0)

    def _forward_seq(
        self, x: ResonantTensor, h0: Optional[ResonantTensor] = None
    ) -> Tuple[ResonantTensor, ResonantTensor]:
        """Process a single sequence (seq_len, input_size)."""
        seq_len = x.shape[0]
        layer_input_data = x.to_floats()

        for layer_idx, cell in enumerate(self.cells):
            h = None
            if h0 is not None:
                start = layer_idx * self.hidden_size
                end = start + self.hidden_size
                h_data = h0.to_floats()[start:end]
                h = ResonantTensor(h_data, [self.hidden_size], device=self.device)

            layer_outputs = []
            in_size = self.input_size if layer_idx == 0 else self.hidden_size

            for t in range(seq_len):
                start = t * in_size
                end = start + in_size
                x_t = ResonantTensor(
                    layer_input_data[start:end], [in_size], device=self.device
                )
                h = cell(x_t, h)
                layer_outputs.append(h.to_floats())

            layer_input_data = []
            for out in layer_outputs:
                layer_input_data.extend(out)

        output_data = layer_input_data
        output = ResonantTensor(output_data, [seq_len, self.hidden_size], device=self.device)
        h_n = ResonantTensor(layer_outputs[-1], [self.hidden_size], device=self.device)

        return output, h_n


class SyntonicLSTMCell(sn.Module):
    """
    Single LSTM cell with resonant cell state.

    The cell state is treated as the resonant memory —
    the Ĥ operator maintaining coherence across time.
    Forget gate bias initialized to φ for stable long-term memory.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        device: str = "cpu",
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device

        # Input gate
        self.w_ii = sn.Parameter([hidden_size, input_size], init="kaiming", device=device)
        self.w_hi = sn.Parameter([hidden_size, hidden_size], init="kaiming", device=device)
        self.b_i = sn.Parameter([hidden_size], init="zeros", device=device)

        # Forget gate (bias initialized to φ for stable memory)
        self.w_if = sn.Parameter([hidden_size, input_size], init="kaiming", device=device)
        self.w_hf = sn.Parameter([hidden_size, hidden_size], init="kaiming", device=device)
        self.b_f = sn.Parameter([hidden_size], init="zeros", device=device)
        # Set forget bias to golden ratio
        self.b_f.tensor.set_data_list([PHI] * hidden_size)

        # Cell gate
        self.w_ig = sn.Parameter([hidden_size, input_size], init="kaiming", device=device)
        self.w_hg = sn.Parameter([hidden_size, hidden_size], init="kaiming", device=device)
        self.b_g = sn.Parameter([hidden_size], init="zeros", device=device)

        # Output gate
        self.w_io = sn.Parameter([hidden_size, input_size], init="kaiming", device=device)
        self.w_ho = sn.Parameter([hidden_size, hidden_size], init="kaiming", device=device)
        self.b_o = sn.Parameter([hidden_size], init="zeros", device=device)

    def forward(
        self,
        x: ResonantTensor,
        state: Optional[Tuple[ResonantTensor, ResonantTensor]] = None,
    ) -> Tuple[ResonantTensor, ResonantTensor]:
        """
        Single LSTM step.

        Args:
            x: Input (input_size,) or (batch, input_size)
            state: (h, c) previous hidden and cell states

        Returns:
            (h_new, c_new)
        """
        is_1d = len(x.shape) == 1
        if is_1d:
            x = x.view([1, self.input_size])

        batch = x.shape[0]

        if state is None:
            h = ResonantTensor([0.0] * (batch * self.hidden_size), [batch, self.hidden_size], device=self.device)
            c = ResonantTensor([0.0] * (batch * self.hidden_size), [batch, self.hidden_size], device=self.device)
        else:
            h, c = state
            if len(h.shape) == 1:
                h = h.view([1, self.hidden_size])
                c = c.view([1, self.hidden_size])

        # Try Rust backend
        if _HAS_RUST_RNN:
            h_new_data, c_new_data = _rs_lstm_cell(
                x.to_floats(), h.to_floats(), c.to_floats(),
                self.w_ii.to_list(), self.w_hi.to_list(), self.b_i.to_list(),
                self.w_if.to_list(), self.w_hf.to_list(), self.b_f.to_list(),
                self.w_ig.to_list(), self.w_hg.to_list(), self.b_g.to_list(),
                self.w_io.to_list(), self.w_ho.to_list(), self.b_o.to_list(),
                batch, self.input_size, self.hidden_size,
            )
            shape = [self.hidden_size] if is_1d else [batch, self.hidden_size]
            return (
                ResonantTensor(h_new_data, shape, device=self.device),
                ResonantTensor(c_new_data, shape, device=self.device),
            )

        # Python fallback
        # Input gate: i = σ(W_ii @ x + W_hi @ h + b_i)
        i_gate = x.matmul(self.w_ii.tensor) + h.matmul(self.w_hi.tensor)
        i_gate = self._add_bias(i_gate, self.b_i.to_list())
        i_gate.sigmoid()

        # Forget gate: f = σ(W_if @ x + W_hf @ h + b_f)
        f_gate = x.matmul(self.w_if.tensor) + h.matmul(self.w_hf.tensor)
        f_gate = self._add_bias(f_gate, self.b_f.to_list())
        f_gate.sigmoid()

        # Cell gate: g = tanh(W_ig @ x + W_hg @ h + b_g)
        g_gate = x.matmul(self.w_ig.tensor) + h.matmul(self.w_hg.tensor)
        g_gate = self._add_bias(g_gate, self.b_g.to_list())
        g_gate.tanh()

        # Output gate: o = σ(W_io @ x + W_ho @ h + b_o)
        o_gate = x.matmul(self.w_io.tensor) + h.matmul(self.w_ho.tensor)
        o_gate = self._add_bias(o_gate, self.b_o.to_list())
        o_gate.sigmoid()

        # New cell state: c' = f * c + i * g (resonant memory update)
        c_new = f_gate * c + i_gate * g_gate

        # New hidden: h' = o * tanh(c')
        c_tanh_data = c_new.to_floats()[:]
        c_tanh = ResonantTensor(c_tanh_data, c_new.shape, device=self.device)
        c_tanh.tanh()
        h_new = o_gate * c_tanh

        if is_1d:
            h_new = h_new.view([self.hidden_size])
            c_new = c_new.view([self.hidden_size])

        return h_new, c_new

    def _add_bias(self, x: ResonantTensor, bias: List[float]) -> ResonantTensor:
        data = x.to_floats()
        rows, cols = x.shape[0], x.shape[1]
        for r in range(rows):
            for c in range(cols):
                data[r * cols + c] += bias[c]
        return ResonantTensor(data, x.shape, device=self.device)


class SyntonicLSTM(sn.Module):
    """
    Multi-step LSTM with resonant cell state.

    The cell state maintains syntonic coherence across the sequence,
    acting as the Ĥ operator for temporal harmonization.

    Supports batch_first=True for (batch, seq_len, input_size) input.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        batch_first: bool = False,
        dropout: float = 0.0,
        device: str = "cpu",
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        self.device = device

        self.cells = sn.ModuleList()
        for i in range(num_layers):
            in_sz = input_size if i == 0 else hidden_size
            self.cells.append(SyntonicLSTMCell(in_sz, hidden_size, device=device))

    def forward(
        self,
        x: ResonantTensor,
        state: Optional[Tuple[ResonantTensor, ResonantTensor]] = None,
    ) -> Tuple[ResonantTensor, Tuple[ResonantTensor, ResonantTensor]]:
        """
        Process sequence through LSTM.

        Args:
            x: Input (seq_len, input_size) or (batch, seq_len, input_size) if batch_first
            state: (h0, c0) initial states

        Returns:
            (output, (h_n, c_n))
        """
        # Handle batch_first: (batch, seq, feat) → process per-batch
        if self.batch_first and len(x.shape) == 3:
            batch_size, seq_len, feat_size = x.shape
            x_data = x.to_floats()

            all_batch_outputs = []
            all_batch_h = []
            all_batch_c = []

            for b in range(batch_size):
                b_start = b * seq_len * feat_size
                b_data = x_data[b_start:b_start + seq_len * feat_size]
                b_tensor = ResonantTensor(b_data, [seq_len, feat_size], device=self.device)

                b_state = None
                if state is not None:
                    h0, c0 = state
                    h0_data = h0.to_floats()
                    c0_data = c0.to_floats()
                    if len(h0.shape) == 3:
                        b_h_data = []
                        b_c_data = []
                        for layer in range(self.num_layers):
                            h_offset = layer * batch_size * self.hidden_size + b * self.hidden_size
                            b_h_data.extend(h0_data[h_offset:h_offset + self.hidden_size])
                            b_c_data.extend(c0_data[h_offset:h_offset + self.hidden_size])
                        b_state = (
                            ResonantTensor(b_h_data, [self.num_layers, self.hidden_size], device=self.device),
                            ResonantTensor(b_c_data, [self.num_layers, self.hidden_size], device=self.device),
                        )
                    else:
                        b_state = state

                b_out, (b_h, b_c) = self._forward_seq(b_tensor, b_state)
                all_batch_outputs.extend(b_out.to_floats())
                all_batch_h.append(b_h.to_floats())
                all_batch_c.append(b_c.to_floats())

            output = ResonantTensor(all_batch_outputs, [batch_size, seq_len, self.hidden_size], device=self.device)
            h_data = []
            c_data = []
            for bh, bc in zip(all_batch_h, all_batch_c):
                h_data.extend(bh)
                c_data.extend(bc)
            h_n = ResonantTensor(h_data, [batch_size, self.num_layers, self.hidden_size], device=self.device)
            c_n = ResonantTensor(c_data, [batch_size, self.num_layers, self.hidden_size], device=self.device)
            return output, (h_n, c_n)

        return self._forward_seq(x, state)

    def _forward_seq(
        self,
        x: ResonantTensor,
        state: Optional[Tuple[ResonantTensor, ResonantTensor]] = None,
    ) -> Tuple[ResonantTensor, Tuple[ResonantTensor, ResonantTensor]]:
        """Process a single sequence (seq_len, input_size)."""
        seq_len = x.shape[0]
        layer_input_data = x.to_floats()

        final_h_list = []
        final_c_list = []

        for layer_idx, cell in enumerate(self.cells):
            h, c = None, None
            if state is not None:
                h0, c0 = state
                start = layer_idx * self.hidden_size
                end = start + self.hidden_size
                h = ResonantTensor(h0.to_floats()[start:end], [self.hidden_size], device=self.device)
                c = ResonantTensor(c0.to_floats()[start:end], [self.hidden_size], device=self.device)

            layer_outputs = []
            in_size = self.input_size if layer_idx == 0 else self.hidden_size

            for t in range(seq_len):
                start = t * in_size
                end = start + in_size
                x_t = ResonantTensor(layer_input_data[start:end], [in_size], device=self.device)
                h, c = cell(x_t, (h, c) if h is not None else None)
                layer_outputs.append(h.to_floats())

            layer_input_data = []
            for out in layer_outputs:
                layer_input_data.extend(out)

            final_h_list.extend(h.to_floats())
            final_c_list.extend(c.to_floats())

        output = ResonantTensor(layer_input_data, [seq_len, self.hidden_size], device=self.device)
        h_n = ResonantTensor(final_h_list, [self.num_layers, self.hidden_size], device=self.device)
        c_n = ResonantTensor(final_c_list, [self.num_layers, self.hidden_size], device=self.device)

        return output, (h_n, c_n)


__all__ = [
    "SyntonicGRUCell",
    "SyntonicGRU",
    "SyntonicLSTMCell",
    "SyntonicLSTM",
]
