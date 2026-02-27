"""
Syntonic Convolution Layers — Pure SRT implementations.

Convolution as the D̂ (differentiation) operator: kernels discover
local patterns with φ-scaled initialization.

Includes:
- SyntonicConv1d: 1D convolution with DHSR processing
- SyntonicConv2d: 2D convolution with DHSR processing
- SyntonicConvTranspose2d: 2D transposed convolution (upsampling path)

Source: CRT.md §12.2
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from srt_library.core import sn
from srt_library.core.nn.resonant_tensor import ResonantTensor

PHI = (1 + math.sqrt(5)) / 2


def _conv1d_pure(
    x: List[float],
    seq_len: int,
    in_channels: int,
    kernel: List[float],
    kernel_size: int,
    out_channels: int,
    stride: int = 1,
    padding: int = 0,
) -> Tuple[List[float], int]:
    """Pure Python 1D convolution."""
    out_len = (seq_len + 2 * padding - kernel_size) // stride + 1
    output = []

    for oc in range(out_channels):
        for i in range(out_len):
            val = 0.0
            pos = i * stride - padding
            for k in range(kernel_size):
                input_pos = pos + k
                if 0 <= input_pos < seq_len:
                    for ic in range(in_channels):
                        x_idx = input_pos * in_channels + ic
                        k_idx = oc * (in_channels * kernel_size) + ic * kernel_size + k
                        if x_idx < len(x) and k_idx < len(kernel):
                            val += x[x_idx] * kernel[k_idx]
            output.append(val)

    return output, out_len


def _conv2d_pure(
    x: List[float],
    batch: int,
    h: int,
    w: int,
    in_channels: int,
    kernel: List[float],
    kernel_size: int,
    out_channels: int,
    stride: int = 1,
    padding: int = 0,
) -> Tuple[List[float], List[int]]:
    """Pure Python 2D convolution (NHWC layout)."""
    out_h = (h + 2 * padding - kernel_size) // stride + 1
    out_w = (w + 2 * padding - kernel_size) // stride + 1
    output = []

    for b in range(batch):
        for oh in range(out_h):
            for ow in range(out_w):
                for oc in range(out_channels):
                    val = 0.0
                    for kh in range(kernel_size):
                        for kw in range(kernel_size):
                            ih = oh * stride - padding + kh
                            iw = ow * stride - padding + kw
                            if 0 <= ih < h and 0 <= iw < w:
                                for ic in range(in_channels):
                                    x_idx = b * (h * w * in_channels) + ih * (w * in_channels) + iw * in_channels + ic
                                    k_idx = oc * (kernel_size * kernel_size * in_channels) + kh * (kernel_size * in_channels) + kw * in_channels + ic
                                    if x_idx < len(x) and k_idx < len(kernel):
                                        val += x[x_idx] * kernel[k_idx]
                    output.append(val)

    return output, [batch, out_h, out_w, out_channels]


class SyntonicConv1d(sn.Module):
    """
    1D convolution with DHSR processing.

    The kernel acts as the D̂ operator, discovering local sequential
    patterns with φ-scaled initialization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = True,
        precision: int = 100,
        device: str = "cpu",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.precision = precision
        self.device = device

        self.kernel = sn.Parameter(
            [out_channels, in_channels, kernel_size], init="kaiming", device=device
        )
        self.bias_param = sn.Parameter([out_channels], init="zeros", device=device) if bias else None
        self._syntony: Optional[float] = None

    def forward(self, x: ResonantTensor) -> ResonantTensor:
        """Forward: (seq_len, in_channels) -> (out_len, out_channels)."""
        x_data = x.to_floats()
        shape = x.shape
        seq_len = shape[0]

        kernel_data = self.kernel.to_list()
        output_data, out_len = _conv1d_pure(
            x_data, seq_len, self.in_channels,
            kernel_data, self.kernel_size, self.out_channels,
            self.stride, self.padding,
        )

        if self.bias_param is not None:
            bias_data = self.bias_param.to_list()
            for i in range(len(output_data)):
                oc = i % self.out_channels
                output_data[i] += bias_data[oc]

        # Transpose from (out_channels, out_len) to (out_len, out_channels)
        reshaped = []
        for i in range(out_len):
            for oc in range(self.out_channels):
                reshaped.append(output_data[oc * out_len + i])

        output = ResonantTensor(reshaped, [out_len, self.out_channels], device=self.device)
        self._syntony = output.syntony
        return output

    @property
    def syntony(self) -> Optional[float]:
        return self._syntony


class SyntonicConv2d(sn.Module):
    """
    2D convolution with DHSR processing.

    Uses either Rust backend (if available) or pure Python fallback.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = True,
        precision: int = 100,
        device: str = "cpu",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.precision = precision
        self.device = device

        self.kernel = sn.Parameter(
            [out_channels, kernel_size, kernel_size, in_channels],
            init="kaiming", device=device,
        )
        self.bias_param = sn.Parameter([out_channels], init="zeros", device=device) if bias else None
        self._syntony: Optional[float] = None

        # Try to use Rust backend
        self._use_rust = False
        try:
            from srt_library.core._core import py_conv2d
            self._use_rust = True
        except ImportError:
            pass

    def forward(self, x: ResonantTensor) -> ResonantTensor:
        """Forward: (batch, H, W, C) or (H, W, C) -> (batch, oH, oW, out_channels)."""
        x_data = x.to_floats()
        shape = x.shape

        if len(shape) == 3:
            h, w, c = shape
            batch = 1
            input_shape = [1, h, w, c]
        else:
            batch, h, w, c = shape
            input_shape = list(shape)

        kernel_data = self.kernel.to_list()
        kernel_shape = [self.out_channels, self.kernel_size, self.kernel_size, self.in_channels]

        if self._use_rust:
            from srt_library.core._core import py_conv2d
            output_data, out_shape = py_conv2d(
                x_data, input_shape, kernel_data, kernel_shape,
                (self.stride, self.stride), (self.padding, self.padding),
            )
        else:
            output_data, out_shape = _conv2d_pure(
                x_data, batch, h, w, self.in_channels,
                kernel_data, self.kernel_size, self.out_channels,
                self.stride, self.padding,
            )

        if self.bias_param is not None:
            bias_data = self.bias_param.to_list()
            _, oh, ow, oc = out_shape
            for b in range(batch):
                for i in range(oh):
                    for j in range(ow):
                        for c_idx in range(oc):
                            idx = b * (oh * ow * oc) + i * (ow * oc) + j * oc + c_idx
                            output_data[idx] += bias_data[c_idx]

        output = ResonantTensor(output_data, out_shape, device=self.device)
        self._syntony = output.syntony
        return output

    @property
    def syntony(self) -> Optional[float]:
        return self._syntony


class SyntonicConvTranspose2d(sn.Module):
    """
    2D transposed (deconvolution) layer with DHSR structure.

    Used for upsampling in generators/decoders. The Ĥ operator
    rebuilds spatial coherence from compressed representations.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        output_padding: int = 1,
        bias: bool = True,
        precision: int = 100,
        device: str = "cpu",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.precision = precision
        self.device = device

        # Kernel: [in_channels, kernel_size, kernel_size, out_channels]
        # (reversed from Conv2d since we scatter rather than gather)
        self.kernel = sn.Parameter(
            [in_channels, kernel_size, kernel_size, out_channels],
            init="kaiming", device=device,
        )
        self.bias_param = sn.Parameter([out_channels], init="zeros", device=device) if bias else None
        self._syntony: Optional[float] = None

    def forward(self, x: ResonantTensor) -> ResonantTensor:
        """
        Transposed convolution forward pass.

        Args:
            x: Input (batch, H, W, in_channels) or (H, W, in_channels)

        Returns:
            Output with upsampled spatial dimensions.
        """
        x_data = x.to_floats()
        shape = x.shape

        squeeze_batch = len(shape) == 3
        if len(shape) == 3:
            h, w, c = shape
            batch = 1
        else:
            batch, h, w, c = shape

        # Compute output spatial dimensions
        out_h = (h - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        out_w = (w - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding

        # Initialize output
        out_size = batch * out_h * out_w * self.out_channels
        output_data = [0.0] * out_size

        kernel_data = self.kernel.to_list()

        # Transposed conv: scatter input values through kernel
        for b in range(batch):
            for ih in range(h):
                for iw in range(w):
                    for ic in range(c):
                        x_val = x_data[b * (h * w * c) + ih * (w * c) + iw * c + ic]

                        for kh in range(self.kernel_size):
                            for kw in range(self.kernel_size):
                                oh = ih * self.stride - self.padding + kh
                                ow_pos = iw * self.stride - self.padding + kw

                                if 0 <= oh < out_h and 0 <= ow_pos < out_w:
                                    for oc in range(self.out_channels):
                                        k_idx = (ic * (self.kernel_size * self.kernel_size * self.out_channels)
                                                 + kh * (self.kernel_size * self.out_channels)
                                                 + kw * self.out_channels + oc)
                                        o_idx = (b * (out_h * out_w * self.out_channels)
                                                 + oh * (out_w * self.out_channels)
                                                 + ow_pos * self.out_channels + oc)
                                        if k_idx < len(kernel_data) and o_idx < out_size:
                                            output_data[o_idx] += x_val * kernel_data[k_idx]

        if self.bias_param is not None:
            bias_data = self.bias_param.to_list()
            for b in range(batch):
                for i in range(out_h):
                    for j in range(out_w):
                        for oc in range(self.out_channels):
                            idx = b * (out_h * out_w * self.out_channels) + i * (out_w * self.out_channels) + j * self.out_channels + oc
                            output_data[idx] += bias_data[oc]

        out_shape = [batch, out_h, out_w, self.out_channels]
        if squeeze_batch:
            out_shape = [out_h, out_w, self.out_channels]
            output_data = output_data[:out_h * out_w * self.out_channels]

        output = ResonantTensor(output_data, out_shape, device=self.device)
        self._syntony = output.syntony
        return output

    @property
    def syntony(self) -> Optional[float]:
        return self._syntony


__all__ = [
    "SyntonicConv1d",
    "SyntonicConv2d",
    "SyntonicConvTranspose2d",
]
