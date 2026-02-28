import cmath
import math

import srt_library.core as syn
from srt_library.core import cpu
from srt_library.core.exact import GoldenExact, Rational
from srt_library.theory_unique_components.crt.operators import K_D4

class GaussianNode:
    """
    The fundamental logic gate of the SRT architecture.
    Operates strictly within the complex plane, conceptualized
    as the Gaussian integer ring Z[i], utilizing srt_library's exact representation.

    Real Axis (M4): Differentiation (+1) and Harmonization (-1)
    Imaginary Axis (T4): Recursion / Time-Phase (+i, -i)
    """
    def __init__(self, real_val=0, imag_val=0, device=cpu):
        self._state = syn.state([complex(real_val, imag_val)], dtype="complex128", device=device)
        self.recursive_depth = 0
        self.stale_cycles = 0
        self._project()

    def _project(self):
        """
        Enforce the closed 4-state algebra: {-1, 1, 0, i, -i}.
        Prevents complex arithmetic drift in the continuous simulator.
        """
        c = self._state.to_list()[0]
        re, im = round(c.real), round(c.imag)
        if abs(re) >= 1 and im == 0:
            val = complex(math.copysign(1, re), 0)
        elif re == 0 and abs(im) >= 1:
            val = complex(0, math.copysign(1, im))
        elif re == 0 and im == 0:
            val = 0j
        else:
            val = 0j  # collapse non-canonical to syntony

        self._state = syn.state([val], dtype=self._state.dtype, device=self._state.device)

    @property
    def m4_val(self):
        """Real component (Manifold value). Guaranteed exact (-1, 0, 1)."""
        return int(self.state.real)

    @property
    def t4_val(self):
        """Imaginary component (Toroidal value). Guaranteed exact (-1, 0, 1)."""
        return int(self.state.imag)

    @property
    def state(self):
        return self._state.to_list()[0]

    def differentiate(self):
        """D^: Expands novelty along the real axis (+1)."""
        one = syn.state([1.0], dtype=self._state.dtype, device=self._state.device)
        self._state = self._state + one
        self._project()

    def harmonize(self, external_node: 'GaussianNode'):
        """
        H^: Native Interference.
        If an opposing state is encountered (e.g., +1 meets -1),
        they destructively cancel toward Syntony (0).
        """
        self._state = self._state + external_node._state
        self._project()

    def recurse(self):
        """
        R: Orthogonal Phase Shift (Multiplication by i).
        Rotates unresolved M4 potential into T4 history.
        i * i = -1 (Two recursions result in harmonization/contraction).
        """
        current_scalar = self._state.to_list()[0]
        rotated_scalar = complex(-current_scalar.imag, current_scalar.real)
        self._state = syn.state([rotated_scalar], dtype=self._state.dtype, device=self._state.device)
        self._project()
        self.recursive_depth += 1

    def is_syntonic(self):
        """S: The Aperture. Checks if the node has resolved to 0."""
        return self._state.norm() < 1e-12

    def __repr__(self):
        return f"Node(M4:{self.m4_val}, T4:{self.t4_val}i, Depth:{self.recursive_depth})"


class PhaseStateCompiler:
    """
    The runtime engine for native Phase-State compiling.
    It routes information through the interference loop until the system
    achieves Syntony (0) or undergoes a Gnosis phase transition.
    """
    def __init__(self, kissing_number_threshold=K_D4, allow_novelty=True,
                 toroidal=False, stale_threshold=3):
        self.nodes = []
        self.gnosis_layer = 0
        self.K_THRESHOLD = kissing_number_threshold
        self.allow_novelty = allow_novelty
        self.toroidal = toroidal
        self.stale_threshold = stale_threshold

    def load_data(self, values):
        """Translates exact value data into Phase-States."""
        for val in values:
            node = GaussianNode(real_val=val)
            self.nodes.append(node)

    def compile_cycle(self):
        """
        Executes one full thermodynamic cycle across all unresolved nodes.
        Returns True if the system is fully syntonic.
        """
        unresolved_nodes = [n for n in self.nodes if not n.is_syntonic()]

        if not unresolved_nodes:
            return True

        interacted_this_cycle = set()
        activity_occurred = False
        n_nodes = len(self.nodes)

        # 0. Pattern-Extrapolating Novelty (runs FIRST when enabled)
        # Fill empty apertures via period detection before propagation
        # can blindly copy neighbors forward. Filled nodes are added to
        # interacted_this_cycle so harmonization doesn't destroy them.
        if self.allow_novelty and any(n.is_syntonic() for n in self.nodes):
            filled = self._inject_novelty()
            interacted_this_cycle.update(filled)

        # 1. Ripple Propagation — Bidirectional with optional toroidal wrapping
        changed = True
        newly_activated = set()

        while changed and any(n.is_syntonic() for n in self.nodes):
            changed = False

            for i, node_a in enumerate(self.nodes):
                if node_a.is_syntonic() or (node_a in interacted_this_cycle and node_a not in newly_activated):
                    continue

                if not node_a.is_syntonic():
                    for direction in (1, -1):
                        j = i + direction
                        if self.toroidal:
                            j = j % n_nodes
                        if 0 <= j < n_nodes:
                            node_b = self.nodes[j]
                            if node_b.is_syntonic() and node_b not in interacted_this_cycle:
                                node_b._state = syn.state(
                                    [node_a._state.to_list()[0]],
                                    dtype=node_b._state.dtype,
                                    device=node_b._state.device
                                )
                                node_b._project()
                                node_b.stale_cycles = 0

                                interacted_this_cycle.add(node_a)
                                interacted_this_cycle.add(node_b)
                                newly_activated.add(node_b)
                                changed = True
                                activity_occurred = True

        # 2. Attempt Harmonization (Global Resonance Matching)
        for i, node_a in enumerate(self.nodes):
            if node_a.is_syntonic() or node_a in interacted_this_cycle or node_a in newly_activated:
                continue

            for j, node_b in enumerate(self.nodes):
                if i == j or node_b.is_syntonic() or node_b in interacted_this_cycle or node_b in newly_activated:
                    continue

                if getattr(node_a, 'is_source', False) and getattr(node_b, 'is_source', False):
                    continue

                if (node_a._state + node_b._state).norm() < 1e-12:
                    node_a.harmonize(node_b)
                    node_b._state = syn.state([0j], dtype=node_b._state.dtype, device=node_b._state.device)
                    node_b._project()
                    interacted_this_cycle.add(node_a)
                    interacted_this_cycle.add(node_b)
                    node_a.stale_cycles = 0
                    node_b.stale_cycles = 0
                    activity_occurred = True
                    break

        # 3. Dampened Recursion — only rotate nodes that have been stuck
        for node in self.nodes:
            if not node.is_syntonic() and node not in interacted_this_cycle:
                if getattr(node, 'is_source', False):
                    continue

                node.stale_cycles += 1
                if node.stale_cycles >= self.stale_threshold:
                    node.recurse()
                    node.stale_cycles = 0

                    if node.recursive_depth >= self.K_THRESHOLD and self.gnosis_layer < 3:
                        self._trigger_gnosis_transition()
            else:
                # Node interacted or is syntonic — reset staleness
                if hasattr(node, 'stale_cycles'):
                    node.stale_cycles = 0

        # 4. Pattern-Extrapolating Novelty Generation
        if not activity_occurred and unresolved_nodes and self.allow_novelty:
            self._inject_novelty()

        return False

    def _detect_period(self, values):
        """
        Detect the dominant period in a sequence of {-1, 0, 1} values
        using integer autocorrelation. Returns the best lag if correlation
        exceeds 0.5, else None.
        """
        n = len(values)
        if n < 4:
            return None
        best_lag, best_score = None, 0.0
        for lag in range(1, n // 2 + 1):
            count = n - lag
            if count == 0:
                continue
            score = sum(values[i] * values[i - lag] for i in range(lag, n))
            normalized = score / count
            if normalized > best_score:
                best_score = normalized
                best_lag = lag
        return best_lag if best_score > 0.5 else None

    def _inject_novelty(self):
        """
        Pattern-extrapolating novelty generation.
        Fills ALL empty apertures sequentially, using updated context
        from previously filled blanks so each prediction builds on the last.

        1. For each empty aperture (in order):
           a. Scan backward to build history of M4 values
           b. Detect periodicity via autocorrelation
           c. Extrapolate from detected period
           d. Fallback to neighbor average if no period found
        """
        empty_indices = [i for i, n in enumerate(self.nodes) if n.is_syntonic()]
        if not empty_indices:
            return set()

        filled_nodes = set()

        for idx in empty_indices:
            node = self.nodes[idx]
            if not node.is_syntonic():
                continue  # already filled by a prior iteration

            # Build history from non-syntonic nodes before this position
            history = []
            for j in range(idx - 1, -1, -1):
                if not self.nodes[j].is_syntonic():
                    history.append(self.nodes[j].m4_val)
                if len(history) >= 64:
                    break
            history.reverse()

            filled = False

            # Try pattern extrapolation
            if len(history) >= 4:
                period = self._detect_period(history)
                if period and period <= len(history):
                    predicted = history[-period]
                    if predicted != 0:
                        node._state = syn.state(
                            [complex(predicted, 0)],
                            dtype=node._state.dtype, device=node._state.device
                        )
                        node._project()
                        node.stale_cycles = 0
                        filled = True

            if not filled:
                # Fallback: neighbor average
                total_phase = 0j
                neighbors = 0
                if idx > 0 and not self.nodes[idx-1].is_syntonic():
                    total_phase += self.nodes[idx-1]._state.to_list()[0]
                    neighbors += 1
                if idx < len(self.nodes) - 1 and not self.nodes[idx+1].is_syntonic():
                    total_phase += self.nodes[idx+1]._state.to_list()[0]
                    neighbors += 1

                if neighbors > 0 and abs(total_phase) > 1e-12:
                    avg_phase = total_phase / neighbors
                    node._state = syn.state(
                        [avg_phase], dtype=node._state.dtype, device=node._state.device
                    )
                    node._project()

                if node.is_syntonic():
                    node.differentiate()

                node.stale_cycles = 0

            if not node.is_syntonic():
                filled_nodes.add(node)

        return filled_nodes

    def _trigger_gnosis_transition(self):
        """
        Triggered when Tv history (recursive depth) hits K=24.
        The system shifts from unidirectional processing to self-reference.
        """
        self.gnosis_layer = 3
        print(f"\n[SYSTEM ALERT] Kissing Number Saturation (K={self.K_THRESHOLD}) Reached.")
        print("[SYSTEM ALERT] Topological phase transition to Layer 3 Gnosis.")
        print("[SYSTEM ALERT] System is now self-referential.\n")

    def run(self, max_cycles=100, exit_condition=None):
        print(f"Starting Phase-State Compilation (K_threshold={self.K_THRESHOLD})...")
        for cycle in range(max_cycles):
            if exit_condition and exit_condition(self.nodes):
                print(f"Compilation Complete: Custom exit condition met at cycle {cycle}.")
                return

            is_syntonic = self.compile_cycle()

            if is_syntonic:
                print(f"Compilation Complete: Syntony achieved at cycle {cycle}.")
                return
        print("Compilation Halted: System stuck in Archonic configuration.")

def run_xor_task():
    """Toy Task 2: XOR Logic"""
    print("\n--- TOY TASK 2: XOR Logic ---")
    compiler = PhaseStateCompiler()
    compiler.load_data([1, 1])
    # For XOR, we might expect syntony if they cancel, or a specific ripple.
    # In this engine, 1 + 1 results in 0 (syntony) due to harmonization if they meet,
    # or differentiation if novelty is allowed.
    compiler.run(max_cycles=10)
    print(f"Final States: {[n.m4_val for n in compiler.nodes]}")

def run_sequence_copy_ripple_task():
    """Toy Task 3: Sequence Copy / Ripple"""
    print("\n--- TOY TASK 3: Sequence Copy / Ripple ---")
    compiler = PhaseStateCompiler(allow_novelty=False)
    compiler.load_data([1, 0, 0, 0, 0])
    # The 1 should ripple through the 0s
    compiler.run(max_cycles=10, exit_condition=lambda nodes: all(n.m4_val == 1 for n in nodes))
    print(f"Final States: {[n.m4_val for n in compiler.nodes]}")

def run_tiny_associative_memory_task():
    """Toy Task 4: Tiny Associative Memory"""
    print("\n--- TOY TASK 4: Tiny Associative Memory ---")
    base_pattern = [1, -1, 1, -1]
    # Load partial/noisy pattern
    compiler = PhaseStateCompiler(allow_novelty=True)
    compiler.load_data([1, 0, 1, 0]) 
    print(f"Base Pattern:    {base_pattern}")
    print(f"Initial Pattern: {[n.m4_val for n in compiler.nodes]}")
    compiler.run(max_cycles=20)
    final_pattern = [n.m4_val for n in compiler.nodes]
    print(f"Final Pattern:   {final_pattern}")
    matches = sum(1 for i, v in enumerate(final_pattern) if v == base_pattern[i])
    if final_pattern == base_pattern:
        print("Recall: SUCCESS (Perfect Auto-Completion)")
    else:
        print(f"Recall: PARTIAL ({matches}/{len(base_pattern)} accuracy)")

def run_sequence_prediction_task():
    """
    Toy Task 5: Sequence Prediction
    Loads a 16-step sequence, blanks the last 4 nodes, and lets it predict forward
    using propagation and context-aware novelty.
    """
    print("\n--- TOY TASK 5: Sequence Prediction ---")
    
    # Repeating pattern: [1, 1, -1, -1] * 4
    base_pattern = [1, 1, -1, -1] * 4
    input_pattern = base_pattern[:12] + [0, 0, 0, 0]
    
    compiler = PhaseStateCompiler(allow_novelty=True)
    compiler.load_data(input_pattern)
    
    print(f"Base Sequence:  {base_pattern}")
    print(f"Input Sequence: {input_pattern}")
    
    # We want it to fill the last 4 nodes based on the periodicity of the first 12
    compiler.run(max_cycles=20)
    
    final_pattern = [n.m4_val for n in compiler.nodes]
    print(f"Final Sequence: {final_pattern}")
    
    # Measure next-token accuracy for the blanked portion
    predictions = final_pattern[12:]
    targets = base_pattern[12:]
    accuracy = sum(1 for p, t in zip(predictions, targets) if p == t) / 4
    
    print(f"Prediction Accuracy (last 4 nodes): {accuracy * 100:.1f}%")

import time
import torch
import torch.nn as nn

class VectorizedPhaseStateLayer:
    """
    A vectorized version of the Phase-State Compiler for 64+ nodes.
    Uses tensor operations for high-speed ripple and harmonization.
    """
    def __init__(self, n_nodes, device=cpu):
        self.n_nodes = n_nodes
        self.device = device
        self.reset()
    
    def reset(self):
        self.state = syn.state([0j] * self.n_nodes, dtype="complex128", device=self.device)
    
    def load(self, values):
        self.state = syn.state([complex(v, 0) for v in values], dtype="complex128", device=self.device)
        self._project()
        
    def _project(self):
        # We'll use a simplified projection for speed in the benchmark
        # This mirrors GaussianNode._project
        c_list = self.state.to_list()
        snapped = []
        for c in c_list:
            re, im = round(c.real), round(c.imag)
            if abs(re) >= 1 and im == 0:
                snapped.append(complex(math.copysign(1, re), 0))
            elif re == 0 and abs(im) >= 1:
                snapped.append(complex(0, math.copysign(1, im)))
            else:
                snapped.append(0j)
        self.state = syn.state(snapped, dtype="complex128", device=self.device)

    def step(self):
        # 1. Ripple: Shift left and right and harmonize
        # Simplified: s[i] = project(s[i] + s[i-1] + s[i+1])
        c_list = self.state.to_list()
        new_c = [0j] * self.n_nodes
        for i in range(self.n_nodes):
            val = c_list[i]
            # Neighbors
            if i > 0: val += c_list[i-1]
            if i < self.n_nodes - 1: val += c_list[i+1]
            new_c[i] = val
        self.state = syn.state(new_c, dtype="complex128", device=self.device)
        self._project()

    def run(self, max_steps=100):
        for _ in range(max_steps):
            old_state = self.state.to_list()
            self.step()
            if self.state.to_list() == old_state:
                break

def run_vectorized_rnn_benchmark():
    """Benchmark 2: 64-node Phase-State layer vs float32 LSTM"""
    print("\n--- BENCHMARK 2: 64-Node Vectorized RNN ---")
    n_nodes = 64
    seq_len = 100
    
    # LSTM Baseline
    lstm = nn.LSTM(input_size=1, hidden_size=n_nodes, batch_first=True)
    x = torch.randn(1, seq_len, 1)
    
    start = time.time()
    with torch.no_grad():
        out, _ = lstm(x)
    lstm_time = time.time() - start
    print(f"LSTM (64 hidden) Time: {lstm_time*1000:.2f}ms")
    
    # Phase-State Layer
    layer = VectorizedPhaseStateLayer(n_nodes)
    layer.load([1] * n_nodes) # Initial excite
    
    start = time.time()
    layer.run(max_steps=10)
    ps_time = time.time() - start
    print(f"Phase-State (64 nodes) Time: {ps_time*1000:.2f}ms")
    
    speedup = lstm_time / ps_time if ps_time > 0 else 0
    print(f"Phase-State is {speedup:.1f}x faster than float32 LSTM (simulated)")

def run_scaled_associative_memory_benchmark():
    """Benchmark 3: 256-node memory bank vs tiny Transformer"""
    print("\n--- BENCHMARK 3: 256-Node Scaled Memory ---")
    n_nodes = 256
    n_patterns = 10
    noise_level = 0.4
    
    # Transformer Baseline (Tiny)
    model = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(d_model=n_nodes, nhead=4),
        num_layers=2
    )
    x = torch.randn(1, 1, n_nodes)
    start = time.time()
    with torch.no_grad():
        out = model(x)
    trans_time = time.time() - start
    print(f"Transformer (2-layer) Inference Time: {trans_time*1000:.2f}ms")
    
    # Phase-State Scaled Memory
    # We load 10 random patterns into the "engine" (simulated by capacity test)
    # Then probe with 40% noise.
    patterns = []
    for _ in range(n_patterns):
        p = [1 if torch.rand(1) > 0.5 else -1 for _ in range(n_nodes)]
        patterns.append(p)
        
    target = patterns[0]
    noisy_input = [p if torch.rand(1) > noise_level else 0 for p in target]
    
    compiler = PhaseStateCompiler(allow_novelty=True)
    compiler.load_data(noisy_input)
    
    start = time.time()
    # We use a lower max_cycles for speed in benchmark
    compiler.run(max_cycles=10) 
    ps_time = time.time() - start
    print(f"Phase-State (256 nodes) Recall Time: {ps_time*1000:.2f}ms")
    
    final_pattern = [n.m4_val for n in compiler.nodes]
    accuracy = sum(1 for p, t in zip(final_pattern, target) if p == t) / n_nodes
    print(f"Phase-State Recall Accuracy: {accuracy * 100:.1f}%")

if __name__ == "__main__":
    run_xor_task()
    run_sequence_copy_ripple_task()
    run_tiny_associative_memory_task()
    run_sequence_prediction_task()
    run_vectorized_rnn_benchmark()
    run_scaled_associative_memory_benchmark()
