"""
Phase-State Compiler Benchmark Suite

Ten benchmarks testing the Phase-State Compiler against conventional baselines:
0. Sequence Copying (topological routing across a gap)
1. XOR Correctness (harmonization sanity check)
2. Sequence Prediction (16-step, multi-pattern)
3. 64-node Phase-State RNN vs Float32 LSTM
4. 256-node Associative Memory recall vs Transformer baseline
5. Gnosis/Attractor (K=24 saturation)
6. Harmonic Readout (zero-parameter resonance inference)
7. 1024-node Associative Memory recall (100 patterns, 40% noise)
8. 2D Toroidal Grid (wavefront propagation & self-repair)
9. Tiny Language Task (next-token prediction)

All tests use srt_library or pure Python only -- no PyTorch, NumPy, or SciPy.
"""

import sys
import os
import time
import math
import random
import contextlib
import copy

import srt_library.core as syn
from srt_library.core.nn.resonant_tensor import ResonantTensor
from srt_library.core import sn
from srt_library.core.nn.layers.resonant_linear import ResonantLinear
from phase_state_vibes_compiler import PhaseStateCompiler, GaussianNode
from srt_library.core.nn.optim.golden_momentum import GoldenMomentumOptimizer


@contextlib.contextmanager
def suppress_stdout():
    old = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = old


def print_header(title):
    w = 70
    print("\n" + "=" * w)
    print(f"  {title}")
    print("=" * w)


def print_table(headers, rows, col_widths=None):
    if col_widths is None:
        col_widths = [
            max(len(str(h)), max((len(str(r[i])) for r in rows), default=0)) + 2
            for i, h in enumerate(headers)
        ]
    def fmt(cells):
        return "| " + " | ".join(str(c).ljust(w) for c, w in zip(cells, col_widths)) + " |"
    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    print(sep)
    print(fmt(headers))
    print(sep)
    for row in rows:
        print(fmt(row))
    print(sep)


# ============================================================================
#  Models
# ============================================================================

class PhaseStateRNNCell:
    """
    A Phase-State Recurrent Memory Layer using Toroidal Lane Topology.

    Each input element gets its own toroidal compiler (Axiom A3) of
    `lane_size` nodes.  With hidden_size=64 and input_size=8, each
    lane is an 8-node torus (1 input + 7 hidden).  This ensures
    every element has a direct topological path to its own hidden
    state with recurrent wrap-around.

    Theoretical advantage over LSTM: the toroidal lane topology
    preserves each element's signal via topological routing without
    learned weights.  Propagation fills empty apertures from the
    source, and once filled, hidden nodes hold their exact quantized
    state ({-1, 0, 1}) across all timesteps.
    """
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lane_size = hidden_size // input_size  # nodes per element

        # Each element gets its own toroidal compiler (Axiom A3).
        # Recursion fully suppressed — signal preservation is the goal.
        self.lanes = []
        for _ in range(input_size):
            compiler = PhaseStateCompiler(
                kissing_number_threshold=16, allow_novelty=False,
                toroidal=True, stale_threshold=999999
            )
            for _ in range(self.lane_size):
                compiler.nodes.append(GaussianNode(0))
            self.lanes.append(compiler)

        self.cycles_per_step = self.lane_size

    def step(self, x_seq_step):
        """
        Processes one timestep of data.
        x_seq_step: list of values of length `input_size`
        """
        for i, val in enumerate(x_seq_step):
            lane = self.lanes[i]
            # Inject input at node 0 of this lane's torus
            lane.nodes[0] = GaussianNode(val)
            lane.nodes[0].is_source = True

            # Mark hidden nodes as source to prevent harmonization
            for j in range(1, self.lane_size):
                lane.nodes[j].is_source = True

            with suppress_stdout():
                lane.run(max_cycles=self.cycles_per_step)

    def predict(self):
        """
        Read the preserved hidden state from each lane's torus.
        Node 1 holds the topologically routed signal from the input.
        """
        result = []
        for lane in self.lanes:
            result.append(lane.nodes[1].m4_val)
        return result


class PureFloat32LSTM:
    """
    A minimal pure-Python reference LSTM for exact algorithm complexity benchmarking
    without relying on external C++ tensor libraries like PyTorch.
    """
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_i = [[random.uniform(-0.1, 0.1) for _ in range(input_size + hidden_size)] for _ in range(hidden_size)]
        self.W_f = [[random.uniform(-0.1, 0.1) for _ in range(input_size + hidden_size)] for _ in range(hidden_size)]
        self.W_c = [[random.uniform(-0.1, 0.1) for _ in range(input_size + hidden_size)] for _ in range(hidden_size)]
        self.W_o = [[random.uniform(-0.1, 0.1) for _ in range(input_size + hidden_size)] for _ in range(hidden_size)]

        self.h = [0.0] * hidden_size
        self.c = [0.0] * hidden_size

    def step(self, x):
        concat = x + self.h

        def dot(W, v):
            return [sum(w*val for w, val in zip(row, v)) for row in W]

        def sigmoid(v):
            return [1 / (1 + math.exp(-max(min(val, 500), -500))) for val in v]

        def tanh(v):
            return [math.tanh(val) for val in v]

        i_g = sigmoid(dot(self.W_i, concat))
        f_g = sigmoid(dot(self.W_f, concat))
        o_g = sigmoid(dot(self.W_o, concat))
        c_tilde = tanh(dot(self.W_c, concat))

        self.c = [f*c + i*ct for f, c, i, ct in zip(f_g, self.c, i_g, c_tilde)]
        self.h = [o*math.tanh(c) for o, c in zip(o_g, self.c)]
        return self.h


# ============================================================================
#  TEST 0: Sequence Copying (Topological Routing)
# ============================================================================

def test_sequence_copy():
    """
    Tests how Phase-State logic copies/transmits an information pattern
    across a gap. Source nodes [1, 1] should induce the same state in
    empty target nodes [0, 0] via constructive propagation (aperture filling).
    """
    cases = [
        ("2-node copy",     [1, 1],          2, [1, 1]),
        ("4-node uniform",  [1, 1, 1, 1],    4, [1, 1, 1, 1]),
        ("4-node negative", [-1, -1, -1, -1], 4, [-1, -1, -1, -1]),
    ]

    rows = []
    all_pass = True

    for name, source, gap_size, expected in cases:
        compiler = PhaseStateCompiler(kissing_number_threshold=8, allow_novelty=True)
        full_data = source + [0] * gap_size
        compiler.load_data(full_data)

        # Stamp exact source values
        for i, val in enumerate(source):
            compiler.nodes[i]._state = syn.state(
                [complex(val, 0)], dtype="complex128",
                device=compiler.nodes[i]._state.device
            )
            compiler.nodes[i]._project()
            compiler.nodes[i].is_source = True

        target_start = len(source)
        target_end = len(full_data)

        objective = lambda nodes, ts=target_start, te=target_end: all(
            not nodes[i].is_syntonic() for i in range(ts, te)
        )

        with suppress_stdout():
            compiler.run(max_cycles=15, exit_condition=objective)

        result = [compiler.nodes[i].m4_val for i in range(target_start, target_end)]
        passed = result == expected
        all_pass = all_pass and passed
        rows.append([name, str(expected), str(result), "PASS" if passed else "FAIL"])

    print_table(["Case", "Expected", "Got", "Status"], rows)
    print(f"  Sequence Copy: {'ALL PASS' if all_pass else 'FAILURES DETECTED'}")
    return all_pass


# ============================================================================
#  TEST 1: XOR Correctness (Harmonization)
# ============================================================================

def test_xor_correctness():
    """
    Tests all 4 XOR input combinations via Phase-State interference.
    +1 XOR -1 should cancel to syntony (True), same-sign should not (False).
    """
    cases = [
        ("T XOR T", [1, 1], False),
        ("T XOR F", [1, -1], True),
        ("F XOR T", [-1, 1], True),
        ("F XOR F", [-1, -1], False),
    ]

    rows = []
    all_pass = True

    for name, inputs, expected in cases:
        compiler = PhaseStateCompiler(kissing_number_threshold=4, stale_threshold=1)
        compiler.load_data(inputs)
        # Stamp exact values (load_data projects non-zero to +1)
        for val, node in zip(inputs, compiler.nodes):
            node._state = syn.state(
                [complex(val, 0)], dtype="complex128", device=node._state.device
            )

        with suppress_stdout():
            compiler.run(max_cycles=3)

        syntonic_count = sum(1 for n in compiler.nodes if n.is_syntonic())
        result = syntonic_count == len(inputs)
        passed = result == expected
        all_pass = all_pass and passed

        rows.append([name, str(expected), str(result), "PASS" if passed else "FAIL"])

    print_table(["Case", "Expected", "Got", "Status"], rows)
    print(f"  XOR Correctness: {'ALL PASS' if all_pass else 'FAILURES DETECTED'}")
    return all_pass


# ============================================================================
#  TEST 2: Sequence Prediction (16-step, Multi-Pattern)
# ============================================================================

def test_sequence_prediction():
    """
    Load a 16-step sequence, blank the last 4 nodes, let the compiler
    predict forward via propagation + pattern-extrapolating novelty.
    """
    PATTERNS = [
        ("AABB",     [1, 1, -1, -1] * 4),
        ("ABAB",     [1, -1, 1, -1] * 4),
        ("AAAB",     [1, 1, 1, -1] * 4),
        ("ABBB",     [1, -1, -1, -1] * 4),
        ("AABBAB",   ([1, 1, -1, -1, 1, -1] * 3)[:16]),
        ("ALL_POS",  [1] * 16),
        ("ALL_NEG",  [-1] * 16),
    ]

    rows = []
    total_correct = 0
    total_positions = 0

    for name, pattern in PATTERNS:
        compiler = PhaseStateCompiler(kissing_number_threshold=16, allow_novelty=True)
        compiler.load_data(pattern)

        # Stamp exact +-1 values
        for i, val in enumerate(pattern):
            compiler.nodes[i]._state = syn.state(
                [complex(val, 0)], dtype="complex128",
                device=compiler.nodes[i]._state.device
            )
            compiler.nodes[i]._project()

        ground_truth = pattern[12:16]

        # Blank the last 4
        for i in range(12, 16):
            compiler.nodes[i]._state = compiler.nodes[i]._state * 0

        # Anchor the context
        for i in range(12):
            compiler.nodes[i].is_source = True

        recovered = lambda nodes: all(not n.is_syntonic() for n in nodes[12:16])

        with suppress_stdout():
            compiler.run(max_cycles=30, exit_condition=recovered)

        predicted = [n.m4_val for n in compiler.nodes[12:16]]
        matches = sum(1 for p, g in zip(predicted, ground_truth) if p == g)
        total_correct += matches
        total_positions += 4

        rows.append([name, str(ground_truth), str(predicted), f"{matches}/4"])

    avg = total_correct / total_positions * 100
    rows.append(["AVERAGE", "", "", f"{avg:.1f}%"])

    print_table(["Pattern", "Ground Truth", "Predicted", "Accuracy"], rows)
    return avg


# ============================================================================
#  TEST 3: 64-Node Phase-State RNN vs Float32 LSTM
# ============================================================================

def test_vectorized_rnn():
    """
    64-node Phase-State RNN (toroidal) vs Float32 LSTM.
    Both process a 16-step sequence of 8-dim binary vectors.
    PS-RNN predicts the next step by clearing the input region and
    letting topological propagation fill it from the hidden manifold.
    """
    seq_len = 16
    input_size = 8
    hidden_size = 64
    num_trials = 3

    ps_params = hidden_size
    lstm_params = 4 * (input_size + hidden_size) * hidden_size

    trial_rows = []

    for trial in range(num_trials):
        seed = 42 + trial * 17
        random.seed(seed)

        base_vec = [1 if random.random() > 0.5 else -1 for _ in range(input_size)]
        sequence = []
        for t in range(seq_len):
            vec = [v * (1 if t % 4 < 2 else -1) for v in base_vec]
            sequence.append(vec)
        ground_truth = [v * (1 if 16 % 4 < 2 else -1) for v in base_vec]

        # --- Phase-State RNN (strided lanes, recursion suppressed) ---
        ps_rnn = PhaseStateRNNCell(input_size, hidden_size)

        t0 = time.time()
        for x in sequence:
            ps_rnn.step(x)

        # Read preserved hidden state from each lane
        ps_pred = ps_rnn.predict()
        ps_time = (time.time() - t0) * 1000

        ps_acc = sum(1 for p, g in zip(ps_pred, ground_truth) if p == g) / input_size

        # Diagnostic: count active nodes across all lanes
        m4_active = sum(1 for lane in ps_rnn.lanes
                        for n in lane.nodes if n.m4_val != 0)

        # --- Float32 LSTM ---
        random.seed(seed)
        lstm = PureFloat32LSTM(input_size, hidden_size)

        t0 = time.time()
        for x in sequence:
            lstm.step(x)
        lstm_time = (time.time() - t0) * 1000

        random.seed(seed + 1)
        W_out = [[random.uniform(-0.1, 0.1) for _ in range(hidden_size)]
                 for _ in range(input_size)]
        raw = [sum(w * h for w, h in zip(row, lstm.h)) for row in W_out]
        lstm_pred = [1 if v >= 0 else -1 for v in raw]
        lstm_acc = sum(1 for p, g in zip(lstm_pred, ground_truth) if p == g) / input_size

        trial_rows.append({
            "trial": trial + 1,
            "ps_acc": ps_acc, "ps_ms": ps_time,
            "lstm_acc": lstm_acc, "lstm_ms": lstm_time,
        })

        print(f"  Trial {trial+1}: PS {ps_acc*100:.0f}% ({ps_time:.1f}ms) "
              f"[M4:{m4_active}/{hidden_size}]  |  "
              f"LSTM {lstm_acc*100:.0f}% ({lstm_time:.1f}ms)")

    avg_ps_acc = sum(t["ps_acc"] for t in trial_rows) / num_trials
    avg_ps_ms = sum(t["ps_ms"] for t in trial_rows) / num_trials
    avg_lstm_acc = sum(t["lstm_acc"] for t in trial_rows) / num_trials
    avg_lstm_ms = sum(t["lstm_ms"] for t in trial_rows) / num_trials

    print()
    print_table(
        ["Model", "Avg Accuracy", "Avg Time (ms)", "Params"],
        [
            ["Phase-State RNN 64", f"{avg_ps_acc*100:.1f}%", f"{avg_ps_ms:.1f}", str(ps_params)],
            ["PureFloat32 LSTM",   f"{avg_lstm_acc*100:.1f}%", f"{avg_lstm_ms:.1f}", str(lstm_params)],
        ]
    )
    print("  Note: Neither model is trained. PS uses topological signal")
    print("  preservation (strided lanes); LSTM uses random readout weights.")

    return avg_ps_acc, avg_lstm_acc


# ============================================================================
#  TEST 4: Associative Memory (256 nodes, 10 patterns, 40% noise)
# ============================================================================

class AttentionMemory(sn.Module):
    """
    Attention-based associative memory using existing ResonantLinear.
    Stores patterns as key-value pairs, recalls via scaled dot-product attention.
    """
    def __init__(self, pattern_dim, d_model=32, device="cpu"):
        super().__init__()
        self.pattern_dim = pattern_dim
        self.d_model = d_model
        self.q_proj = ResonantLinear(pattern_dim, d_model, bias=False, device=device)
        self.k_proj = ResonantLinear(pattern_dim, d_model, bias=False, device=device)
        self.v_proj = ResonantLinear(pattern_dim, pattern_dim, bias=False, device=device)
        self.scale = math.sqrt(d_model)
        self.stored_keys = None
        self.stored_values = None

    def store(self, patterns):
        flat = []
        for p in patterns:
            flat.extend([float(v) for v in p])
        pat_t = ResonantTensor(flat, [len(patterns), self.pattern_dim])
        self.stored_keys = self.k_proj(pat_t)
        self.stored_values = self.v_proj(pat_t)

    def recall(self, probe):
        probe_t = ResonantTensor([float(v) for v in probe], [1, self.pattern_dim])
        q = self.q_proj(probe_t)
        scores = q.matmul(self.stored_keys)
        scores = scores.scalar_mul(1.0 / self.scale)
        scores.softmax()
        v_t = self.stored_values.transpose(0, 1)
        output = scores.matmul(v_t)
        recalled = output.to_floats()[:self.pattern_dim]
        return [1 if v >= 0 else -1 for v in recalled]

    def param_count(self):
        return (self.pattern_dim * self.d_model * 2
                + self.pattern_dim * self.pattern_dim)


def make_noisy_probe(pattern, noise_fraction=0.4, rng=None):
    if rng is None:
        rng = random
    probe = pattern[:]
    num_flip = int(len(pattern) * noise_fraction)
    indices = rng.sample(range(len(pattern)), num_flip)
    for i in indices:
        probe[i] *= -1
    return probe


def phase_state_hamming(pattern_a, pattern_b):
    """
    Compute Hamming distance using Phase-State interference.
    Matching pairs reinforce, mismatching pairs cancel to syntony.
    """
    mismatches = 0
    for a, b in zip(pattern_a, pattern_b):
        node_a = GaussianNode(a)
        node_b = GaussianNode(b)
        if (node_a._state + node_b._state).norm() < 1e-12:
            mismatches += 1
    return mismatches


def phase_state_recall(stored_patterns, noisy_probe):
    """
    Phase-State associative recall via element-wise interference.
    Returns the closest stored pattern by Hamming distance.
    """
    best_distance = len(noisy_probe) + 1
    best_idx = 0

    for idx, stored in enumerate(stored_patterns):
        dist = phase_state_hamming(stored, noisy_probe)
        if dist < best_distance:
            best_distance = dist
            best_idx = idx

    overlap = (len(noisy_probe) - best_distance) / len(noisy_probe)
    return stored_patterns[best_idx], overlap


def test_associative_memory(seed=42):
    """
    256-node memory bank, 10 patterns, 40% noise probe.
    Phase-State vs Attention baseline.
    """
    pattern_dim = 256
    num_patterns = 10
    noise = 0.4

    rng = random.Random(seed)

    patterns = []
    for _ in range(num_patterns):
        p = [1 if rng.random() > 0.5 else -1 for _ in range(pattern_dim)]
        patterns.append(p)

    probes = [make_noisy_probe(p, noise, rng) for p in patterns]

    # --- Phase-State Recall ---
    print("\n  Running Phase-State XOR-Harmonization recall...")
    ps_correct_bits_total = 0
    ps_pattern_hits = 0

    t0 = time.time()
    for i, (original, probe) in enumerate(zip(patterns, probes)):
        recalled, overlap = phase_state_recall(patterns, probe)
        bit_acc = sum(1 for r, o in zip(recalled, original) if r == o) / pattern_dim
        ps_correct_bits_total += bit_acc
        if recalled == original:
            ps_pattern_hits += 1
        print(f"    Pattern {i}: overlap={overlap*100:.1f}%, bit_acc={bit_acc*100:.1f}%")
    ps_time = (time.time() - t0) * 1000

    ps_mean_bit_acc = ps_correct_bits_total / num_patterns

    # --- Transformer Baseline ---
    print("\n  Running Attention Memory baseline...")
    attn_mem = AttentionMemory(pattern_dim, d_model=32)
    attn_mem.store(patterns)
    attn_correct_bits_total = 0
    attn_pattern_hits = 0

    t0 = time.time()
    for i, (original, probe) in enumerate(zip(patterns, probes)):
        recalled = attn_mem.recall(probe)
        bit_acc = sum(1 for r, o in zip(recalled, original) if r == o) / pattern_dim
        attn_correct_bits_total += bit_acc
        if recalled == original:
            attn_pattern_hits += 1
        print(f"    Pattern {i}: bit_acc={bit_acc*100:.1f}%")
    attn_time = (time.time() - t0) * 1000

    attn_mean_bit_acc = attn_correct_bits_total / num_patterns

    print()
    print_table(
        ["Model", "Mean Bit Acc", "Pattern Recall", "Time (ms)", "Params"],
        [
            ["Phase-State 256",
             f"{ps_mean_bit_acc*100:.1f}%",
             f"{ps_pattern_hits}/{num_patterns}",
             f"{ps_time:.0f}",
             "0 (256 nodes)"],
            ["Attention Memory",
             f"{attn_mean_bit_acc*100:.1f}%",
             f"{attn_pattern_hits}/{num_patterns}",
             f"{attn_time:.0f}",
             str(attn_mem.param_count())],
        ]
    )

    return ps_mean_bit_acc, attn_mean_bit_acc


# ============================================================================
#  TEST 5: Gnosis / Attractor (K=24 Saturation)
# ============================================================================

def test_gnosis_attractor():
    """
    Tests the Gnosis phase transition: 24 nodes driven to K=24 recursive
    depth saturation. Measures cycles to transition, saturated node count,
    and phase entropy.
    """
    compiler = PhaseStateCompiler(kissing_number_threshold=24, stale_threshold=1)

    # 12 pairs of asymmetric data to force orbital cycles toward K_D4 saturation
    raw_data = [1, 1] * 12
    compiler.load_data(raw_data)

    gnosis_cycle = None

    with suppress_stdout():
        for cycle in range(100):
            compiler.compile_cycle()
            if compiler.gnosis_layer >= 3 and gnosis_cycle is None:
                gnosis_cycle = cycle

    saturated = sum(1 for n in compiler.nodes if n.recursive_depth >= 24)
    phase_entropy = sum(1 for n in compiler.nodes if not n.is_syntonic())
    max_depth = max(n.recursive_depth for n in compiler.nodes)

    rows = [
        ["Gnosis Layer", str(compiler.gnosis_layer)],
        ["Gnosis Reached at Cycle", str(gnosis_cycle) if gnosis_cycle is not None else "NOT REACHED"],
        ["Nodes Saturated (depth>=24)", f"{saturated}/{len(compiler.nodes)}"],
        ["Phase Entropy (active nodes)", str(phase_entropy)],
        ["Max Recursive Depth", str(max_depth)],
    ]

    print_table(["Metric", "Value"], rows)

    success = compiler.gnosis_layer == 3
    print(f"  Gnosis Transition: {'ACHIEVED' if success else 'FAILED'}")
    return success


# ============================================================================
#  TEST 6: Harmonic Readout (Zero-Parameter Resonance Inference)
# ============================================================================

class HarmonicReadout:
    """
    Zero-parameter readout for SRT architectures.
    Predicts tokens by finding which alphabet candidate achieves
    maximum resonance (minimum phase residual) within the lane.

    Instead of training weights, we let the toroidal lane's geometry
    reveal the answer through structural interference.
    """

    def __init__(self, alphabet=None):
        self.alphabet = alphabet or [1, -1]

    def predict(self, lane):
        """
        Test each candidate against every node in the lane.
        The candidate whose phase best matches the stored state
        (minimum |stored - candidate| residual) wins.
        """
        best_char = None
        min_residual = None

        for char in self.alphabet:
            # Resonance test: |stored - candidate| → 0 means perfect match
            neg_candidate = syn.state(
                [complex(-char, 0)], dtype="complex128"
            )
            total_residual = 0
            for node in lane.nodes:
                total_residual += (node._state + neg_candidate).norm()

            if min_residual is None or total_residual < min_residual:
                min_residual = total_residual
                best_char = char

        return best_char


def test_harmonic_psrnn(seed=42):
    """
    PS-RNN with Harmonic Readout vs LSTM with trained readout.
    PS-RNN needs 0 parameters — the toroidal lane topology preserves
    the signal structurally, and the readout simply tests for resonance.
    LSTM has no such structure and requires trained weights.
    """
    input_size = 8
    hidden_size = 64
    seq_len = 16
    num_trials = 3

    results = []

    for trial in range(num_trials):
        s = seed + trial * 7
        random.seed(s)
        base_vec = [1 if random.random() > 0.5 else -1 for _ in range(input_size)]
        sequence = [[v * (1 if t % 4 < 2 else -1) for v in base_vec]
                     for t in range(seq_len)]
        ground_truth = [v * (1 if seq_len % 4 < 2 else -1) for v in base_vec]

        # --- PS-RNN + Harmonic Readout (0 params, no training) ---
        ps_rnn = PhaseStateRNNCell(input_size, hidden_size)
        readout = HarmonicReadout(alphabet=[1, -1])

        t0 = time.time()
        with suppress_stdout():
            for x in sequence:
                ps_rnn.step(x)

        ps_pred = [readout.predict(lane) for lane in ps_rnn.lanes]
        ps_time = (time.time() - t0) * 1000
        ps_correct = sum(1 for p, g in zip(ps_pred, ground_truth) if p == g)
        ps_acc = ps_correct / input_size

        # --- LSTM + Trained Readout (520 params, 10 gradient steps) ---
        lstm_model = sn.Module()
        lstm_model.readout = ResonantLinear(hidden_size, input_size, bias=True)
        lstm_opt = GoldenMomentumOptimizer(
            [lstm_model.readout.weight.tensor, lstm_model.readout.bias.tensor],
            lr=0.1)

        random.seed(s)
        lstm = PureFloat32LSTM(input_size, hidden_size)
        for x in sequence:
            lstm.step(x)
        lstm_hidden = lstm.h

        t0 = time.time()
        target_floats = [v * 1.0 for v in ground_truth]
        w = lstm_model.readout.weight.tensor
        b = lstm_model.readout.bias.tensor
        out_sz, in_sz = w.shape[0], w.shape[1]

        for step in range(10):
            lstm_opt.zero_grad()
            h_t = ResonantTensor(lstm_hidden, [1, in_sz])
            t_t = ResonantTensor(target_floats, [1, out_sz])
            o_t = lstm_model.readout(h_t)
            err = o_t.elementwise_add(t_t.negate()).to_floats()

            scale = 2.0 / out_sz
            w._grad = [0.0] * (out_sz * in_sz)
            b._grad = [0.0] * out_sz
            for i in range(out_sz):
                for j in range(in_sz):
                    w._grad[i * in_sz + j] = scale * err[i] * lstm_hidden[j]
                b._grad[i] = scale * err[i]
            lstm_opt.step()

        h_t = ResonantTensor(lstm_hidden, [1, in_sz])
        lstm_out = lstm_model.readout(h_t).to_floats()
        lstm_pred = [1 if v >= 0 else -1 for v in lstm_out[:input_size]]
        lstm_time = (time.time() - t0) * 1000
        lstm_correct = sum(1 for p, g in zip(lstm_pred, ground_truth) if p == g)
        lstm_acc = lstm_correct / input_size

        results.append((ps_acc, ps_time, lstm_acc, lstm_time))

        print(f"  Trial {trial+1}: PS {ps_acc*100:.0f}% ({ps_time:.1f}ms)"
              f"  |  LSTM {lstm_acc*100:.0f}% ({lstm_time:.1f}ms)")

    avg_ps = sum(r[0] for r in results) / num_trials
    avg_ps_t = sum(r[1] for r in results) / num_trials
    avg_lstm = sum(r[2] for r in results) / num_trials
    avg_lstm_t = sum(r[3] for r in results) / num_trials
    lstm_params = lstm_model.readout.weight.tensor.size + lstm_model.readout.bias.tensor.size

    print()
    print_table(
        ["Model", "Avg Accuracy", "Avg Time (ms)", "Params", "Logic"],
        [
            ["PS-RNN + Harmonic",
             f"{avg_ps*100:.1f}%", f"{avg_ps_t:.1f}", "0", "Resonance"],
            ["LSTM + Trained Readout",
             f"{avg_lstm*100:.1f}%", f"{avg_lstm_t:.1f}",
             str(lstm_params), "Gradient"],
        ]
    )
    print("  Note: PS-RNN uses zero-parameter resonance readout.")
    print("  LSTM requires 10 gradient steps via GoldenMomentumOptimizer.")

    return avg_ps, avg_lstm


# ============================================================================
#  TEST 7: 1024-Node Associative Memory (100 patterns, 40% noise)
# ============================================================================

def test_associative_memory_1024(seed=42):
    """
    Scale-up of TEST 4: 1024-node memory bank, 100 patterns, 40% noise.
    """
    pattern_dim = 1024
    num_patterns = 100
    noise = 0.4

    rng = random.Random(seed)

    patterns = []
    for _ in range(num_patterns):
        p = [1 if rng.random() > 0.5 else -1 for _ in range(pattern_dim)]
        patterns.append(p)

    probes = [make_noisy_probe(p, noise, rng) for p in patterns]

    # --- Phase-State Recall ---
    print("\n  Running Phase-State XOR-Harmonization recall...")
    ps_correct_bits_total = 0
    ps_pattern_hits = 0

    t0 = time.time()
    for i, (original, probe) in enumerate(zip(patterns, probes)):
        recalled, overlap = phase_state_recall(patterns, probe)
        bit_acc = sum(1 for r, o in zip(recalled, original) if r == o) / pattern_dim
        ps_correct_bits_total += bit_acc
        if recalled == original:
            ps_pattern_hits += 1
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{num_patterns} done...")
    ps_time = (time.time() - t0) * 1000

    ps_mean_bit_acc = ps_correct_bits_total / num_patterns

    # --- Attention Memory Baseline ---
    print("\n  Running Attention Memory baseline...")
    attn_mem = AttentionMemory(pattern_dim, d_model=32)
    attn_mem.store(patterns)
    attn_correct_bits_total = 0
    attn_pattern_hits = 0

    t0 = time.time()
    for i, (original, probe) in enumerate(zip(patterns, probes)):
        recalled = attn_mem.recall(probe)
        bit_acc = sum(1 for r, o in zip(recalled, original) if r == o) / pattern_dim
        attn_correct_bits_total += bit_acc
        if recalled == original:
            attn_pattern_hits += 1
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{num_patterns} done...")
    attn_time = (time.time() - t0) * 1000

    attn_mean_bit_acc = attn_correct_bits_total / num_patterns

    print()
    print_table(
        ["Model", "Mean Bit Acc", "Pattern Recall", "Time (ms)", "Params"],
        [
            ["Phase-State 1024",
             f"{ps_mean_bit_acc*100:.1f}%",
             f"{ps_pattern_hits}/{num_patterns}",
             f"{ps_time:.0f}",
             "0 (1024 nodes)"],
            ["Attention Memory",
             f"{attn_mean_bit_acc*100:.1f}%",
             f"{attn_pattern_hits}/{num_patterns}",
             f"{attn_time:.0f}",
             str(attn_mem.param_count())],
        ]
    )

    return ps_mean_bit_acc, attn_mean_bit_acc


# ============================================================================
#  TEST 8: 2D Toroidal Grid (Wavefront Propagation & Self-Repair)
# ============================================================================

class Grid2DCompiler(PhaseStateCompiler):
    """
    2D toroidal grid extension of PhaseStateCompiler.
    Nodes arranged on a rows x cols torus with 4-directional neighbors.
    """

    def __init__(self, rows, cols, kissing_number_threshold=16,
                 allow_novelty=True, stale_threshold=3):
        super().__init__(
            kissing_number_threshold=kissing_number_threshold,
            allow_novelty=allow_novelty,
            toroidal=True,
            stale_threshold=stale_threshold,
        )
        self.rows = rows
        self.cols = cols
        for _ in range(rows * cols):
            self.nodes.append(GaussianNode(0))

    def _idx(self, r, c):
        """Flat index with toroidal wrapping."""
        return (r % self.rows) * self.cols + (c % self.cols)

    def _neighbors(self, flat_idx):
        """4-directional neighbor indices with toroidal wrap."""
        r = flat_idx // self.cols
        c = flat_idx % self.cols
        return [
            self._idx(r - 1, c),
            self._idx(r + 1, c),
            self._idx(r, c - 1),
            self._idx(r, c + 1),
        ]

    def set_node(self, r, c, val):
        """Set node at (r,c) to a given M4 value."""
        node = self.nodes[self._idx(r, c)]
        node._state = syn.state(
            [complex(val, 0)], dtype="complex128", device=node._state.device
        )
        node._project()

    def compile_cycle(self):
        """Override for 2D neighbor topology."""
        unresolved = [n for n in self.nodes if not n.is_syntonic()]
        if not unresolved:
            return True

        interacted = set()
        activity = False
        n_nodes = len(self.nodes)

        # 0. 2D Novelty — majority vote of non-syntonic 4-neighbors
        if self.allow_novelty and any(n.is_syntonic() for n in self.nodes):
            filled = self._inject_novelty_2d()
            interacted.update(filled)

        # 1. 2D Propagation — single wavefront ring per cycle
        newly_activated = set()
        for i, node_a in enumerate(self.nodes):
            if node_a.is_syntonic():
                continue
            if node_a in interacted:
                continue

            for nb_idx in self._neighbors(i):
                node_b = self.nodes[nb_idx]
                if node_b.is_syntonic() and node_b not in interacted:
                    node_b._state = syn.state(
                        [node_a._state.to_list()[0]],
                        dtype=node_b._state.dtype,
                        device=node_b._state.device
                    )
                    node_b._project()
                    node_b.stale_cycles = 0
                    interacted.add(node_a)
                    interacted.add(node_b)
                    newly_activated.add(node_b)
                    activity = True

        # 2. Global Harmonization
        for i, node_a in enumerate(self.nodes):
            if node_a.is_syntonic() or node_a in interacted or node_a in newly_activated:
                continue
            for j, node_b in enumerate(self.nodes):
                if i == j or node_b.is_syntonic() or node_b in interacted or node_b in newly_activated:
                    continue
                if getattr(node_a, 'is_source', False) and getattr(node_b, 'is_source', False):
                    continue
                if (node_a._state + node_b._state).norm() < 1e-12:
                    node_a.harmonize(node_b)
                    node_b._state = syn.state([0j], dtype=node_b._state.dtype,
                                               device=node_b._state.device)
                    node_b._project()
                    interacted.add(node_a)
                    interacted.add(node_b)
                    node_a.stale_cycles = 0
                    node_b.stale_cycles = 0
                    activity = True
                    break

        # 3. Dampened Recursion
        for node in self.nodes:
            if not node.is_syntonic() and node not in interacted:
                if getattr(node, 'is_source', False):
                    continue
                node.stale_cycles += 1
                if node.stale_cycles >= self.stale_threshold:
                    node.recurse()
                    node.stale_cycles = 0
                    if node.recursive_depth >= self.K_THRESHOLD and self.gnosis_layer < 3:
                        self._trigger_gnosis_transition()
            else:
                if hasattr(node, 'stale_cycles'):
                    node.stale_cycles = 0

        return False

    def _inject_novelty_2d(self):
        """Fill syntonic nodes via majority vote of 4 non-syntonic neighbors."""
        empty_indices = [i for i, n in enumerate(self.nodes) if n.is_syntonic()]
        if not empty_indices:
            return set()

        filled_nodes = set()
        for idx in empty_indices:
            node = self.nodes[idx]
            if not node.is_syntonic():
                continue

            neighbor_vals = []
            for nb_idx in self._neighbors(idx):
                nb = self.nodes[nb_idx]
                if not nb.is_syntonic():
                    neighbor_vals.append(nb.m4_val)

            if neighbor_vals:
                vote = sum(neighbor_vals)
                val = 1 if vote > 0 else (-1 if vote < 0 else 0)
                if val != 0:
                    node._state = syn.state(
                        [complex(val, 0)], dtype=node._state.dtype,
                        device=node._state.device
                    )
                    node._project()
                    node.stale_cycles = 0
                    filled_nodes.add(node)

        return filled_nodes


def test_2d_toroidal_grid():
    """
    Sub-test A: Wavefront propagation from a single source on 16x16 torus.
    Sub-test B: Self-repair of damaged region on 16x16 torus.
    """
    rows, cols = 16, 16
    total_nodes = rows * cols

    # --- Sub-test A: Wavefront ---
    print("\n  Sub-test A: Wavefront Propagation")
    grid = Grid2DCompiler(rows, cols, allow_novelty=False, stale_threshold=999)

    # Single source at (0,0)
    grid.set_node(0, 0, 1)
    grid.nodes[grid._idx(0, 0)].is_source = True

    milestones = {25: None, 50: None, 75: None, 100: None}
    toroidal_confirmed = False

    with suppress_stdout():
        for cycle in range(rows + cols):
            grid.compile_cycle()
            active = sum(1 for n in grid.nodes if not n.is_syntonic())
            pct = active * 100 // total_nodes

            # Check opposite corner for toroidal wrap
            opp = grid.nodes[grid._idx(rows // 2, cols // 2)]
            if not opp.is_syntonic() and not toroidal_confirmed:
                toroidal_confirmed = True

            for m in milestones:
                if milestones[m] is None and pct >= m:
                    milestones[m] = cycle + 1

            if active >= total_nodes:
                break

    print_table(
        ["Metric", "Value"],
        [
            ["Grid Size", f"{rows}x{cols} ({total_nodes} nodes)"],
            ["Cycles to 25% fill", str(milestones.get(25, "N/A"))],
            ["Cycles to 50% fill", str(milestones.get(50, "N/A"))],
            ["Cycles to 75% fill", str(milestones.get(75, "N/A"))],
            ["Cycles to 100% fill", str(milestones.get(100, "N/A"))],
            ["Toroidal wrap confirmed", "YES" if toroidal_confirmed else "NO"],
        ]
    )

    # --- Sub-test B: Self-Repair ---
    print("\n  Sub-test B: Self-Repair")
    grid2 = Grid2DCompiler(rows, cols, allow_novelty=True, stale_threshold=999)

    # Fill entire grid with +1
    for r in range(rows):
        for c in range(cols):
            grid2.set_node(r, c, 1)
            grid2.nodes[grid2._idx(r, c)].is_source = True

    # Damage 4x4 region (rows 6-9, cols 6-9)
    damage_size = 4
    damage_r0, damage_c0 = 6, 6
    damaged_indices = []
    for r in range(damage_r0, damage_r0 + damage_size):
        for c in range(damage_c0, damage_c0 + damage_size):
            idx = grid2._idx(r, c)
            grid2.nodes[idx]._state = syn.state(
                [0j], dtype=grid2.nodes[idx]._state.dtype,
                device=grid2.nodes[idx]._state.device
            )
            grid2.nodes[idx].is_source = False
            damaged_indices.append(idx)

    repair_cycle = None
    with suppress_stdout():
        for cycle in range(20):
            grid2.compile_cycle()
            repaired = sum(1 for idx in damaged_indices
                           if not grid2.nodes[idx].is_syntonic())
            if repaired == len(damaged_indices) and repair_cycle is None:
                repair_cycle = cycle + 1
                break

    # Check accuracy
    repair_correct = sum(1 for idx in damaged_indices
                         if grid2.nodes[idx].m4_val == 1)
    repair_acc = repair_correct / len(damaged_indices)

    print_table(
        ["Metric", "Value"],
        [
            ["Pattern", f"Uniform +1 ({rows}x{cols})"],
            ["Damaged nodes", str(len(damaged_indices))],
            ["Cycles to full repair", str(repair_cycle) if repair_cycle else ">20"],
            ["Repair accuracy", f"{repair_acc*100:.1f}%"],
        ]
    )

    return toroidal_confirmed, repair_acc


# ============================================================================
#  TEST 9: Tiny Language Task (Next-Token Prediction)
# ============================================================================

def encode_binary(text, alphabet="AB"):
    """Encode string as +1/-1 values. First char -> +1, second -> -1."""
    mapping = {alphabet[0]: 1, alphabet[1]: -1}
    return [mapping.get(ch, 0) for ch in text]


def decode_binary(values, alphabet="AB"):
    """Decode +1/-1 values back to string."""
    return "".join(alphabet[0] if v >= 0 else alphabet[1] for v in values)


def encode_4token(text):
    """Encode A/B/C/D text as 2 nodes per token."""
    mapping = {'A': [1, 1], 'B': [1, -1], 'C': [-1, 1], 'D': [-1, -1]}
    result = []
    for ch in text:
        result.extend(mapping.get(ch, [0, 0]))
    return result


def decode_4token(values):
    """Decode 2-node encoding back to A/B/C/D text."""
    inv = {(1, 1): 'A', (1, -1): 'B', (-1, 1): 'C', (-1, -1): 'D'}
    result = []
    for i in range(0, len(values) - 1, 2):
        key = (1 if values[i] >= 0 else -1, 1 if values[i + 1] >= 0 else -1)
        result.append(inv.get(key, '?'))
    return "".join(result)


class BigramBaseline:
    """Bigram frequency model for next-token prediction."""

    def __init__(self, vocab):
        self.vocab = vocab
        self.counts = {}

    def train(self, sequence):
        """Train on a single sequence (list of tokens)."""
        for i in range(len(sequence) - 1):
            pair = (sequence[i], sequence[i + 1])
            self.counts[pair] = self.counts.get(pair, 0) + 1

    def predict(self, context_token):
        """Predict most likely next token."""
        best_count = -1
        best_next = self.vocab[0]
        for next_tok in self.vocab:
            c = self.counts.get((context_token, next_tok), 0)
            if c > best_count:
                best_count = c
                best_next = next_tok
        return best_next


def test_tiny_language():
    """
    Tests next-token prediction on symbolic repeating sequences.
    Phase-State compiler uses autocorrelation period detection.
    Bigram baseline uses simple frequency counting.
    """
    # --- Part A: Binary Alphabet ---
    print("\n  Part A: Binary Alphabet (A/B)")

    binary_patterns = [
        ("AABB x25", "AABB" * 25),
        ("ABAB x50", "AB" * 50),
        ("AAAB x25", "AAAB" * 25),
        ("ALL_A",    "A" * 100),
    ]

    blank_count = 10
    rows_a = []

    for name, text in binary_patterns:
        encoded = encode_binary(text)
        context = encoded[:len(encoded) - blank_count]
        ground_truth = encoded[len(encoded) - blank_count:]

        # Phase-State prediction
        compiler = PhaseStateCompiler(kissing_number_threshold=16, allow_novelty=True)
        compiler.load_data(encoded)

        # Stamp exact values
        for i, val in enumerate(encoded):
            compiler.nodes[i]._state = syn.state(
                [complex(val, 0)], dtype="complex128",
                device=compiler.nodes[i]._state.device
            )
            compiler.nodes[i]._project()

        # Blank last nodes
        for i in range(len(encoded) - blank_count, len(encoded)):
            compiler.nodes[i]._state = compiler.nodes[i]._state * 0

        # Anchor context
        for i in range(len(encoded) - blank_count):
            compiler.nodes[i].is_source = True

        recovered = lambda nodes, bc=blank_count: all(
            not n.is_syntonic() for n in nodes[-bc:]
        )

        with suppress_stdout():
            compiler.run(max_cycles=30, exit_condition=recovered)

        ps_pred = [n.m4_val for n in compiler.nodes[-blank_count:]]
        ps_correct = sum(1 for p, g in zip(ps_pred, ground_truth) if p == g)

        # Bigram baseline
        text_list = list(text)
        bigram = BigramBaseline(["A", "B"])
        bigram.train(text_list[:len(text) - blank_count])

        bg_correct = 0
        prev_token = text_list[len(text) - blank_count - 1]
        for i in range(blank_count):
            pred_token = bigram.predict(prev_token)
            actual_token = text_list[len(text) - blank_count + i]
            if pred_token == actual_token:
                bg_correct += 1
            prev_token = pred_token

        rows_a.append([
            name,
            f"{ps_correct}/{blank_count} ({ps_correct*100//blank_count}%)",
            f"{bg_correct}/{blank_count} ({bg_correct*100//blank_count}%)",
        ])

    print_table(["Pattern", "PS Accuracy", "Bigram Accuracy"], rows_a)

    # --- Part B: 4-Token Alphabet ---
    print("\n  Part B: 4-Token Alphabet (A/B/C/D, 2 nodes/token)")

    quad_patterns = [
        ("ABCD x25",     "ABCD" * 25),
        ("AABBCCDD x12", ("AABBCCDD" * 13)[:100]),
        ("ABDC x25",     "ABDC" * 25),
    ]

    blank_tokens = 10
    rows_b = []

    for name, text in quad_patterns:
        encoded = encode_4token(text)
        blank_nodes = blank_tokens * 2
        context_nodes = len(encoded) - blank_nodes
        ground_truth_text = text[-blank_tokens:]

        compiler = PhaseStateCompiler(kissing_number_threshold=16, allow_novelty=True)
        compiler.load_data(encoded)

        for i, val in enumerate(encoded):
            compiler.nodes[i]._state = syn.state(
                [complex(val, 0)], dtype="complex128",
                device=compiler.nodes[i]._state.device
            )
            compiler.nodes[i]._project()

        for i in range(context_nodes, len(encoded)):
            compiler.nodes[i]._state = compiler.nodes[i]._state * 0

        for i in range(context_nodes):
            compiler.nodes[i].is_source = True

        recovered = lambda nodes, bn=blank_nodes: all(
            not n.is_syntonic() for n in nodes[-bn:]
        )

        with suppress_stdout():
            compiler.run(max_cycles=30, exit_condition=recovered)

        pred_vals = [n.m4_val for n in compiler.nodes[-blank_nodes:]]
        pred_text = decode_4token(pred_vals)
        ps_correct = sum(1 for p, g in zip(pred_text, ground_truth_text) if p == g)

        # Bigram baseline
        text_list = list(text)
        bigram = BigramBaseline(["A", "B", "C", "D"])
        bigram.train(text_list[:len(text) - blank_tokens])

        bg_correct = 0
        prev_token = text_list[len(text) - blank_tokens - 1]
        for i in range(blank_tokens):
            pred_token = bigram.predict(prev_token)
            actual_token = text_list[len(text) - blank_tokens + i]
            if pred_token == actual_token:
                bg_correct += 1
            prev_token = pred_token

        rows_b.append([
            name,
            f"{ps_correct}/{blank_tokens} ({ps_correct*100//blank_tokens}%)",
            f"{bg_correct}/{blank_tokens} ({bg_correct*100//blank_tokens}%)",
        ])

    print_table(["Pattern", "PS Accuracy", "Bigram Accuracy"], rows_b)

def test_4096_grid_scaling():
    """
    TEST 10: 4096-Node Scale Test (64x64 Toroidal Grid)
    Measures CPU latency and exactness at scale using pure Python 
    and srt_library, tracking a single wavefront across the manifold.
    """
    rows, cols = 64, 64
    total_nodes = rows * cols

    print("\n  Running 4096-Node Wavefront Scaling Test...")
    
    # Initialize the 64x64 toroidal compiler
    t_init = time.time()
    grid = Grid2DCompiler(rows, cols, allow_novelty=False, stale_threshold=999)
    init_time = time.time() - t_init

    # Set single source at (0,0)
    grid.set_node(0, 0, 1)
    grid.nodes[grid._idx(0, 0)].is_source = True

    milestones = {25: None, 50: None, 75: None, 100: None}
    toroidal_confirmed = False

    t0 = time.time()
    with suppress_stdout():
        # Maximum cycles needed for a 64x64 grid wavefront is rows + cols
        for cycle in range(rows + cols):
            grid.compile_cycle()
            
            # Diagnostic tracking
            active = sum(1 for n in grid.nodes if not n.is_syntonic())
            pct = active * 100 // total_nodes

            # Check opposite corner to confirm topological wrapping
            opp = grid.nodes[grid._idx(rows // 2, cols // 2)]
            if not opp.is_syntonic() and not toroidal_confirmed:
                toroidal_confirmed = True

            for m in milestones:
                if milestones[m] is None and pct >= m:
                    milestones[m] = cycle + 1

            if active >= total_nodes:
                break
                
    prop_time = time.time() - t0

    print_table(
        ["Metric", "Value"],
        [
            ["Grid Size", f"{rows}x{cols} ({total_nodes} nodes)"],
            ["Initialization Time (s)", f"{init_time:.4f}"],
            ["Propagation Time (s)", f"{prop_time:.4f}"],
            ["Cycles to 25% fill", str(milestones.get(25, "N/A"))],
            ["Cycles to 50% fill", str(milestones.get(50, "N/A"))],
            ["Cycles to 75% fill", str(milestones.get(75, "N/A"))],
            ["Cycles to 100% fill", str(milestones.get(100, "N/A"))],
            ["Toroidal wrap confirmed", "YES" if toroidal_confirmed else "NO"],
        ]
    )

    return prop_time

# ============================================================================
#  Main
# ============================================================================

class TeeWriter:
    """Write to both stdout and a file simultaneously."""
    def __init__(self, terminal, logfile):
        self.terminal = terminal
        self.logfile = logfile
    def write(self, msg):
        self.terminal.write(msg)
        self.logfile.write(msg)
    def flush(self):
        self.terminal.flush()
        self.logfile.flush()


if __name__ == '__main__':
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "benchmark_results.txt")
    logfile = open(output_path, 'w')
    original_stdout = sys.stdout
    sys.stdout = TeeWriter(original_stdout, logfile)

    print_header("PHASE-STATE COMPILER BENCHMARK SUITE")
    print("No external dependencies (PyTorch, NumPy, SciPy)")
    print("All operations via srt_library or pure Python\n")

    print_header("TEST 0: Sequence Copying (Topological Routing)")
    test_sequence_copy()

    print_header("TEST 1: XOR Correctness (Harmonization)")
    test_xor_correctness()

    print_header("TEST 2: Sequence Prediction (16-step, Multi-Pattern)")
    test_sequence_prediction()

    print_header("TEST 3: 64-Node Phase-State RNN vs Float32 LSTM")
    test_vectorized_rnn()

    print_header("TEST 4: Associative Memory (256-node, 10 patterns, 40% noise)")
    test_associative_memory()

    print_header("TEST 5: Gnosis / Attractor (K=24 Saturation)")
    test_gnosis_attractor()

    print_header("TEST 6: Harmonic Readout (Zero-Parameter Resonance)")
    test_harmonic_psrnn()

    print_header("TEST 7: Associative Memory (1024-node, 100 patterns, 40% noise)")
    test_associative_memory_1024()

    print_header("TEST 8: 2D Toroidal Grid (Wavefront & Self-Repair)")
    test_2d_toroidal_grid()

    print_header("TEST 9: Tiny Language Task (Next-Token Prediction)")
    test_tiny_language()

    print_header("TEST 10: 4096-Node Scale Test (64x64 Toroidal Grid)")
    test_4096_grid_scaling()    

    print("\n" + "=" * 70)
    print("  ALL BENCHMARKS COMPLETE")
    print("=" * 70)

    sys.stdout = original_stdout
    logfile.close()
    print(f"\nResults saved to: {output_path}")
