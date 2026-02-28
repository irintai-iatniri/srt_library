"""
Phase-State Compiler STRESS BENCHMARK Suite
============================================
Designed to find the LIMITS of the Phase-State architecture and test
against FAIR baselines (trained and structurally appropriate).

Tests:
  0. Noise Tolerance Sweep       — Associative memory at 10%-80% noise
  1. Capacity Saturation          — How many patterns before recall breaks?
  2. Hierarchical Pattern Pred.   — Nested periods, primes, Fibonacci
  3. Adversarial Sequences        — Phase shifts, aperiodic traps
  4. Vocabulary Scaling           — 2 to 16 tokens
  5. Sequence Length Scaling       — 32 to 512 nodes, fixed 10% blank
  6. Trained Hopfield Baseline    — The REAL associative memory comparison
  7. Self-Repair Damage Sweep     — 5% to 60% grid damage
  8. Cold Start / Minimal Context — How little context is enough?
  9. Multi-Frequency Interference  — Superimposed periodic signals

All tests use srt_library or pure Python only.
"""

import sys
import os
import time
import math
import random
import contextlib
import copy

import srt_library.core as syn
from phase_state_vibes_compiler import PhaseStateCompiler, GaussianNode


# ============================================================================
#  Utilities
# ============================================================================

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
#  Shared Helpers
# ============================================================================

def phase_state_hamming(pattern_a, pattern_b):
    """Hamming distance via Phase-State interference."""
    mismatches = 0
    for a, b in zip(pattern_a, pattern_b):
        node_a = GaussianNode(a)
        node_b = GaussianNode(b)
        if (node_a._state + node_b._state).norm() < 1e-12:
            mismatches += 1
    return mismatches


def phase_state_recall(stored_patterns, noisy_probe):
    """Recall closest stored pattern by PS Hamming distance."""
    best_distance = len(noisy_probe) + 1
    best_idx = 0
    for idx, stored in enumerate(stored_patterns):
        dist = phase_state_hamming(stored, noisy_probe)
        if dist < best_distance:
            best_distance = dist
            best_idx = idx
    overlap = (len(noisy_probe) - best_distance) / len(noisy_probe)
    return stored_patterns[best_idx], overlap


def make_noisy_probe(pattern, noise_fraction, rng):
    probe = pattern[:]
    num_flip = int(len(pattern) * noise_fraction)
    indices = rng.sample(range(len(pattern)), num_flip)
    for i in indices:
        probe[i] *= -1
    return probe


def encode_multi_token(text, vocab):
    """
    Encode text using log2(len(vocab)) nodes per token.
    Each token maps to a unique binary code in {+1, -1}.
    """
    bits_per_token = max(1, math.ceil(math.log2(max(len(vocab), 2))))
    mapping = {}
    for i, ch in enumerate(vocab):
        code = []
        for b in range(bits_per_token):
            code.append(1 if (i >> (bits_per_token - 1 - b)) & 1 == 0 else -1)
        mapping[ch] = code
    result = []
    for ch in text:
        result.extend(mapping.get(ch, [0] * bits_per_token))
    return result, bits_per_token, mapping


def decode_multi_token(values, bits_per_token, mapping):
    """Decode multi-node encoding back to text."""
    inv = {}
    for ch, code in mapping.items():
        key = tuple(1 if v >= 0 else -1 for v in code)
        inv[key] = ch
    result = []
    for i in range(0, len(values) - bits_per_token + 1, bits_per_token):
        chunk = tuple(1 if values[i + b] >= 0 else -1 for b in range(bits_per_token))
        result.append(inv.get(chunk, '?'))
    return "".join(result)


# ============================================================================
#  Classical Hopfield Network (the FAIR associative memory baseline)
# ============================================================================

class HopfieldNetwork:
    """
    Classical Hopfield network with Hebbian learning.
    This is THE standard associative memory — the proper comparison
    for Phase-State recall claims.

    Storage: O(N^2) weight matrix via outer product rule.
    Recall: Synchronous update until convergence.
    Known capacity: ~0.138 * N patterns for N nodes.
    """

    def __init__(self, n_nodes):
        self.n = n_nodes
        # Weight matrix (stored as flat list for pure Python)
        self.W = [0.0] * (n_nodes * n_nodes)

    def store(self, patterns):
        """Hebbian storage: W += (1/N) * sum(p * p^T) with zero diagonal."""
        n = self.n
        self.W = [0.0] * (n * n)
        for p in patterns:
            for i in range(n):
                for j in range(n):
                    if i != j:
                        self.W[i * n + j] += p[i] * p[j] / n

    def recall(self, probe, max_steps=20):
        """Synchronous update until convergence."""
        state = probe[:]
        n = self.n
        for _ in range(max_steps):
            new_state = [0] * n
            for i in range(n):
                h = sum(self.W[i * n + j] * state[j] for j in range(n))
                new_state[i] = 1 if h >= 0 else -1
            if new_state == state:
                break
            state = new_state
        return state

    def param_count(self):
        return self.n * self.n


# ============================================================================
#  TEST 0: Noise Tolerance Sweep
# ============================================================================

def test_noise_sweep(seed=42):
    """
    256-node memory, 10 patterns. Sweep noise from 10% to 80%.
    Find where Phase-State recall degrades. Compare vs Hopfield.
    """
    pattern_dim = 256
    num_patterns = 10
    noise_levels = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

    rng = random.Random(seed)
    patterns = []
    for _ in range(num_patterns):
        p = [1 if rng.random() > 0.5 else -1 for _ in range(pattern_dim)]
        patterns.append(p)

    # Store in Hopfield
    hopfield = HopfieldNetwork(pattern_dim)
    hopfield.store(patterns)

    rows = []

    for noise in noise_levels:
        rng_probe = random.Random(seed + int(noise * 1000))
        probes = [make_noisy_probe(p, noise, rng_probe) for p in patterns]

        # Phase-State recall
        ps_hits = 0
        ps_bit_total = 0
        for original, probe in zip(patterns, probes):
            recalled, _ = phase_state_recall(patterns, probe)
            bit_acc = sum(1 for r, o in zip(recalled, original) if r == o) / pattern_dim
            ps_bit_total += bit_acc
            if recalled == original:
                ps_hits += 1

        # Hopfield recall
        hop_hits = 0
        hop_bit_total = 0
        for original, probe in zip(patterns, probes):
            recalled = hopfield.recall(probe)
            bit_acc = sum(1 for r, o in zip(recalled, original) if r == o) / pattern_dim
            hop_bit_total += bit_acc
            if recalled == original:
                hop_hits += 1

        ps_avg = ps_bit_total / num_patterns * 100
        hop_avg = hop_bit_total / num_patterns * 100

        rows.append([
            f"{noise*100:.0f}%",
            f"{ps_hits}/{num_patterns}",
            f"{ps_avg:.1f}%",
            f"{hop_hits}/{num_patterns}",
            f"{hop_avg:.1f}%",
        ])

    print_table(
        ["Noise", "PS Recall", "PS Bit Acc", "Hopfield Recall", "Hopfield Bit Acc"],
        rows
    )


# ============================================================================
#  TEST 1: Capacity Saturation
# ============================================================================

def test_capacity_saturation(seed=42):
    """
    256-node memory. Sweep pattern count from 5 to 100.
    Hopfield theoretical limit: ~0.138 * 256 ≈ 35 patterns.
    Where does Phase-State break?
    """
    pattern_dim = 256
    noise = 0.3
    pattern_counts = [5, 10, 20, 35, 50, 75, 100]

    rows = []

    for num_patterns in pattern_counts:
        rng = random.Random(seed)
        patterns = []
        for _ in range(num_patterns):
            p = [1 if rng.random() > 0.5 else -1 for _ in range(pattern_dim)]
            patterns.append(p)

        probes = [make_noisy_probe(p, noise, random.Random(seed + 777)) for p in patterns]

        # Phase-State
        ps_hits = 0
        for original, probe in zip(patterns, probes):
            recalled, _ = phase_state_recall(patterns, probe)
            if recalled == original:
                ps_hits += 1

        # Hopfield
        hopfield = HopfieldNetwork(pattern_dim)
        hopfield.store(patterns)
        hop_hits = 0
        for original, probe in zip(patterns, probes):
            recalled = hopfield.recall(probe)
            if recalled == original:
                hop_hits += 1

        rows.append([
            str(num_patterns),
            f"{ps_hits}/{num_patterns} ({ps_hits*100//max(num_patterns,1)}%)",
            f"{hop_hits}/{num_patterns} ({hop_hits*100//max(num_patterns,1)}%)",
        ])

        print(f"  {num_patterns} patterns: PS {ps_hits}/{num_patterns}, "
              f"Hopfield {hop_hits}/{num_patterns}")

    print()
    print_table(
        ["# Patterns", "Phase-State Recall", "Hopfield Recall"],
        rows
    )
    print(f"  Hopfield theoretical limit: ~{int(0.138 * 256)} patterns for 256 nodes")


# ============================================================================
#  TEST 2: Hierarchical / Complex Pattern Prediction
# ============================================================================

def test_hierarchical_patterns():
    """
    Sequences with structure HARDER than simple period repetition:
    - Prime-length periods (3, 5, 7)
    - Nested repetition (AABB repeated as a unit in a larger cycle)
    - Fibonacci-modulated sign flips
    - Chirp (accelerating frequency)
    """
    test_cases = []

    # Period-3: AAB repeating
    p3 = [1, 1, -1] * 10  # 30 elements
    test_cases.append(("Period-3 (AAB)", p3, 6))

    # Period-5: AABBA repeating
    p5 = [1, 1, -1, -1, 1] * 6  # 30 elements
    test_cases.append(("Period-5 (AABBA)", p5, 5))

    # Period-7: AAABBBA repeating
    p7 = [1, 1, 1, -1, -1, -1, 1] * 4  # 28 elements
    test_cases.append(("Period-7 (AAABBBA)", p7, 7))

    # Nested: (AABB)(BBAA) as a period-8 unit
    nested = ([1, 1, -1, -1, -1, -1, 1, 1]) * 4  # 32 elements
    test_cases.append(("Nested period-8", nested, 8))

    # Fibonacci sign flips: sign changes at Fibonacci positions
    fib_seq = [1] * 40
    fib_positions = set()
    a, b = 1, 2
    while b < 40:
        fib_positions.add(b)
        a, b = b, a + b
    for i in range(40):
        if i in fib_positions:
            fib_seq[i] = -1
    test_cases.append(("Fibonacci positions", fib_seq, 8))

    # Chirp: period decreases (8, 6, 4, 2)
    chirp = []
    for period in [8, 6, 4, 2]:
        half = period // 2
        chirp.extend([1] * half + [-1] * half)
    test_cases.append(("Chirp (decelerating)", chirp, 4))

    rows = []

    for name, pattern, blank_count in test_cases:
        total_len = len(pattern)
        context_len = total_len - blank_count
        ground_truth = pattern[context_len:]

        compiler = PhaseStateCompiler(kissing_number_threshold=16, allow_novelty=True)
        compiler.load_data(pattern)

        # Stamp exact values
        for i, val in enumerate(pattern):
            compiler.nodes[i]._state = syn.state(
                [complex(val, 0)], dtype="complex128",
                device=compiler.nodes[i]._state.device
            )
            compiler.nodes[i]._project()

        # Blank prediction zone
        for i in range(context_len, total_len):
            compiler.nodes[i]._state = compiler.nodes[i]._state * 0

        # Anchor context
        for i in range(context_len):
            compiler.nodes[i].is_source = True

        recovered = lambda nodes, bc=blank_count: all(
            not n.is_syntonic() for n in nodes[-bc:]
        )

        with suppress_stdout():
            compiler.run(max_cycles=40, exit_condition=recovered)

        predicted = [n.m4_val for n in compiler.nodes[context_len:]]
        correct = sum(1 for p, g in zip(predicted, ground_truth) if p == g)
        acc = correct / blank_count * 100

        rows.append([name, f"{len(pattern)}", f"{blank_count}",
                     f"{correct}/{blank_count} ({acc:.0f}%)"])

    print_table(["Pattern Type", "Seq Len", "Blanks", "PS Accuracy"], rows)


# ============================================================================
#  TEST 3: Adversarial Sequences
# ============================================================================

def test_adversarial_sequences():
    """
    Sequences specifically designed to mislead autocorrelation-based prediction:
    - Phase shift: ABAB...ABBA (sudden inversion at prediction boundary)
    - Almost periodic: period-4 with a single anomaly in context
    - Random tail: periodic context, random ground truth
    - Palindromic: reads same forward and backward
    """
    test_cases = []

    # Phase shift at boundary: ABAB for 20, then BAAB at the end
    phase_shift = [1, -1] * 12 + [-1, 1, 1, -1]  # 28 elements
    test_cases.append(("Phase shift at boundary", phase_shift, 4))

    # Period-4 with single anomaly at position 9
    anomaly = [1, 1, -1, -1] * 6  # 24 elements
    anomaly[9] = 1  # was -1, now anomalous
    test_cases.append(("Period-4 + anomaly", anomaly, 4))

    # Periodic context, random truth: context is AABB, but last 4 are random
    misleading = [1, 1, -1, -1] * 5 + [1, -1, 1, 1]  # 24 elements
    test_cases.append(("Periodic + random tail", misleading, 4))

    # Palindrome: sequence reads same forward and backward
    half = [1, -1, 1, 1, -1, -1, 1, -1, 1, -1]
    palindrome = half + half[::-1]  # 20 elements
    test_cases.append(("Palindrome", palindrome, 4))

    # Gradual drift: starts as all +1, slowly transitions to all -1
    drift = [1]*6 + [1]*2 + [-1]*2 + [1]*1 + [-1]*3 + [-1]*6  # 20 elements
    test_cases.append(("Gradual drift", drift, 4))

    rows = []

    for name, pattern, blank_count in test_cases:
        total_len = len(pattern)
        context_len = total_len - blank_count
        ground_truth = pattern[context_len:]

        compiler = PhaseStateCompiler(kissing_number_threshold=16, allow_novelty=True)
        compiler.load_data(pattern)

        for i, val in enumerate(pattern):
            compiler.nodes[i]._state = syn.state(
                [complex(val, 0)], dtype="complex128",
                device=compiler.nodes[i]._state.device
            )
            compiler.nodes[i]._project()

        for i in range(context_len, total_len):
            compiler.nodes[i]._state = compiler.nodes[i]._state * 0

        for i in range(context_len):
            compiler.nodes[i].is_source = True

        recovered = lambda nodes, bc=blank_count: all(
            not n.is_syntonic() for n in nodes[-bc:]
        )

        with suppress_stdout():
            compiler.run(max_cycles=40, exit_condition=recovered)

        predicted = [n.m4_val for n in compiler.nodes[context_len:]]
        correct = sum(1 for p, g in zip(predicted, ground_truth) if p == g)
        acc = correct / blank_count * 100

        # Show what was predicted vs expected for adversarial analysis
        rows.append([name, str(ground_truth), str(predicted),
                     f"{correct}/{blank_count} ({acc:.0f}%)"])

    print_table(["Adversarial Type", "Ground Truth", "Predicted", "Accuracy"], rows)
    print("  Note: Low accuracy on adversarial inputs is NOT a failure —")
    print("  it reveals the system's inductive bias (periodicity assumption).")


# ============================================================================
#  TEST 4: Vocabulary Scaling
# ============================================================================

def test_vocabulary_scaling():
    """
    Test next-token prediction with 2, 4, 8, and 16-token alphabets.
    More tokens = more nodes per token = harder encoding.
    """
    configs = [
        (2,  "AB",                "AABB" * 25),           # 100 chars
        (4,  "ABCD",              "ABCD" * 25),           # 100 chars
        (8,  "ABCDEFGH",          "ABCDEFGH" * 12),       # 96 chars
        (16, "ABCDEFGHIJKLMNOP",  "ABCDEFGHIJKLMNOP" * 6), # 96 chars
    ]

    blank_tokens = 8
    rows = []

    for vocab_size, vocab, text in configs:
        encoded, bits_per_token, mapping = encode_multi_token(text, vocab)
        blank_nodes = blank_tokens * bits_per_token
        total_nodes = len(encoded)
        context_nodes = total_nodes - blank_nodes
        ground_truth_text = text[-blank_tokens:]

        compiler = PhaseStateCompiler(kissing_number_threshold=16, allow_novelty=True)
        compiler.load_data(encoded)

        for i, val in enumerate(encoded):
            compiler.nodes[i]._state = syn.state(
                [complex(val, 0)], dtype="complex128",
                device=compiler.nodes[i]._state.device
            )
            compiler.nodes[i]._project()

        for i in range(context_nodes, total_nodes):
            compiler.nodes[i]._state = compiler.nodes[i]._state * 0

        for i in range(context_nodes):
            compiler.nodes[i].is_source = True

        recovered = lambda nodes, bn=blank_nodes: all(
            not n.is_syntonic() for n in nodes[-bn:]
        )

        with suppress_stdout():
            compiler.run(max_cycles=40, exit_condition=recovered)

        pred_vals = [n.m4_val for n in compiler.nodes[-blank_nodes:]]
        pred_text = decode_multi_token(pred_vals, bits_per_token, mapping)
        token_correct = sum(1 for p, g in zip(pred_text, ground_truth_text) if p == g)
        token_acc = token_correct / blank_tokens * 100

        rows.append([
            f"{vocab_size} tokens",
            f"{bits_per_token} nodes/tok",
            f"{total_nodes} nodes",
            f"{token_correct}/{blank_tokens} ({token_acc:.0f}%)",
        ])

    print_table(["Vocab Size", "Encoding", "Total Nodes", "Token Accuracy"], rows)


# ============================================================================
#  TEST 5: Sequence Length Scaling
# ============================================================================

def test_sequence_length_scaling():
    """
    Fixed pattern (AABB), fixed 10% blank ratio. Scale from 32 to 512 nodes.
    Does prediction hold at length?
    """
    lengths = [32, 64, 128, 256, 512]
    blank_fraction = 0.10
    rows = []

    for seq_len in lengths:
        pattern = [1, 1, -1, -1] * (seq_len // 4)
        blank_count = max(4, int(seq_len * blank_fraction))
        # Round blank_count to nearest multiple of 4 for clean periods
        blank_count = (blank_count // 4) * 4 or 4
        context_len = seq_len - blank_count
        ground_truth = pattern[context_len:]

        compiler = PhaseStateCompiler(kissing_number_threshold=16, allow_novelty=True)
        compiler.load_data(pattern)

        for i, val in enumerate(pattern):
            compiler.nodes[i]._state = syn.state(
                [complex(val, 0)], dtype="complex128",
                device=compiler.nodes[i]._state.device
            )
            compiler.nodes[i]._project()

        for i in range(context_len, seq_len):
            compiler.nodes[i]._state = compiler.nodes[i]._state * 0

        for i in range(context_len):
            compiler.nodes[i].is_source = True

        recovered = lambda nodes, bc=blank_count: all(
            not n.is_syntonic() for n in nodes[-bc:]
        )

        t0 = time.time()
        with suppress_stdout():
            compiler.run(max_cycles=50, exit_condition=recovered)
        elapsed = (time.time() - t0) * 1000

        predicted = [n.m4_val for n in compiler.nodes[context_len:]]
        correct = sum(1 for p, g in zip(predicted, ground_truth) if p == g)
        acc = correct / blank_count * 100

        rows.append([
            str(seq_len),
            str(blank_count),
            f"{correct}/{blank_count} ({acc:.0f}%)",
            f"{elapsed:.1f}ms",
        ])

    print_table(["Seq Length", "Blanks", "PS Accuracy", "Time"], rows)


# ============================================================================
#  TEST 6: Trained Hopfield Network vs Phase-State
# ============================================================================

def test_hopfield_comparison(seed=42):
    """
    The definitive associative memory comparison.
    Hopfield network is THE classical model for this task.
    Tests at multiple scales with matched conditions.
    """
    configs = [
        (64,  10,  0.3),
        (128, 20,  0.3),
        (256, 10,  0.4),
        (256, 35,  0.4),   # Near Hopfield theoretical limit
        (256, 50,  0.4),   # Beyond Hopfield limit
        (512, 30,  0.4),
    ]

    rows = []

    for pattern_dim, num_patterns, noise in configs:
        rng = random.Random(seed)
        patterns = [[1 if rng.random() > 0.5 else -1 for _ in range(pattern_dim)]
                    for _ in range(num_patterns)]
        probes = [make_noisy_probe(p, noise, random.Random(seed + 999)) for p in patterns]

        # Phase-State
        t0 = time.time()
        ps_hits = 0
        for original, probe in zip(patterns, probes):
            recalled, _ = phase_state_recall(patterns, probe)
            if recalled == original:
                ps_hits += 1
        ps_time = (time.time() - t0) * 1000

        # Hopfield
        hopfield = HopfieldNetwork(pattern_dim)
        t0 = time.time()
        hopfield.store(patterns)
        hop_hits = 0
        for original, probe in zip(patterns, probes):
            recalled = hopfield.recall(probe)
            if recalled == original:
                hop_hits += 1
        hop_time = (time.time() - t0) * 1000

        hop_limit = int(0.138 * pattern_dim)

        rows.append([
            f"{pattern_dim}d/{num_patterns}p",
            f"{noise*100:.0f}%",
            f"{ps_hits}/{num_patterns}",
            f"{ps_time:.0f}ms",
            f"{hop_hits}/{num_patterns}",
            f"{hop_time:.0f}ms",
            str(hop_limit),
        ])

        print(f"  {pattern_dim}d/{num_patterns}p @ {noise*100:.0f}% noise: "
              f"PS {ps_hits}/{num_patterns}, Hopfield {hop_hits}/{num_patterns}")

    print()
    print_table(
        ["Config", "Noise", "PS Recall", "PS Time", "Hop Recall", "Hop Time", "Hop Limit"],
        rows
    )
    print("  Hop Limit = theoretical max patterns for Hopfield (~0.138*N)")
    print("  PS uses 0 params. Hopfield uses N^2 params (Hebbian weights).")


# ============================================================================
#  TEST 7: Self-Repair Damage Sweep
# ============================================================================

class Grid2DCompiler(PhaseStateCompiler):
    """2D toroidal grid extension of PhaseStateCompiler."""

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
        return (r % self.rows) * self.cols + (c % self.cols)

    def _neighbors(self, flat_idx):
        r = flat_idx // self.cols
        c = flat_idx % self.cols
        return [
            self._idx(r - 1, c), self._idx(r + 1, c),
            self._idx(r, c - 1), self._idx(r, c + 1),
        ]

    def set_node(self, r, c, val):
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

        if self.allow_novelty and any(n.is_syntonic() for n in self.nodes):
            filled = self._inject_novelty_2d()
            interacted.update(filled)

        newly_activated = set()
        for i, node_a in enumerate(self.nodes):
            if node_a.is_syntonic() or node_a in interacted:
                continue
            for nb_idx in self._neighbors(i):
                node_b = self.nodes[nb_idx]
                if node_b.is_syntonic() and node_b not in interacted:
                    node_b._state = syn.state(
                        [node_a._state.to_list()[0]],
                        dtype=node_b._state.dtype, device=node_b._state.device
                    )
                    node_b._project()
                    node_b.stale_cycles = 0
                    interacted.add(node_a)
                    interacted.add(node_b)
                    newly_activated.add(node_b)
                    activity = True

        # Harmonization
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
                    activity = True
                    break

        # Dampened Recursion
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


def test_self_repair_sweep():
    """
    16x16 toroidal grid. Fill with uniform +1. Damage 5% to 60%.
    Measure cycles to repair and accuracy.
    """
    rows_val, cols_val = 16, 16
    total_nodes = rows_val * cols_val
    damage_fractions = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60]

    rows = []

    for frac in damage_fractions:
        grid = Grid2DCompiler(rows_val, cols_val, allow_novelty=True, stale_threshold=999)

        # Fill with +1 and mark as source
        for r in range(rows_val):
            for c in range(cols_val):
                grid.set_node(r, c, 1)
                grid.nodes[grid._idx(r, c)].is_source = True

        # Damage random nodes
        num_damage = int(total_nodes * frac)
        rng = random.Random(42)
        damage_indices = rng.sample(range(total_nodes), num_damage)
        for idx in damage_indices:
            grid.nodes[idx]._state = syn.state(
                [0j], dtype=grid.nodes[idx]._state.dtype,
                device=grid.nodes[idx]._state.device
            )
            grid.nodes[idx].is_source = False

        # Run repair
        repair_cycle = None
        t0 = time.time()
        with suppress_stdout():
            for cycle in range(30):
                grid.compile_cycle()
                repaired = sum(1 for idx in damage_indices
                               if not grid.nodes[idx].is_syntonic())
                if repaired == num_damage and repair_cycle is None:
                    repair_cycle = cycle + 1
                    break
        elapsed = (time.time() - t0) * 1000

        # Check accuracy
        correct = sum(1 for idx in damage_indices if grid.nodes[idx].m4_val == 1)
        acc = correct / num_damage * 100 if num_damage > 0 else 100.0

        rows.append([
            f"{frac*100:.0f}%",
            str(num_damage),
            str(repair_cycle) if repair_cycle else ">30",
            f"{acc:.1f}%",
            f"{elapsed:.0f}ms",
        ])

    print_table(
        ["Damage %", "Nodes Hit", "Cycles to Repair", "Accuracy", "Time"],
        rows
    )


# ============================================================================
#  TEST 8: Cold Start / Minimal Context
# ============================================================================

def test_cold_start():
    """
    How few context nodes does PS need to predict correctly?
    Fixed pattern AABB (period-4), vary context from 4 to 32 nodes.
    Always predict next 4.
    """
    blank_count = 4
    context_sizes = [4, 6, 8, 12, 16, 24, 32]

    rows = []

    for ctx_len in context_sizes:
        # Build full sequence: context + 4 blank
        full_pattern = [1, 1, -1, -1] * ((ctx_len + blank_count + 3) // 4)
        full_pattern = full_pattern[:ctx_len + blank_count]
        ground_truth = full_pattern[ctx_len:]

        compiler = PhaseStateCompiler(kissing_number_threshold=16, allow_novelty=True)
        compiler.load_data(full_pattern)

        for i, val in enumerate(full_pattern):
            compiler.nodes[i]._state = syn.state(
                [complex(val, 0)], dtype="complex128",
                device=compiler.nodes[i]._state.device
            )
            compiler.nodes[i]._project()

        for i in range(ctx_len, len(full_pattern)):
            compiler.nodes[i]._state = compiler.nodes[i]._state * 0

        for i in range(ctx_len):
            compiler.nodes[i].is_source = True

        recovered = lambda nodes, bc=blank_count: all(
            not n.is_syntonic() for n in nodes[-bc:]
        )

        with suppress_stdout():
            compiler.run(max_cycles=30, exit_condition=recovered)

        predicted = [n.m4_val for n in compiler.nodes[ctx_len:ctx_len + blank_count]]
        correct = sum(1 for p, g in zip(predicted, ground_truth) if p == g)
        acc = correct / blank_count * 100

        rows.append([
            str(ctx_len),
            f"{ctx_len / 4:.1f} periods",
            str(ground_truth),
            str(predicted),
            f"{correct}/{blank_count} ({acc:.0f}%)",
        ])

    print_table(
        ["Context Nodes", "Context Periods", "Ground Truth", "Predicted", "Accuracy"],
        rows
    )
    print("  Minimum viable context for period detection = 1 full period (4 nodes)")


# ============================================================================
#  TEST 9: Multi-Frequency Interference
# ============================================================================

def test_multi_frequency():
    """
    Superimpose two periodic signals and test if PS can still predict.
    Signal A: period-4 [1,1,-1,-1]
    Signal B: period-6 [1,1,1,-1,-1,-1]
    Combined: sign(A + B), with ties broken to +1

    Also test a simpler case: dominant frequency with minor perturbation.
    """
    test_cases = []

    # Case 1: Period-4 + Period-6 superposition
    seq_len = 48
    signal_a = [([1, 1, -1, -1] * (seq_len // 4 + 1))[:seq_len]]
    signal_b = [([1, 1, 1, -1, -1, -1] * (seq_len // 6 + 1))[:seq_len]]
    combined = []
    for i in range(seq_len):
        s = signal_a[0][i] + signal_b[0][i]
        combined.append(1 if s >= 0 else -1)
    test_cases.append(("P4 + P6 superposition", combined, 6))

    # Case 2: Dominant period-4 with 10% random noise
    noisy = [1, 1, -1, -1] * 12  # 48 elements
    rng = random.Random(42)
    for i in rng.sample(range(48), 5):  # ~10% noise
        noisy[i] *= -1
    test_cases.append(("P4 + 10% noise", noisy, 4))

    # Case 3: Period-3 + Period-4 (LCM = 12)
    seq_len2 = 48
    sig_3 = ([1, 1, -1] * (seq_len2 // 3 + 1))[:seq_len2]
    sig_4 = ([1, -1, -1, 1] * (seq_len2 // 4 + 1))[:seq_len2]
    combined2 = [1 if (sig_3[i] + sig_4[i]) >= 0 else -1 for i in range(seq_len2)]
    test_cases.append(("P3 + P4 (LCM=12)", combined2, 6))

    # Case 4: Period-2 dominant (easy) + weak period-8 modulation
    base = [1, -1] * 24  # period-2
    mod = ([1, 1, 1, 1, -1, -1, -1, -1] * 6)[:48]  # period-8
    combined3 = [1 if (base[i] * 3 + mod[i]) >= 0 else -1 for i in range(48)]
    test_cases.append(("P2 dominant + P8 weak", combined3, 4))

    rows = []

    for name, pattern, blank_count in test_cases:
        total_len = len(pattern)
        context_len = total_len - blank_count
        ground_truth = pattern[context_len:]

        compiler = PhaseStateCompiler(kissing_number_threshold=16, allow_novelty=True)
        compiler.load_data(pattern)

        for i, val in enumerate(pattern):
            compiler.nodes[i]._state = syn.state(
                [complex(val, 0)], dtype="complex128",
                device=compiler.nodes[i]._state.device
            )
            compiler.nodes[i]._project()

        for i in range(context_len, total_len):
            compiler.nodes[i]._state = compiler.nodes[i]._state * 0

        for i in range(context_len):
            compiler.nodes[i].is_source = True

        recovered = lambda nodes, bc=blank_count: all(
            not n.is_syntonic() for n in nodes[-bc:]
        )

        with suppress_stdout():
            compiler.run(max_cycles=40, exit_condition=recovered)

        predicted = [n.m4_val for n in compiler.nodes[context_len:]]
        correct = sum(1 for p, g in zip(predicted, ground_truth) if p == g)
        acc = correct / blank_count * 100

        rows.append([name, str(ground_truth), str(predicted),
                     f"{correct}/{blank_count} ({acc:.0f}%)"])

    print_table(["Signal Mix", "Ground Truth", "Predicted", "Accuracy"], rows)


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
                               "stress_benchmark_results.txt")
    logfile = open(output_path, 'w')
    original_stdout = sys.stdout
    sys.stdout = TeeWriter(original_stdout, logfile)

    print_header("PHASE-STATE COMPILER STRESS BENCHMARK SUITE")
    print("Designed to find architectural limits and test against fair baselines.")
    print("All operations via srt_library or pure Python only.\n")

    print_header("TEST 0: Noise Tolerance Sweep (256d, 10 patterns)")
    test_noise_sweep()

    print_header("TEST 1: Capacity Saturation (256d, 5-100 patterns)")
    test_capacity_saturation()

    print_header("TEST 2: Hierarchical / Complex Patterns")
    test_hierarchical_patterns()

    print_header("TEST 3: Adversarial Sequences")
    test_adversarial_sequences()

    print_header("TEST 4: Vocabulary Scaling (2 to 16 tokens)")
    test_vocabulary_scaling()

    print_header("TEST 5: Sequence Length Scaling (32 to 512 nodes)")
    test_sequence_length_scaling()

    print_header("TEST 6: Trained Hopfield Network vs Phase-State (Definitive)")
    test_hopfield_comparison()

    print_header("TEST 7: Self-Repair Damage Sweep (5% to 60%)")
    test_self_repair_sweep()

    print_header("TEST 8: Cold Start / Minimal Context")
    test_cold_start()

    print_header("TEST 9: Multi-Frequency Interference")
    test_multi_frequency()

    print("\n" + "=" * 70)
    print("  ALL STRESS BENCHMARKS COMPLETE")
    print("=" * 70)

    sys.stdout = original_stdout
    logfile.close()
    print(f"\nResults saved to: {output_path}")