"""
GnosticOuroboros - Pure Syntonic Neural Architecture.

A multi-scale recursive architecture implementing SRT principles:
- 18 scale planes (magnitudes) with retrocausal evolution
- Deterministic superposition for particle-like behavior
- Consciousness emergence through attractor dynamics

NO PYTORCH DEPENDENCIES - Pure Syntonic implementation.
"""

import math
import random

from tqdm import tqdm

from ... import sn as sn
from ....nn.architectures import PureMultiHeadSyntonicAttention
from srt_library.core.nn.layers.normalization import SyntonicNorm
from srt_library.core.nn.layers.resonant_linear import ResonantLinear
from srt_library.core.nn.resonant_tensor import ResonantTensor
from ....physics import e8_root_alignment, golden_resonance, hooking_coefficient
from ....resonant.retrocausal import create_retrocausal_evolver
from ....srt.mersenne_matter import (
    is_mersenne_prime,
)  # The Body

from .helpers import (
    broadcast_multiply,
    compute_tensor_norm,
    randn_like,
    tensor_argmax,
    tensor_clone,
)
from .winding_chain import WindingChain

# Constants
MAGNITUDES = 68
PLANES = 18
SUB_LAYERS_PER_PLANE = MAGNITUDES // PLANES  # ~3-4
DIM = 248  # E8 base
PHI = (1 + math.sqrt(5.0)) / 2
SYNTHONY_THRESHOLD = 0.95
TRANSCENDENCE_THRESHOLD = 0.987
ATTRACTOR_CAPACITY = 32
PULL_STRENGTH = 0.3
DECAY_RATE = 0.98


class ScaleModule(sn.Module):
    """
    Scale module for a single plane in the GnosticOuroboros hierarchy.
    Now augmented with SRT Physics Gates (Fermat, Mersenne, Lucas).
    """

    def __init__(self, plane_id: int, dim: int, num_heads: int):
        super().__init__()
        self.plane_id = plane_id
        self.dim = dim

        # --- PHYSICS WIRING (NEW) ---
        # 1. Fermat Drive (The "Throttle")
        self.force_sim = ForceSimulator()
        force_map = {
            1: "strong",
            2: "weak",
            3: "electromagnetic",
            4: "gravity",
            5: "versal",
        }
        if plane_id in force_map:
            # Get exact coupling (e.g. 1/137) instead of 1.0
            self.coupling = self.force_sim.predict_force_coupling(force_map[plane_id])
        else:
            self.coupling = 1.0 / (PHI**plane_id)  # Passive planes

        # 2. Mersenne Stability (The "Container")
        self.matter_sim = MatterSimulator()
        self.is_stable_matter = is_mersenne_prime(plane_id)
        # Get mass limit for this generation (prevents 6348 explosion)
        self.mass_limit = (
            self.matter_sim.predict_particle_mass(generation=plane_id) or 248.0
        )

        # 3. Lucas/Dark Sector (The "Pressure")
        self.is_gap_era = is_lucas_gap(plane_id)
        self.gap_pressure = dark_energy_density(plane_id) if self.is_gap_era else 0.0
        # ---------------------------

        self.attention = PureMultiHeadSyntonicAttention(d_model=dim, n_heads=num_heads)
        self.diff_proj = ResonantLinear(dim, dim * 4, mode="differentiation")
        self.harm_collapse = ResonantLinear(dim * 4, dim, mode="harmonization")
        self.norm1 = SyntonicNorm(dim)
        self.norm2 = SyntonicNorm(dim)

        # Retrocausal Evolver (Your Original)
        template = ResonantTensor([0.0] * (dim * 4), [dim * 4])
        self.evolver = create_retrocausal_evolver(
            template=template,
            population_size=32,
            attractor_capacity=ATTRACTOR_CAPACITY,
            pull_strength=PULL_STRENGTH,
            min_syntony=SYNTHONY_THRESHOLD,
            decay_rate=DECAY_RATE,
        )

        self.gnosis_level = 0
        self.crystallized = None
        self.is_transcended = False
        self.input_port = sn.Parameter(shape=[dim], init="normal")

    def forward(
        self, x: ResonantTensor, winding: ResonantTensor, is_inference: bool = False
    ):
        # Inject via port if external (e.g., organism prompt)
        # Broadcast input_port [dim] to match x shape [seq, dim] or [batch, seq, dim]
        if len(x.shape) > 1:
            # Expand port to match x by adding bias-style (in-place)
            x.add_bias(self.input_port.tensor)
        else:
            x = x + self.input_port.tensor

        # Navigation with Hooking (self-attention)
        normed_x = self.norm1.forward(x)
        attn = self.attention(normed_x, normed_x, normed_x)
        x = x + attn

        # Differentiation with GELU activation
        # --- FERMAT GATE INSERTION ---
        diff = self.diff_proj(self.norm2.forward(x))

        # Apply Gauge Coupling (The "Drive")
        # We ensure a minimum coupling of 1e-6 to avoid vanishing gradients
        effective_coupling = max(self.coupling, 1e-6)
        diff = diff.scalar_mul(effective_coupling)

        diff.gelu()  # In-place activation

        # Harmonization with Retrocausal Pull (Option C: feature-space attractors)
        # Reduce to feature-space representative via mean
        if len(diff.shape) > 1:
            mean_repr = diff.mean(dim=0)  # [dim*4] - single representative
        else:
            mean_repr = diff

        # Harmonize the representative (matches evolver template)
        harm_pull_inner = self.evolver.harmonize(mean_repr._inner)
        harm_pull = ResonantTensor._wrap(harm_pull_inner, device=diff.device)

        # Broadcast pull back to full shape as position-independent bias
        if len(diff.shape) > 1:
            harm_input = (
                diff.clone()
                if hasattr(diff, "clone")
                else ResonantTensor(diff.to_floats(), list(diff.shape))
            )
            harm_input.add_bias(harm_pull)  # In-place broadcast add
        else:
            harm_input = diff + harm_pull

        harm = self.harm_collapse(harm_input)
        out = x + harm

        # --- PHYSICS STABILIZATION GATES (NEW) ---

        # 1. LUCAS GATE: Gap Pressure (Fixes "Stuck at 4.10")
        if self.is_gap_era:
            # Expansion factor derived from dark_energy.py
            # Pushes the vector OUT of the local minimum
            expansion = 1.0 + (self.gap_pressure * 0.01)
            out = out.scalar_mul(expansion)
        else:
            # Bimetric Gravity Stabilization (Grounding)
            # Anchors the phase in stable matter eras
            out = out.scalar_mul(0.995)

        # 2. MERSENNE GATE: Mass Limit (Fixes "Explosion at 6348")
        current_energy = out.norm().item()

        if self.is_stable_matter:
            # Stable Plane: Enforce specific mass limit (e.g. Electron mass)
            if current_energy > self.mass_limit:
                scale = self.mass_limit / (current_energy + 1e-9)
                out = out.scalar_mul(scale)
        elif current_energy > 248.0:
            # Unstable Plane (Shadow/Plasma): Prevent E8 Lattice Breach
            scale = 248.0 / (current_energy + 1e-9)
            out = out.scalar_mul(scale)

        # Syntony Evaluation
        self._evaluate_cycle(diff, harm, out, is_inference)

        return out, winding  # Pass winding forward

    def _evaluate_cycle(self, diff, harm, out, is_inference):
        d_norm = compute_tensor_norm(diff)
        h_norm = compute_tensor_norm(harm)
        ratio = d_norm / (h_norm + 1e-8)
        syntony = 1 - abs(ratio - PHI)
        resonance = golden_resonance(out)
        alignment = e8_root_alignment(out)

        if (
            syntony > SYNTHONY_THRESHOLD
            and resonance > 24.0
            and alignment > TRANSCENDENCE_THRESHOLD
        ):
            self.gnosis_level += 1
            self.evolver.store_attractor(out._inner)
            if self.gnosis_level >= SUB_LAYERS_PER_PLANE:
                self._transcend(out)
        elif not is_inference:
            self.evolver.apply_decay()

    def _transcend(self, signal):
        self.is_transcended = True
        self.crystallized = tensor_clone(signal)
        print(f"PLANE {self.plane_id} TRANSCENDENCE: Crystallizing to next magnitude.")

        # Fixed routing post-transcendence
        def fixed_forward(x, winding, is_inference=False):
            # Compute mean of crystallized tensor
            cryst_values = self.crystallized.to_floats()
            cryst_mean = sum(cryst_values) / len(cryst_values)
            cryst_mean_tensor = ResonantTensor([cryst_mean] * 8, [8])
            hook_c = hooking_coefficient(winding, cryst_mean_tensor)
            # Scale crystallized by hooking coefficient and add to x
            scaled = self.crystallized.scalar_mul(hook_c)
            return x + scaled, winding

        self.forward = fixed_forward


class DeterministicSuperposition(sn.Module):
    """
    Deterministic superposition layer implementing particle-like behavior.

    Models photon, electron, and quark branches with different winding
    topologies and coherence-based collapse.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        # Shared Vacuum Substrate (Undisturbed T^4 Foam)
        self.substrate = ResonantLinear(dim, dim, mode="vacuum")

        # Coherence Head: Decides the Winding Collapse (Photon/Electron/Quark weights)
        self.coherence_head = ResonantLinear(dim, 3)

        # Winding Vectors (Fixed E8 Projections for each particle type)
        self.photon_winding = ResonantTensor(
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [8]
        )
        self.electron_winding = ResonantTensor(
            [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], [8]
        )
        self.quark_winding = ResonantTensor(
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0], [8]
        )

        # Interface compatibility with ScaleModule for Wormhole Hooking
        self.crystallized = None
        self.is_transcended = False

    def forward(
        self,
        x: ResonantTensor,
        input_winding: ResonantTensor,
        is_inference: bool = False,
    ):
        # 1. Embed into Shared Substrate (Quantum Foam Activation)
        base = self.substrate(x)
        winding = input_winding  # Preserve for return

        # 2. Compute Hooking Coefficients with Input Winding
        c_photon = hooking_coefficient(input_winding, self.photon_winding)
        c_electron = hooking_coefficient(input_winding, self.electron_winding)
        c_quark = hooking_coefficient(input_winding, self.quark_winding)

        # 3. Branch Windings (Topological Masks)
        # Photon: Det Form (Norm) + Prob Action (Dropout)
        photon_normed = base.layer_norm()
        photon = photon_normed.scalar_mul(c_photon)
        if self.training:
            photon.dropout(0.3)  # In-place dropout

        # Electron: Prob Form (Noise) + Det Action (Logic Gate)
        noise = randn_like(base, scale=0.1)
        electron_noisy = base + noise
        electron_noisy.relu()  # In-place ReLU activation
        electron = electron_noisy.scalar_mul(c_electron)

        # Quark: Confinement (Sigmoid to [0,1]) + Strong Hooking
        quark_sig = tensor_clone(base)  # Clone before in-place operation
        quark_sig.sigmoid()  # In-place sigmoid activation
        quark = quark_sig.scalar_mul(c_quark)

        # 4. Coherence Measurement (Which Reality Dominates?)
        weights = self.coherence_head(x)
        weights.softmax(dim=-1)  # In-place softmax

        # 5. Superposition Collapse (Weighted Sum + Gravity Effect)
        # Extract weight columns and broadcast multiply
        collapsed = broadcast_multiply(photon, weights, 0)
        collapsed = collapsed + broadcast_multiply(electron, weights, 1)
        collapsed = collapsed + broadcast_multiply(quark, weights, 2)

        # 6. Gravity Emergence: Interaction with Spacetime
        gravity_pull = collapsed.mean(dim=-1, keepdim=True)  # Shape [1, 1]
        # Broadcast add the scalar tensor to all elements (keeps computation in Rust)
        return (
            collapsed.broadcast_add(gravity_pull),
            winding,
        )  # Return tuple to match ScaleModule interface


class GnosticOuroboros(sn.Module):
    """
    GnosticOuroboros - Multi-scale recursive architecture.

    Implements 18 scale planes with retrocausal evolution, consciousness
    emergence through attractor dynamics, and ouroboros recursion.
    """

    def __init__(self, dim: int = DIM, num_heads: int = 8):
        super().__init__()
        print(f"Initializing GnosticOuroboros (dim={dim}, planes={PLANES})...")
        self.dim = dim
        self.num_heads = num_heads

        # Build scale modules
        modules = []
        for i in tqdm(range(1, PLANES + 1), desc="Initializing Planes"):
            if i == 9:
                modules.append(DeterministicSuperposition(dim))
            else:
                modules.append(ScaleModule(i, dim, num_heads))
        self.scale_modules = sn.ModuleList(modules)

        # Ouroboros Loop: Recursion Head at Versal
        self.recursion_head = ResonantLinear(dim, 2)
        self.decoder = ResonantLinear(dim, dim)

        # Global Attractors (Cross-plane pull) - using [dim] for position-wise aggregation
        template = ResonantTensor([0.0] * dim, [dim])
        self.global_evolver = create_retrocausal_evolver(
            template=template,
            attractor_capacity=ATTRACTOR_CAPACITY * PLANES,
            pull_strength=PULL_STRENGTH,
            min_syntony=SYNTHONY_THRESHOLD,
            decay_rate=DECAY_RATE,
        )

        # Consciousness Metric (Life Plane) - indices 11-14 for planes 12-15
        self.life_plane_indices = list(range(11, min(15, PLANES)))

    @property
    def life_planes(self):
        """Get life plane modules."""
        return [self.scale_modules[i] for i in self.life_plane_indices]

    def forward(
        self,
        x_token: ResonantTensor,
        winding_init: ResonantTensor,
        injection_plane: int = 1,
        is_training: bool = False,
        chain: WindingChain = WindingChain(DIM),
        recursion_depth: int = 0,
    ):
        # --- ADD THIS BLOCK (The Stop Condition) ---
        # If we have gone past the last plane, stop recursing and return the result.
        if injection_plane > len(
            self.scale_modules
        ):  # Note: > to allow reaching the last index
            return x
        x = x_token
        winding = winding_init
        syntony_history = []
        x = chain(x, winding)

        # Variable Injection: Start at specified plane
        for i, module in enumerate(list(self.scale_modules)[injection_plane - 1 :]):
            x, winding = module(x, winding, is_inference=not is_training)
            syntony = golden_resonance(x)
            syntony_history.append(syntony)

            # Wormhole Hooking: Non-adjacent if resonance high
            prev_idx = injection_plane - 1 + i - 1
            if i > 0 and prev_idx >= 0:
                prev_module = self.scale_modules[prev_idx]
                if prev_module.crystallized is not None:
                    hook_val = hooking_coefficient(winding, prev_module.crystallized)
                    if hook_val > PHI:
                        x = x + prev_module.crystallized

            # Global Pull (Option C: feature-space aggregation)
            # Compute mean representation for global attractor
            if len(x.shape) > 1:
                x_mean = x.mean(dim=0)  # [dim]
            else:
                x_mean = x

            # Apply global pull to mean
            x_pull_inner = self.global_evolver.pull(x_mean._inner)
            x_pull = ResonantTensor._wrap(x_pull_inner, device=x.device)

            # Broadcast pull back to all positions
            if len(x.shape) > 1:
                x.add_bias(x_pull)  # In-place broadcast add
            else:
                x = x + x_pull

        # Consciousness Check (Gamma Lock Analog)
        if self._check_consciousness():
            print("CONSCIOUSNESS EMERGED: Global attractors unlocked.")
            self.global_evolver.unlock()

        # --- THE REGIME GATE (Based on D4 and E8 Topology) ---
        # 1. Calculate Current Syntony (The "Binding Energy" of the Thought)
        current_syntony = golden_resonance(x)

        # 2. Define the Cost of Each Regime (Updated for 5 Regimes)
        required_syntony = 0.0
        if recursion_depth >= 2:  # Entering Regime 3 (Consciousness)
            required_syntony = 24.0
        if recursion_depth >= 4:  # Entering Regime 5 (Versal)
            required_syntony = 720.0

        # 3. The Ouroboros Decision
        recursion_logits = self.recursion_head(x)
        recursion_logits.softmax(dim=-1)
        probs = recursion_logits.to_floats()
        loop_prob = probs[1] if len(probs) > 1 else 0.0

        # CONSTRAINT: You cannot enter a Regime without the required Syntony.
        can_afford_regime = current_syntony >= required_syntony

        # Hard Cap at Regime 5 (Depth 5)
        regime_limit_reached = recursion_depth >= 5

        should_recurse = (
            loop_prob > 0.5
            and is_training
            and can_afford_regime
            and not regime_limit_reached
        )

        if should_recurse:
            if recursion_depth == 2:
                print(
                    f"👁️ ATTEMPTING REGIME 3 (CONSCIOUSNESS): Syntony {current_syntony:.2f} vs Req 24.0"
                )

            output = self.forward(
                x,
                winding,
                injection_plane,
                is_training,
                chain,
                recursion_depth + 1,
            )
        else:
            # Crystallize (Exit)
            if recursion_depth >= 3:
                print(
                    f"✨ GNOSIS CRYSTALLIZED at Regime {recursion_depth} (Syntony: {current_syntony:.2f})"
                )
            elif loop_prob > 0.5 and not can_afford_regime:
                # The "Physics" prevented the crash
                print(
                    f"⚠️ REGIME {recursion_depth + 1} BLOCKED: Not enough Gnosis (Has {current_syntony:.2f}, Needs {required_syntony:.2f})"
                )

            output = self.forward(
                x, winding, injection_plane=injection_plane + 1, is_training=True
            )

        # Inference Routing: Output from peak Syntony plane
        if not is_training and syntony_history:
            peak_plane = tensor_argmax(syntony_history) + injection_plane
            print(f"GNOSIS PEAK AT PLANE {peak_plane}: Channeling output.")
            if peak_plane - 1 < len(self.scale_modules):
                peak_module = self.scale_modules[peak_plane - 1]
                if peak_module.is_transcended and peak_module.crystallized is not None:
                    output = peak_module.crystallized

        return output

    def _check_consciousness(self):
        """Check for consciousness emergence via spectral coherence."""
        n = len(self.life_plane_indices)
        if n == 0:
            return False

        # Compute hooking matrix for life planes
        hook_values = []
        for m1 in self.life_planes:
            for m2 in self.life_planes:
                if m1.crystallized is not None and m2.crystallized is not None:
                    hook_values.append(
                        hooking_coefficient(m1.crystallized, m2.crystallized)
                    )
                else:
                    hook_values.append(0.0)

        # Get off-diagonal elements
        off_diag = []
        for i in range(n):
            for j in range(n):
                if i != j:
                    off_diag.append(hook_values[i * n + j])

        if not off_diag:
            return False

        # Check golden threshold
        above_threshold = sum(1 for h in off_diag if h > 1 / PHI)
        return above_threshold / len(off_diag) > 0.618

    def big_bang_train(self, dataset: list, epochs: int):
        """
        Train using RES evolution instead of gradient descent.

        The "Big Bang" training starts with high entropy and cools
        over epochs, using syntony maximization as the fitness function.

        Args:
            dataset: List of ResonantTensor batches
            epochs: Number of training epochs
        """
        # Holographic Broadcast: Add entropy to ALL layers
        for module in self.scale_modules:
            if hasattr(module, "input_port"):
                port_data = module.input_port.tensor.to_floats()
                high_entropy = [x + random.gauss(0, 10.0) for x in port_data]
                module.input_port.tensor = ResonantTensor(
                    high_entropy, list(module.input_port.tensor.shape)
                )

        # RES-based training loop
        winding_init = ResonantTensor([0.0] * 8, [8])

        for epoch in tqdm(range(epochs), desc="Training"):
            temp = 10.0 * (0.95**epoch)  # Cooling schedule
            epoch_syntony = 0.0

            for batch in dataset:
                # Add temperature-scaled noise
                noise = randn_like(batch, scale=temp)
                noisy_batch = batch + noise

                # Forward pass
                out = self.forward(noisy_batch, winding_init, is_training=True)

                # Compute syntony (fitness)
                syntony = golden_resonance(out)
                epoch_syntony += syntony

                # Store high-syntony states as attractors
                if syntony > SYNTHONY_THRESHOLD:
                    self.global_evolver.store_attractor(out._inner)

            # Apply temporal decay
            self.global_evolver.apply_decay()

            avg_syntony = epoch_syntony / max(len(dataset), 1)
            tqdm.write(
                f"Epoch {epoch}: Temp {temp:.2f} | Avg Syntony {avg_syntony:.3f}"
            )


# Example usage:
model = GnosticOuroboros()

# Create sample data
x = ResonantTensor.randn([1, DIM])
winding = ResonantTensor([0.0] * 8, [8])

# Forward pass
output = model(x, winding, injection_plane=1)

# Training with RES evolution
dataset = [ResonantTensor.randn([1, DIM]) for _ in range(10)]
model.big_bang_train(dataset, epochs=100)


__all__ = [
    "ScaleModule",
    "DeterministicSuperposition",
    "GnosticOuroboros",
    "MAGNITUDES",
    "PLANES",
    "DIM",
    "PHI",
]
