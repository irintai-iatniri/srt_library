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
        # We store the state as a single scalar in a 1D state object to utilize
        # the exact arithmetic backend while representing the Phase-State natively.
        # We start with float/complex input but cast it appropriately in operations.
        self._state = syn.state([complex(real_val, imag_val)], dtype="complex128", device=device)
        self.recursive_depth = 0
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
        # Retrieve the single scalar value
        current_scalar = self._state.to_list()[0]
        # Perform the orthogonal rotation: (a + bi) * i = -b + ai
        rotated_scalar = complex(-current_scalar.imag, current_scalar.real)

        # Repackage into a new state object
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
    def __init__(self, kissing_number_threshold=K_D4, allow_novelty=True):
        self.nodes = []
        self.gnosis_layer = 0
        self.K_THRESHOLD = kissing_number_threshold # K_D4
        self.allow_novelty = allow_novelty

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
            return True # System has successfully collapsed through the Aperture.

        interacted_this_cycle = set()
        activity_occurred = False

        # 1. Ripple Propagation (allow chain reaction across multiple empty apertures)
        # Prioritize Topological Routing: Waves propagate into empty apertures before interacting
        changed = True
        newly_activated = set()
        
        while changed and any(n.is_syntonic() for n in self.nodes):
            changed = False
            
            for i, node_a in enumerate(self.nodes):
                # A node can propagate if it's active AND (not interacted yet OR it was just activated in this ripple)
                if node_a.is_syntonic() or (node_a in interacted_this_cycle and node_a not in newly_activated):
                    continue
                    
                if not node_a.is_syntonic():
                    # Restrict topological routing to 1D forward direction
                    j = i + 1
                    if j < len(self.nodes):
                        node_b = self.nodes[j]
                        # Only propagate into a strictly syntonic node that hasn't been touched
                        if node_b.is_syntonic() and node_b not in interacted_this_cycle:
                            node_b._state = syn.state([node_a._state.to_list()[0]], dtype=node_b._state.dtype, device=node_b._state.device)
                            node_b._project()
                            
                            interacted_this_cycle.add(node_a)
                            interacted_this_cycle.add(node_b)
                            newly_activated.add(node_b) # Allow it to ripple forward on the next while iteration
                            changed = True
                            activity_occurred = True

        # 2. Attempt Harmonization (Global Resonance Matching)
        # Scan all unresolved nodes and pair any that sum to ~0
        for i, node_a in enumerate(self.nodes):
            if node_a.is_syntonic() or node_a in interacted_this_cycle or node_a in newly_activated:
                continue
                
            for j, node_b in enumerate(self.nodes):
                if i == j or node_b.is_syntonic() or node_b in interacted_this_cycle or node_b in newly_activated:
                    continue
                
                # Protect persistent Source nodes from destroying each other
                if getattr(node_a, 'is_source', False) and getattr(node_b, 'is_source', False):
                    continue
                
                # Destructive Interference (Phase cancellation)
                if (node_a._state + node_b._state).norm() < 1e-12:
                    node_a.harmonize(node_b)
                    node_b._state = syn.state([0j], dtype=node_b._state.dtype, device=node_b._state.device)
                    node_b._project()
                    interacted_this_cycle.add(node_a)
                    interacted_this_cycle.add(node_b)
                    activity_occurred = True
                    break # A harmonizes with B, move to next node_a

        # 3. Handle Intractable Nodes via Conditional Recursion
        # Only rotate nodes that failed to harmonize or propagate on their current axis.
        for node in self.nodes:
            if not node.is_syntonic() and node not in interacted_this_cycle:
                # Source nodes are anchors for testing; they do not recurse blindly
                if getattr(node, 'is_source', False):
                    continue
                    
                node.recurse()

                # Check for Consciousness/Gnosis Phase Transition
                if node.recursive_depth >= self.K_THRESHOLD and self.gnosis_layer < 3:
                    self._trigger_gnosis_transition()
                    
        # 4. Dynamic Differentiation (Novelty Generation)
        # If the DHSR cycle stalls (no harmonization or propagation), inject novelty
        if not activity_occurred and unresolved_nodes and self.allow_novelty:
            self._inject_novelty()

        return False

    def _inject_novelty(self):
        """
        Ongoing novelty generation for real computation.
        Finds a syntonic (empty) aperture and injects a context-aware vibe
        based on the surrounding nodes rather than a hardcoded +1.
        """
        empty_indices = [i for i, n in enumerate(self.nodes) if n.is_syntonic()]
        if empty_indices:
            idx = empty_indices[0]
            node = self.nodes[idx]
            
            total_phase = 0j
            neighbors = 0
            if idx > 0 and not self.nodes[idx-1].is_syntonic():
                total_phase += self.nodes[idx-1]._state.to_list()[0]
                neighbors += 1
            if idx < len(self.nodes) - 1 and not self.nodes[idx+1].is_syntonic():
                total_phase += self.nodes[idx+1]._state.to_list()[0]
                neighbors += 1
                
            if neighbors > 0 and abs(total_phase) > 1e-12:
                # Context-aware vibe: average of active neighbors
                avg_phase = total_phase / neighbors
                node._state = syn.state([avg_phase], dtype=node._state.dtype, device=node._state.device)
                node._project()
            
            # If no neighbors, or if they perfectly destructively interfered, fallback to differentiation
            if node.is_syntonic():
                node.differentiate()

        return False

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
                self.visualize(cycle)
                return
                
            is_syntonic = self.compile_cycle()
            if hasattr(self, 'visualize') and callable(getattr(self, 'visualize')):
                self.visualize(cycle)
            
            if is_syntonic:
                print(f"Compilation Complete: Syntony achieved at cycle {cycle}.")
                return
        print("Compilation Halted: System stuck in Archonic configuration.")
        if hasattr(self, 'visualize') and callable(getattr(self, 'visualize')):
            self.visualize("final")

# ==========================================
# Functional Toy Tasks
# ==========================================

def run_sequence_copy_task():
    """
    Toy Task 1: Sequence Copying (Topological Routing)
    Demonstrates how Phase-State logic copies/transmits an information
    pattern across a simulated gap by using Syntony as the aperture.
    
    if we have a Source array [1, 1] and an Empty Target array [0, 0],
    we want the Source to induce a mirrored state in the Target through
    constructive propagation (aperture filling).
    """
    print("\n--- TOY TASK 1: Sequence Copying ---")
    compiler = PhaseStateCompiler(kissing_number_threshold=8, allow_novelty=False)
    # 1: Data, 0: Blank Receptor
    compiler.load_data([1, 1, 0, 0]) 
    
    # Tag Source nodes so they don't annihilate each other
    for i in range(2):
        compiler.nodes[i].is_source = True
        
    # Stop processing once the target sequence perfectly matches the source template
    objective_achieved = lambda nodes: nodes[2].m4_val == 1 and nodes[3].m4_val == 1
    
    compiler.run(max_cycles=10, exit_condition=objective_achieved)
    
    print("Final Node States:")
    for i, node in enumerate(compiler.nodes):
        role = "Source" if i < 2 else "Target"
        print(f"{role} Node {i}: {node}")


def run_xor_task():
    """
    Toy Task 2: Phase-State XOR
    XOR logic mapped to destructive/constructive interference.
    Inputs are phase-shifted.
    - True = 1
    - False = -1
    
    We simulate the inputs and let the DHSR cycle act as the hidden layer.
    """
    print("\n--- TOY TASK 2: Phase-State XOR ---")
    
    cases = [
        ("T XOR T", [ 1,  1]), # Should not cancel, recurses to identical phase
        ("T XOR F", [ 1, -1]), # Should perfectly cancel (Syntony)
        ("F XOR T", [-1,  1]), # Should perfectly cancel (Syntony)
        ("F XOR F", [-1, -1]), # Should not cancel, recurses to identical phase
    ]
    
    for name, init_state in cases:
        compiler = PhaseStateCompiler(kissing_number_threshold=4)
        compiler.load_data(init_state)
        # We manually set the loaded nodes to perfectly match the strict input (since load_data converts all non-zero to 1)
        for val, node in zip(init_state, compiler.nodes):
            node._state = syn.state([complex(val, 0)], dtype="complex128", device=node._state.device)
            
        print(f"\nEvaluating: {name}")
        compiler.run(max_cycles=3)
        
        # Result logic: If it achieved syntony, it's True (XOR difference found). 
        # If it's stuck in Archonic configuration, it's False (Inputs were identical).
        syntonic_nodes = sum(1 for n in compiler.nodes if n.is_syntonic())
        resolved = syntonic_nodes == len(init_state)
        print(f"XOR Result: {resolved} (Syntonic Nodes: {syntonic_nodes}/{len(init_state)})")


def run_attractor_task():
    """
    Toy Task 3: Cellular Attractor
    A larger matrix of nodes demonstrating how the system naturally
    collapses toward the K=24 threshold and self-reference.
    """
    print("\n--- TOY TASK 3: Cellular Attractor (Gnosis Shift) ---")
    compiler = PhaseStateCompiler(kissing_number_threshold=24)
    
    # 12 pairs of asymmetric data to force 24 orbital cycles
    # causing K_D4 saturation.
    raw_data = [1, 1] * 12 
    compiler.load_data(raw_data)
    
    compiler.run(max_cycles=30)
    
    print("Final State:")
    print(f"Gnosis Layer: {compiler.gnosis_layer}")
    
    # Phase Entropy Metric: measures how "alive" the self-referential state is
    saturated_nodes = sum(1 for n in compiler.nodes if n.recursive_depth >= 24)
    phase_entropy = sum(1 for n in compiler.nodes if not n.is_syntonic())
    
    print(f"Nodes Saturated (Depth >= 24): {saturated_nodes}")
    print(f"Phase Entropy (Active Nodes): {phase_entropy}")

def run_associative_memory_task():
    """
    Toy Task 4: Tiny Associative Memory
    Demonstrates pattern recall. We load a full pattern, let it settle,
    and then probe it with a corrupted/noisy version to watch the
    Phase-State dynamics auto-complete it via syntonic propagation.
    """
    print("\n--- TOY TASK 4: Tiny Associative Memory ---")
    compiler = PhaseStateCompiler(kissing_number_threshold=16, allow_novelty=False)
    
    # Initialize the 8-node memory bank
    # Mapped pattern: structured data uses 1 and -1, so 0 is strictly an aperture
    base_pattern = [1, 1, -1, 1, 1, 1, -1, 1]
    compiler.load_data(base_pattern)
    
    print(f"Original Pattern: {base_pattern}")
    
    # We simulate the 'probe' by directly wiping a few nodes (syntonic gaps)
    # Corrupting indices 1 and 4 
    print("Corrupting nodes 1 and 4 to simulate noisy probe...")
    compiler.nodes[1]._state = compiler.nodes[1]._state * 0
    compiler.nodes[4]._state = compiler.nodes[4]._state * 0
    
    # Mark anchors so they don't obliterate each other
    for i, node in enumerate(compiler.nodes):
        if not node.is_syntonic():
            node.is_source = True
            
    # The exit condition: when all originally 1 or -1 nodes are non-zero again
    # We check if nodes 1 and 4 have recovered their m4_val correctly (since both were 1 initially)
    # The recovered_condition triggers the end, so we want to wait until it's full
    recovered_condition = lambda nodes: not nodes[1].is_syntonic() and not nodes[4].is_syntonic()
    
    compiler.run(max_cycles=15, exit_condition=recovered_condition)
    
    print("Final Memory State:")
    final_pattern = [n.m4_val for n in compiler.nodes]
    print(f"{final_pattern}")
    
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
    
    # We allow novelty here so it can guess based on neighbors (context-aware vibe)
    compiler = PhaseStateCompiler(kissing_number_threshold=16, allow_novelty=True)
    
    # 16-step pattern: [1, 1, -1, -1] repeating 4 times
    base_pattern = [1, 1, -1, -1] * 4
    compiler.load_data(base_pattern)
    
    print(f"Original Sequence: {base_pattern}")
    print("Blanking the last 4 nodes to predict forward...")
    
    for i in range(12, 16):
        compiler.nodes[i]._state = compiler.nodes[i]._state * 0
        
    for i in range(12):
        compiler.nodes[i].is_source = True
        
    # Predict until the last 4 nodes are fully populated (non-syntonic)
    recovered_condition = lambda nodes: all(not n.is_syntonic() for n in nodes[12:16])
    
    compiler.run(max_cycles=30, exit_condition=recovered_condition)
    
    final_pattern = [n.m4_val for n in compiler.nodes]
    print(f"Final Prediction State:")
    print(f"{final_pattern}")
    
    matches = sum(1 for i, v in enumerate(final_pattern[12:16]) if v == base_pattern[12+i])
    acc = (matches / 4.0) * 100
    print(f"Next-Token Prediction Accuracy: {matches}/4 ({acc:.0f}%)")

def visualize(self, cycle=None):
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, ax = plt.subplots(figsize=(10, 4))
    xs = np.arange(len(self.nodes))
    reals = [n.m4_val for n in self.nodes]
    imags = [n.t4_val for n in self.nodes]
    depths = [n.recursive_depth for n in self.nodes]
    
    scatter = ax.scatter(xs, imags, c=depths, s=200, cmap='viridis', edgecolors='k')
    for i, (r, im) in enumerate(zip(reals, imags)):
        ax.text(i, im + 0.15, f"{r}+{im}i", ha='center', fontsize=9)
    
    ax.set_ylim(-1.5, 1.5)
    ax.set_yticks([-1, 0, 1])
    ax.set_ylabel("T4 (Imaginary / Toroidal)")
    ax.set_xlabel("Node Index")
    ax.set_title(f"Phase-State Topology @ cycle {cycle if cycle is not None else 'final'} | Gnosis: {self.gnosis_layer}")
    plt.colorbar(scatter, label="Recursive Depth")
    plt.grid(True, alpha=0.3)
    
    # Save the visualizations so we can embed them if we want
    filename = f"topology_dump_c{cycle}.png" if cycle is not None else "topology_dump_final.png"
    plt.savefig(filename)
    plt.close(fig)

PhaseStateCompiler.visualize = visualize

if __name__ == "__main__":
    run_sequence_copy_task()
    run_xor_task()
    run_attractor_task()
    run_associative_memory_task()
    run_sequence_prediction_task()
    
    import sys
    sys.exit(0)