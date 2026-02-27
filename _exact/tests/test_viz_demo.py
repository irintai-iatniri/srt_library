"""
NetworkVisualizer Demo - Bidirectional Resonance Loop

Demonstrates Phase 3C visualization capabilities:
1. Build 2-neuron bidirectional loop (n1 → n2 → n1 with delays)
2. Plot topology showing curved edges with delay/sensor labels
3. Send 50-tick pulse sequence (1000 followed by silence)
4. Plot resonance heatmap showing signal bouncing between neurons

Expected Behavior:
- Topology: Green input node (n0), red output node (n1), curved loop edges
- Heatmap: Signal bounces back and forth with delay=10 spacing
"""

from syntonic_applications.sna import SyntonicNetwork, ResonantOscillator
from syntonic_applications.sna.network_viz import NetworkVisualizer
import matplotlib.pyplot as plt


def main():
    """Build bidirectional loop and visualize topology + resonance activity."""
    
    # Build 2-neuron bidirectional loop
    net = SyntonicNetwork()
    
    # Create two neurons with tuned thresholds for visible spiking
    n0 = net.add_neuron(ResonantOscillator(input_count=2, kernel_taps=31, threshold=200))
    n1 = net.add_neuron(ResonantOscillator(input_count=2, kernel_taps=31, threshold=200))
    
    # n0 → n1 with delay=10
    net.add_synapse(source=n0, target=n1, sensor_idx=0, delay=10)
    
    # n1 → n0 with delay=10 (creates loop)
    net.add_synapse(source=n1, target=n0, sensor_idx=1, delay=10)
    
    # Register n0 as input (receives external stimulus)
    # Register BOTH neurons as outputs (we'll observe activity from both)
    net.register_input(n0)
    net.register_output(n0)
    net.register_output(n1)
    
    print("Network built:")
    print(f"  Neurons: {net.neuron_count()}")
    print(f"  Axons: {len(net.inspect_topology())}")
    print(f"  Inputs: {net.inspect_io()[0]}")
    print(f"  Outputs: {net.inspect_io()[1]}")
    
    # Create visualizer
    viz = NetworkVisualizer(net)
    
    # Plot topology
    print("\nPlotting topology...")
    fig_topo = viz.plot_topology(
        title="Bidirectional Resonance Loop (Delay=10)",
        seed=42,
        font_size=12,
        node_size=1000,
    )
    plt.savefig("topology.png", dpi=150, bbox_inches="tight")
    print("  Saved: topology.png")
    
    # Run 50-tick pulse sequence with strong input
    print("\nRunning 50-tick pulse sequence...")
    input_sequence = [[2000]] + [[0]] * 49  # Strong pulse at t=0, silence after
    output_sequence = net.process_sequence(input_sequence)
    
    print(f"  Input shape: {len(input_sequence)} ticks × {len(input_sequence[0])} channels")
    print(f"  Output shape: {len(output_sequence)} ticks × {len(output_sequence[0])} neurons")
    
    # Count spikes
    spike_count = sum(1 for out in output_sequence for val in out if val != 0)
    print(f"  Total spikes detected: {spike_count}")
    
    # Show spike times
    spike_times = []
    for t, out in enumerate(output_sequence):
        if any(val != 0 for val in out):
            spike_times.append(f"t={t}:{out}")
    if spike_times:
        print(f"  Spike times: {', '.join(spike_times[:10])}")  # Show first 10
    
    # Plot resonance heatmap
    print("\nPlotting resonance heatmap...")
    fig_heat = viz.plot_resonance_heatmap(
        output_sequence,
        title="Resonance Activity (50 Ticks)",
    )
    plt.savefig("heatmap.png", dpi=150, bbox_inches="tight")
    print("  Saved: heatmap.png")
    
    # Show plots
    print("\nDisplaying plots...")
    plt.show()
    
    print("\n✅ Demo complete!")


if __name__ == "__main__":
    main()
