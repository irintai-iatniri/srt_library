#!/usr/bin/env python3
"""
Test TRFT (Ternary Rational Fourier Transform) implementation
"""

# Test imports
try:
    from syntonic._core import (
        py_generate_resonance_ladder,
        py_ternary_decompose,
        py_ternary_synthesize,
    )
    print("✓ Successfully imported TRFT functions from syntonic._core")
except ImportError as e:
    print(f"✗ Failed to import TRFT functions: {e}")
    exit(1)

# Test 1: Generate resonance ladder
print("\n=== Test 1: Generate Resonance Ladder ===")
try:
    ladder = py_generate_resonance_ladder(500)
    print(f"✓ Generated {len(ladder)} Pythagorean triples")
    print(f"  First 5 triples: {ladder[:5]}")
    print(f"  Last 5 triples: {ladder[-5:]}")
except Exception as e:
    print(f"✗ Failed to generate ladder: {e}")
    exit(1)

# Test 2: Simple sine wave decomposition
print("\n=== Test 2: Pure Sine Wave Decomposition ===")
try:
    # Create a simple signal (approximation of sine wave)
    # Using integer approximation: sin(x) scaled by 1000
    import math
    length = 100
    signal = [int(1000 * math.sin(2 * math.pi * i / length)) for i in range(length)]
    
    print(f"  Signal length: {length}")
    print(f"  Signal range: [{min(signal)}, {max(signal)}]")
    
    # Decompose with no hints
    hints = []
    layers = py_ternary_decompose(
        signal,
        hints,
        max_layers=5,
        energy_threshold_dominant=0.20,
        energy_threshold_subtle=0.05
    )
    
    print(f"✓ Decomposed into {len(layers)} layers")
    for i, (a, b, h, gx0, gy0, energy) in enumerate(layers):
        print(f"  Layer {i}: triple=({a},{b},{h}), energy={energy:.3f}")
    
except Exception as e:
    print(f"✗ Failed decomposition: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 3: Reconstruction
print("\n=== Test 3: Wave Reconstruction ===")
try:
    if len(layers) > 0:
        # Extract layer parameters (without energy_ratio)
        layer_params = [(a, b, h, gx0, gy0) for a, b, h, gx0, gy0, _ in layers]
        
        # Synthesize composite
        reconstructed = py_ternary_synthesize(layer_params, length)
        
        print(f"✓ Reconstructed signal length: {len(reconstructed)}")
        print(f"  Reconstructed range: [{min(reconstructed)}, {max(reconstructed)}]")
        
        # Compute residual
        residual = [signal[i] - reconstructed[i] for i in range(length)]
        residual_energy = sum(r*r for r in residual)
        signal_energy = sum(s*s for s in signal)
        
        compression_ratio = 1.0 - (residual_energy / signal_energy) if signal_energy > 0 else 0.0
        print(f"  Compression ratio: {compression_ratio:.1%}")
        print(f"  Residual energy: {residual_energy}")
        print(f"  Signal energy: {signal_energy}")
    else:
        print("⚠ No layers to reconstruct")
        
except Exception as e:
    print(f"✗ Failed reconstruction: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 4: Hint buffer acceleration
print("\n=== Test 4: Hint Buffer Acceleration ===")
try:
    # Use first layer's triple as hint
    if len(layers) > 0:
        first_triple = layers[0][:3]  # (a, b, h)
        hints = [first_triple]
        
        # Decompose again with hint
        layers_with_hint = py_ternary_decompose(
            signal,
            hints,
            max_layers=5,
            energy_threshold_dominant=0.20,
            energy_threshold_subtle=0.05
        )
        
        print(f"✓ Decomposed with hint into {len(layers_with_hint)} layers")
        if len(layers_with_hint) > 0:
            first_layer_hint = layers_with_hint[0]
            print(f"  First layer (with hint): triple={first_layer_hint[:3]}, energy={first_layer_hint[5]:.3f}")
            
            # Check if hint was used (should give same or better result)
            if first_layer_hint[:3] == first_triple:
                print("  ✓ Hint was used (same triple)")
            else:
                print(f"  ⚠ Different triple found: {first_layer_hint[:3]}")
    else:
        print("⚠ No hint available (no layers found)")
        
except Exception as e:
    print(f"✗ Failed hint test: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "="*50)
print("✓ All TRFT tests passed!")
print("="*50)
