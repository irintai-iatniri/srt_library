#!/usr/bin/env python3
"""Test SNA import and basic functionality"""

# Test 1: Import from srt_library._core
print("Test 1: Importing from srt_library._core")
try:
    from srt_library._core import ResonantOscillator, DiscreteHilbertKernel
    print("✓ Successfully imported from srt_library._core")
except Exception as e:
    print(f"✗ Failed to import from srt_library._core: {e}")

# Test 2: Check sys.modules for sna
print("\nTest 2: Checking sys.modules for sna submodule")
import sys
if "srt_library.sna" in sys.modules:
    print(f"✓ Found srt_library.sna in sys.modules")
    sna = sys.modules["srt_library.sna"]
    print(f"  Contents: {dir(sna)}")
else:
    print("✗ srt_library.sna not in sys.modules")

# Test 3: Try importing from syntonic.sna
print("\nTest 3: Importing from syntonic.sna")
try:
    from syntonic_applications.sna import ResonantOscillator, KERNEL_SCALE
    print(f"✓ Successfully imported from syntonic.sna")
    print(f"  KERNEL_SCALE = {KERNEL_SCALE}")
except Exception as e:
    print(f"✗ Failed to import from syntonic.sna: {e}")

# Test 4: Create a neuron
print("\nTest 4: Creating ResonantOscillator instance")
try:
    from srt_library._core import ResonantOscillator
    neuron = ResonantOscillator(1, 31, 500)
    print(f"✓ Created neuron: {neuron}")
except Exception as e:
    print(f"✗ Failed to create neuron: {e}")

# Test 5: Sine wave test
print("\nTest 5: IT IS ALIVE test (sine wave response)")
try:
    import math
    from srt_library._core import ResonantOscillator
    
    neuron = ResonantOscillator(1, 31, 500)
    signal = [int(1000 * math.sin(2 * math.pi * t / 20)) for t in range(100)]
    spikes = neuron.process_batch([signal])
    
    response_str = "".join("▲" if s > 0 else "▼" if s < 0 else "·" for s in spikes)
    has_activity = any(s != 0 for s in spikes)
    
    print(f"✓ Processed {len(spikes)} samples")
    print(f"  Response: {response_str[:80]}...")
    print(f"  IT IS ALIVE: {has_activity}")
except Exception as e:
    print(f"✗ Failed sine wave test: {e}")
    import traceback
    traceback.print_exc()
