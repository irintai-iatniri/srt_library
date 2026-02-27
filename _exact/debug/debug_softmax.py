#!/usr/bin/env python3
from syntonic._core import ResonantTensor, SyntonicSoftmaxMode, SyntonicSoftmaxState

# Create test tensor
data = [1.0, 2.0, 3.0, 4.0, 5.0]
tensor = ResonantTensor(data, [1, 5])
print("Original tensor:", tensor.to_floats())

# Move to GPU
gpu_tensor = tensor.to_device(0)
print("GPU tensor device:", gpu_tensor.device_idx())
print("GPU tensor shape:", gpu_tensor.shape)

# Test CPU softmax first
cpu_softmax = SyntonicSoftmaxState(SyntonicSoftmaxMode.Identity, -1)
cpu_result = cpu_softmax.forward(tensor)
print("CPU softmax result:", cpu_result.to_floats())
print("CPU softmax sum:", sum(cpu_result.to_floats()))

# Test GPU softmax
gpu_softmax = SyntonicSoftmaxState(SyntonicSoftmaxMode.Identity, -1)
gpu_result = gpu_softmax.forward(gpu_tensor)
print("GPU softmax result:", gpu_result.to_floats())
print("GPU softmax sum:", sum(gpu_result.to_floats()))
