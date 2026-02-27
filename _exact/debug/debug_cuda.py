
from syntonic._core import ResonantTensor, SyntonicSoftmaxMode, SyntonicSoftmaxState

# Test data
test_data = [1.0, 2.0, 3.0, 4.0, 5.0]
print(f"Test data: {test_data}")

# CPU test
t_cpu = ResonantTensor(test_data, [5])
print("Tensor created on CPU")
print(f"CPU tensor phase: {t_cpu.phase}")
print(f"CPU tensor device: {t_cpu.device_idx()}")

softmax_cpu = SyntonicSoftmaxState(SyntonicSoftmaxMode.Identity, -1)
result_cpu = softmax_cpu.forward(t_cpu)
cpu_output = result_cpu.to_floats()
print(f"CPU softmax result: {cpu_output}")
print(f"CPU result phase: {result_cpu.phase}")
print(f"CPU result device: {result_cpu.device_idx()}")

# GPU test
t_gpu = t_cpu.to_device(0)
print("Tensor moved to GPU")
print(f"GPU tensor phase: {t_gpu.phase}")
print(f"GPU tensor device: {t_gpu.device_idx()}")

softmax_gpu = SyntonicSoftmaxState(SyntonicSoftmaxMode.Identity, -1)
result_gpu = softmax_gpu.forward(t_gpu)
print(f"GPU softmax result (before crystallize): {result_gpu.to_floats()}")

# Try to crystallize the result
try:
    flux_values = result_gpu.wake_flux()
    result_gpu.crystallize(flux_values, 100)
    gpu_output = result_gpu.to_floats()
    print(f"GPU softmax result (after crystallize): {gpu_output}")
except Exception as e:
    print(f"Crystallize failed: {e}")
    gpu_output = result_gpu.to_floats()
    print(f"GPU softmax result (no crystallize): {gpu_output}")

import traceback

traceback.print_exc()
