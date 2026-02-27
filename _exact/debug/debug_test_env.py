from syntonic_applications.core import ResonantTensor


def has_cuda_debug():
    print("Checking CUDA availability...")
    try:
        t = ResonantTensor([1.0], [1])
        print("Tensor created")
        t_gpu = t.to_device(0)
        print(f"Moved to GPU: {t_gpu.device_idx()}")
        return True
    except Exception as e:
        print(f"CUDA check failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print(f"CUDA Available: {has_cuda_debug()}")
