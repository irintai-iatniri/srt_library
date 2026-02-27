#!/usr/bin/env python3
"""
Syntonic CUDA Kernel Compilation Script (Python Version)

Compiles all CUDA kernels to PTX for multiple GPU architectures.
Cross-platform alternative to bash script.

Usage:
    python compile_kernels.py                          # Compile all
    python compile_kernels.py --kernel=phi_residual    # Specific kernel
    python compile_kernels.py --arch=sm_80             # Specific architecture
    python compile_kernels.py --clean                  # Clean PTX files
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

# Configuration
KERNEL_DIR = (Path(__file__).parent / "rust/kernels").resolve()
PTX_DIR = KERNEL_DIR / "ptx"
NVCC_PATHS = [
    "/usr/local/cuda/bin/nvcc",
    "/usr/local/cuda-13.1/bin/nvcc",
    "/usr/local/cuda-13/bin/nvcc",
    "nvcc",
]
NVCC_FLAGS = ["-ptx", "--use_fast_math", "-O3", f"-I{KERNEL_DIR}"]

# GPU architectures (using compute capability numbers)
ARCHITECTURES = ["75", "80", "86", "90"]

# All kernels
KERNELS = [
    "phi_residual",
    "golden_batch_norm",
    "syntonic_softmax",
    "golden_ops",
    "e8_projection",
    "heat_kernel",
    "dhsr",
    "corrections",
    "resonant_d",
    "core_ops",
    "elementwise",
    "matmul",
    "conv_ops",
    "winding_ops",
]


# ANSI color codes
class Colors:
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    BLUE = "\033[0;34m"
    BOLD = "\033[1m"
    NC = "\033[0m"  # No Color

    @staticmethod
    def enabled():
        """Check if colors should be enabled (terminal support)"""
        return sys.stdout.isatty()


def print_colored(color: str, message: str):
    """Print colored message if terminal supports it"""
    if Colors.enabled():
        print(f"{color}{message}{Colors.NC}")
    else:
        print(message)


def print_header(title: str):
    """Print section header"""
    print()
    print_colored(Colors.BLUE, "━" * 70)
    print_colored(Colors.BLUE, f"  {title}")
    print_colored(Colors.BLUE, "━" * 70)
    print()


def check_nvcc() -> Optional[str]:
    """Check if nvcc is available and get version. Returns path to working nvcc or None."""
    for nvcc_path in NVCC_PATHS:
        try:
            result = subprocess.run(
                [nvcc_path, "--version"], capture_output=True, text=True, check=True
            )
            # Extract version from output
            for line in result.stdout.split("\n"):
                if "release" in line:
                    version = line.split("release")[1].split(",")[0].strip()
                    print_colored(
                        Colors.GREEN, f"✓ Found nvcc at {nvcc_path}: version {version}"
                    )
                    return nvcc_path
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue

    print_colored(Colors.RED, "ERROR: nvcc not found!")
    print_colored(Colors.YELLOW, "Tried paths:")
    for path in NVCC_PATHS:
        print_colored(Colors.YELLOW, f"  - {path}")
    print_colored(
        Colors.YELLOW,
        "Please install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads",
    )
    return None


def setup_directories():
    """Create PTX directory if it doesn't exist"""
    if not PTX_DIR.exists():
        print_colored(Colors.YELLOW, f"Creating PTX directory: {PTX_DIR}")
        PTX_DIR.mkdir(parents=True, exist_ok=True)
    print_colored(Colors.GREEN, f"✓ PTX directory ready: {PTX_DIR}")


def compile_kernel_arch(kernel: str, arch: str, nvcc_path: str) -> bool:
    """Compile a single kernel for a single architecture"""
    cu_file = KERNEL_DIR / f"{kernel}.cu"
    ptx_file = PTX_DIR / f"{kernel}_sm{arch}.ptx"

    if not cu_file.exists():
        print_colored(Colors.YELLOW, f"⚠ Skipping {kernel}.cu (file not found)")
        return False

    print_colored(Colors.BLUE, f"  Compiling {kernel}.cu for sm_{arch}...")

    try:
        cmd = (
            [nvcc_path, f"-arch=compute_{arch}"]
            + NVCC_FLAGS
            + ["-o", str(ptx_file), str(cu_file)]
        )
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Get file size
        size = ptx_file.stat().st_size
        size_str = format_size(size)
        print_colored(Colors.GREEN, f"    ✓ Generated {ptx_file.name} ({size_str})")
        return True

    except subprocess.CalledProcessError as e:
        print_colored(Colors.RED, f"    ✗ Failed to compile {kernel}.cu for {arch}")
        if e.stderr:
            print_colored(Colors.RED, f"    Error: {e.stderr[:200]}")
        return False


def format_size(size_bytes: int) -> str:
    """Format file size in human-readable format"""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f}TB"


def compile_kernel(kernel: str, nvcc_path: str) -> Tuple[int, int]:
    """Compile a single kernel for all architectures

    Returns:
        (successful_count, total_count)
    """
    print_colored(Colors.BLUE, f"Compiling {kernel}.cu...")

    success = 0
    total = len(ARCHITECTURES)

    for arch in ARCHITECTURES:
        if compile_kernel_arch(kernel, arch, nvcc_path):
            success += 1

    if success == total:
        print_colored(
            Colors.GREEN, f"✓ {kernel}: All {total} architectures compiled successfully"
        )
    elif success > 0:
        print_colored(
            Colors.YELLOW, f"⚠ {kernel}: {success}/{total} architectures compiled"
        )
    else:
        print_colored(
            Colors.RED, f"✗ {kernel}: Compilation failed for all architectures"
        )

    return success, total


def compile_all(nvcc_path: str) -> bool:
    """Compile all kernels for all architectures"""
    print_header("Compiling All CUDA Kernels")

    total_kernels = 0
    successful_kernels = 0

    for kernel in KERNELS:
        total_kernels += 1
        success, _ = compile_kernel(kernel, nvcc_path)
        if success > 0:
            successful_kernels += 1
        print()

    print_header("Compilation Summary")
    print_colored(Colors.BLUE, f"Total kernels: {total_kernels}")
    print_colored(Colors.GREEN, f"Successful: {successful_kernels}")

    if successful_kernels < total_kernels:
        failed = total_kernels - successful_kernels
        print_colored(Colors.RED, f"Failed: {failed}")

    # List all generated PTX files
    print()
    print_colored(Colors.BLUE, "Generated PTX files:")
    ptx_files = sorted(PTX_DIR.glob("*.ptx"))

    if ptx_files:
        for ptx_file in ptx_files:
            size = format_size(ptx_file.stat().st_size)
            print(f"  {ptx_file.name} ({size})")
    else:
        print_colored(Colors.YELLOW, "  No PTX files generated")

    print()
    if successful_kernels == total_kernels:
        print_colored(Colors.GREEN, "━" * 70)
        print_colored(Colors.GREEN, "  ✓ All kernels compiled successfully!")
        print_colored(Colors.GREEN, "━" * 70)
        print()
        return True
    else:
        print_colored(Colors.YELLOW, "━" * 70)
        print_colored(Colors.YELLOW, "  ⚠ Some kernels failed to compile")
        print_colored(Colors.YELLOW, "━" * 70)
        print()
        return False


def clean():
    """Remove all generated PTX files"""
    print_header("Cleaning PTX Files")

    if PTX_DIR.exists():
        ptx_files = list(PTX_DIR.glob("*.ptx"))
        if ptx_files:
            print_colored(Colors.YELLOW, f"Removing {len(ptx_files)} PTX files...")
            for ptx_file in ptx_files:
                ptx_file.unlink()
            print_colored(Colors.GREEN, "✓ PTX files cleaned")
        else:
            print_colored(Colors.BLUE, "No PTX files to clean")
    else:
        print_colored(Colors.BLUE, "PTX directory doesn't exist")


def list_kernels():
    """List available kernels"""
    print_header("Available Kernels")

    for kernel in KERNELS:
        cu_file = KERNEL_DIR / f"{kernel}.cu"
        if cu_file.exists():
            size = format_size(cu_file.stat().st_size)
            print_colored(Colors.GREEN, f"  ✓ {kernel}.cu ({size})")
        else:
            print_colored(Colors.RED, f"  ✗ {kernel}.cu (missing)")

    print()
    print_colored(Colors.BLUE, "To compile a specific kernel:")
    print_colored(Colors.BLUE, "  python compile_kernels.py --kernel=phi_residual")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Syntonic CUDA Kernel Compilation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compile_kernels.py                          # Compile all kernels
  python compile_kernels.py --kernel=phi_residual    # Compile specific kernel
  python compile_kernels.py --arch=80                # Compile for specific arch (sm_80)
  python compile_kernels.py --clean                  # Clean PTX files
  python compile_kernels.py --list                   # List available kernels

Supported Architectures:
  75  (Turing:  RTX 20 series, T4)
  80  (Ampere:  A100, RTX 30 series)
  86  (Ampere:  RTX 30 series)
  90  (Hopper:  H100)
        """,
    )

    parser.add_argument(
        "--kernel", type=str, help="Compile specific kernel (e.g., phi_residual)"
    )
    parser.add_argument(
        "--arch",
        type=str,
        choices=ARCHITECTURES,
        help="Compile for specific architecture (e.g., 80)",
    )
    parser.add_argument(
        "--clean", action="store_true", help="Remove all generated PTX files"
    )
    parser.add_argument("--list", action="store_true", help="List available kernels")
    parser.add_argument(
        "--all", action="store_true", help="Compile all kernels (default behavior)"
    )

    args = parser.parse_args()

    # Handle clean
    if args.clean:
        clean()
        return 0

    # Handle list
    if args.list:
        list_kernels()
        return 0

    # Print banner
    print_header("Syntonic CUDA Kernel Compiler")

    # Check prerequisites
    nvcc_path = check_nvcc()
    if nvcc_path is None:
        return 1

    setup_directories()
    print()

    # Compile based on arguments
    try:
        if args.kernel and args.arch:
            # Specific kernel and architecture
            print_colored(Colors.BLUE, f"Compiling {args.kernel}.cu for {args.arch}...")
            if compile_kernel_arch(args.kernel, args.arch, nvcc_path):
                print_colored(Colors.GREEN, "✓ Compilation successful")
                return 0
            else:
                print_colored(Colors.RED, "✗ Compilation failed")
                return 1

        elif args.kernel:
            # Specific kernel, all architectures
            success, total = compile_kernel(args.kernel, nvcc_path)
            return 0 if success > 0 else 1

        elif args.arch:
            # All kernels, specific architecture
            print_header(f"Compiling All Kernels for {args.arch}")
            success = 0
            total = len(KERNELS)

            for kernel in KERNELS:
                if compile_kernel_arch(kernel, args.arch, nvcc_path):
                    success += 1

            print()
            print_colored(
                Colors.BLUE, f"Compiled {success}/{total} kernels for {args.arch}"
            )

            if success == total:
                print_colored(Colors.GREEN, "✓ All kernels compiled successfully")
                return 0
            else:
                print_colored(Colors.YELLOW, "⚠ Some kernels failed to compile")
                return 1

        else:
            # All kernels, all architectures (default)
            return 0 if compile_all(nvcc_path) else 1

    except KeyboardInterrupt:
        print()
        print_colored(Colors.YELLOW, "⚠ Compilation interrupted by user")
        return 130


if __name__ == "__main__":
    sys.exit(main())
