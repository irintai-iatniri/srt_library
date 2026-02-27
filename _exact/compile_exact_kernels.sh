#!/bin/bash
# Syntonic CUDA Kernel Compilation Script
# Compiles all CUDA kernels to PTX for multiple GPU architectures
#
# Usage: ./compile_kernels.sh [--kernel=phi_residual] [--arch=sm_80]
#        ./compile_kernels.sh --all
#        ./compile_kernels.sh --clean

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
KERNEL_DIR="rust/kernels"
PTX_DIR="rust/kernels/ptx"
NVCC_FLAGS="-ptx --use_fast_math -O3 -I${KERNEL_DIR}"

# GPU architectures to compile for
ARCHITECTURES=("sm_75" "sm_80" "sm_86" "sm_90")

# All kernels to compile
KERNELS=(
    "phi_residual"
    "golden_batch_norm"
    "syntonic_softmax"
    "golden_ops"
    "golden_gelu"
    "e8_projection"
    "heat_kernel"
    "dhsr"
    "corrections"
    "resonant_d"
    "core_ops"
    "elementwise"
    "matmul"
    "hierarchy"
    "prime_selection"
    "prime_ops"
    "gnosis"
    "conv_ops"
    "winding_ops"
    "reduction"
    "autograd"
    "attractor"
    "attention"
    "dhsr_fp64"
)

# Print colored message
print_msg() {
    local color=$1
    shift
    echo -e "${color}$@${NC}"
}

# Print section header
print_header() {
    echo ""
    print_msg "$BLUE" "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    print_msg "$BLUE" "  $1"
    print_msg "$BLUE" "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
}

# Check if nvcc is available
check_nvcc() {
    if ! command -v nvcc &> /dev/null; then
        print_msg "$RED" "ERROR: nvcc not found!"
        print_msg "$YELLOW" "Please install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads"
        exit 1
    fi

    local nvcc_version=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    print_msg "$GREEN" "✓ Found nvcc version: $nvcc_version"
}

# Create PTX directory
setup_directories() {
    if [ ! -d "$PTX_DIR" ]; then
        print_msg "$YELLOW" "Creating PTX directory: $PTX_DIR"
        mkdir -p "$PTX_DIR"
    fi
    print_msg "$GREEN" "✓ PTX directory ready: $PTX_DIR"
}

# Compile a single kernel for a single architecture
compile_kernel_arch() {
    local kernel=$1
    local arch=$2
    local arch_clean=${arch//_/}
    local cu_file="${KERNEL_DIR}/${kernel}.cu"
    local ptx_file="${PTX_DIR}/${kernel}_${arch_clean}.ptx"

    if [ ! -f "$cu_file" ]; then
        print_msg "$YELLOW" "⚠ Skipping ${kernel}.cu (file not found)"
        return 1
    fi

    print_msg "$BLUE" "  Compiling ${kernel}.cu for ${arch}..."

    if nvcc -arch=${arch} ${NVCC_FLAGS} -o "$ptx_file" "$cu_file" 2>&1; then
        local size=$(du -h "$ptx_file" | cut -f1)
        print_msg "$GREEN" "    ✓ Generated ${ptx_file} (${size})"
        return 0
    else
        print_msg "$RED" "    ✗ Failed to compile ${kernel}.cu for ${arch}"
        return 1
    fi
}

# Compile a single kernel for all architectures
compile_kernel() {
    local kernel=$1
    print_msg "$BLUE" "Compiling ${kernel}.cu..."

    local success=0
    local total=${#ARCHITECTURES[@]}

    for arch in "${ARCHITECTURES[@]}"; do
        if compile_kernel_arch "$kernel" "$arch"; then
            ((success++))
        fi
    done

    if [ $success -eq $total ]; then
        print_msg "$GREEN" "✓ ${kernel}: All $total architectures compiled successfully"
        return 0
    elif [ $success -gt 0 ]; then
        print_msg "$YELLOW" "⚠ ${kernel}: $success/$total architectures compiled"
        return 0
    else
        print_msg "$RED" "✗ ${kernel}: Compilation failed for all architectures"
        return 1
    fi
}

# Compile all kernels
compile_all() {
    print_header "Compiling All CUDA Kernels"

    local total_kernels=0
    local successful_kernels=0

    for kernel in "${KERNELS[@]}"; do
        ((total_kernels++))
        if compile_kernel "$kernel"; then
            ((successful_kernels++))
        fi
        echo ""
    done

    print_header "Compilation Summary"
    print_msg "$BLUE" "Total kernels: $total_kernels"
    print_msg "$GREEN" "Successful: $successful_kernels"

    if [ $successful_kernels -lt $total_kernels ]; then
        local failed=$((total_kernels - successful_kernels))
        print_msg "$RED" "Failed: $failed"
    fi

    # List all generated PTX files
    echo ""
    print_msg "$BLUE" "Generated PTX files:"
    ls -lh "${PTX_DIR}"/*.ptx 2>/dev/null | awk '{print "  "$9" ("$5")"}'

    if [ $successful_kernels -eq $total_kernels ]; then
        echo ""
        print_msg "$GREEN" "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        print_msg "$GREEN" "  ✓ All kernels compiled successfully!"
        print_msg "$GREEN" "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        return 0
    else
        echo ""
        print_msg "$YELLOW" "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        print_msg "$YELLOW" "  ⚠ Some kernels failed to compile"
        print_msg "$YELLOW" "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        return 1
    fi
}

# Clean PTX files
clean() {
    print_header "Cleaning PTX Files"

    if [ -d "$PTX_DIR" ]; then
        local count=$(ls -1 "${PTX_DIR}"/*.ptx 2>/dev/null | wc -l)
        if [ $count -gt 0 ]; then
            print_msg "$YELLOW" "Removing $count PTX files..."
            rm -f "${PTX_DIR}"/*.ptx
            print_msg "$GREEN" "✓ PTX files cleaned"
        else
            print_msg "$BLUE" "No PTX files to clean"
        fi
    else
        print_msg "$BLUE" "PTX directory doesn't exist"
    fi
}

# Show help
show_help() {
    cat << EOF
Syntonic CUDA Kernel Compilation Script

USAGE:
    ./compile_kernels.sh [OPTIONS]

OPTIONS:
    --all                   Compile all kernels for all architectures (default)
    --kernel=NAME           Compile specific kernel (e.g., --kernel=phi_residual)
    --arch=ARCH             Compile for specific architecture (e.g., --arch=sm_80)
    --clean                 Remove all generated PTX files
    --list                  List available kernels
    --help                  Show this help message

EXAMPLES:
    ./compile_kernels.sh                          # Compile all kernels
    ./compile_kernels.sh --all                    # Same as above
    ./compile_kernels.sh --kernel=phi_residual    # Compile only phi_residual
    ./compile_kernels.sh --arch=sm_80             # Compile all for SM80 only
    ./compile_kernels.sh --kernel=phi_residual --arch=sm_80  # Specific kernel & arch
    ./compile_kernels.sh --clean                  # Clean all PTX files

AVAILABLE KERNELS:
$(for k in "${KERNELS[@]}"; do echo "    - $k"; done)

SUPPORTED ARCHITECTURES:
    - sm_75  (Turing:  RTX 20 series, T4)
    - sm_80  (Ampere:  A100, RTX 30 series)
    - sm_86  (Ampere:  RTX 30 series)
    - sm_90  (Hopper:  H100)

EOF
}

# List available kernels
list_kernels() {
    print_header "Available Kernels"

    for kernel in "${KERNELS[@]}"; do
        local cu_file="${KERNEL_DIR}/${kernel}.cu"
        if [ -f "$cu_file" ]; then
            local size=$(du -h "$cu_file" | cut -f1)
            print_msg "$GREEN" "  ✓ ${kernel}.cu (${size})"
        else
            print_msg "$RED" "  ✗ ${kernel}.cu (missing)"
        fi
    done

    echo ""
    print_msg "$BLUE" "To compile a specific kernel:"
    print_msg "$BLUE" "  ./compile_kernels.sh --kernel=phi_residual"
}

# Main function
main() {
    local specific_kernel=""
    local specific_arch=""
    local do_clean=false
    local do_list=false

    # Parse arguments
    for arg in "$@"; do
        case $arg in
            --help|-h)
                show_help
                exit 0
                ;;
            --clean)
                do_clean=true
                ;;
            --list)
                do_list=true
                ;;
            --all)
                # Default behavior
                ;;
            --kernel=*)
                specific_kernel="${arg#*=}"
                ;;
            --arch=*)
                specific_arch="${arg#*=}"
                ;;
            *)
                print_msg "$RED" "Unknown option: $arg"
                print_msg "$YELLOW" "Use --help for usage information"
                exit 1
                ;;
        esac
    done

    # Handle clean
    if [ "$do_clean" = true ]; then
        clean
        exit 0
    fi

    # Handle list
    if [ "$do_list" = true ]; then
        list_kernels
        exit 0
    fi

    # Print banner
    print_header "Syntonic CUDA Kernel Compiler"

    # Check prerequisites
    check_nvcc
    setup_directories

    echo ""

    # Compile based on arguments
    if [ -n "$specific_kernel" ] && [ -n "$specific_arch" ]; then
        # Specific kernel and architecture
        print_msg "$BLUE" "Compiling ${specific_kernel}.cu for ${specific_arch}..."
        if compile_kernel_arch "$specific_kernel" "$specific_arch"; then
            print_msg "$GREEN" "✓ Compilation successful"
            exit 0
        else
            print_msg "$RED" "✗ Compilation failed"
            exit 1
        fi
    elif [ -n "$specific_kernel" ]; then
        # Specific kernel, all architectures
        if compile_kernel "$specific_kernel"; then
            exit 0
        else
            exit 1
        fi
    elif [ -n "$specific_arch" ]; then
        # All kernels, specific architecture
        print_header "Compiling All Kernels for ${specific_arch}"
        local success=0
        local total=${#KERNELS[@]}

        for kernel in "${KERNELS[@]}"; do
            if compile_kernel_arch "$kernel" "$specific_arch"; then
                ((success++))
            fi
        done

        echo ""
        print_msg "$BLUE" "Compiled $success/$total kernels for ${specific_arch}"

        if [ $success -eq $total ]; then
            print_msg "$GREEN" "✓ All kernels compiled successfully"
            exit 0
        else
            print_msg "$YELLOW" "⚠ Some kernels failed to compile"
            exit 1
        fi
    else
        # All kernels, all architectures
        if compile_all; then
            exit 0
        else
            exit 1
        fi
    fi
}

# Run main function
main "$@"
