/**
 * SRT Constants Implementation
 *
 * This file defines constant memory that must be defined only once
 * when linking multiple CUDA object files.
 */

#include <cuda_runtime.h>

#ifndef MAX_MERSENNE
#define MAX_MERSENNE 32
#endif

#define SRT_CONSTANTS_IMPL
#include "srt_constants.cuh"

// Define the constant memory array
__device__ __constant__ bool c_mersenne_mask[MAX_MERSENNE] = {
    false, false, true,  true,  false, true,  false, true,
    false, false, false, false, false, true,  false, false,
    false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false
};
