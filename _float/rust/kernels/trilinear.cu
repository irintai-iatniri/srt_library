// =============================================================================
// SRT Trilinear Interpolation Kernels: Theory-Aligned Grid Sampling
// =============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "srt_constants.cuh"
using namespace std;

// =============================================================================
// Helper Functions
// =============================================================================

// Sample from 3D grid with boundary handling
__device__ __forceinline__ double sample_grid_3d(
    const double* grid, int x, int y, int z,
    int D, int H, int W, int boundary_mode
) {
    bool in_bounds = (x >= 0 && x < W && y >= 0 && y < H && z >= 0 && z < D);

    if (!in_bounds) {
        switch (boundary_mode) {
            case 0: // CLAMP
                x = max(0, min(x, W-1));
                y = max(0, min(y, H-1));
                z = max(0, min(z, D-1));
                break;
            case 1: // ZERO
                return 0.0;
            case 2: // TOROIDAL
                x = (x % W + W) % W;
                y = (y % H + H) % H;
                z = (z % D + D) % D;
                break;
            case 3: // REFLECT
                if (x < 0) x = -x;
                if (x >= W) x = 2*W - x - 2;
                if (y < 0) y = -y;
                if (y >= H) y = 2*H - y - 2;
                if (z < 0) z = -z;
                if (z >= D) z = 2*D - z - 2;
                break;
        }
    }

    return grid[z * H * W + y * W + x];
}

// Sample from 4D grid (temporal) with boundary handling
__device__ __forceinline__ double sample_grid_4d(
    const double* grid, int t, int x, int y, int z,
    int T, int D, int H, int W, int boundary_mode
) {
    bool in_bounds = (t >= 0 && t < T && x >= 0 && x < W &&
                     y >= 0 && y < H && z >= 0 && z < D);

    if (!in_bounds) {
        switch (boundary_mode) {
            case 0: // CLAMP
                t = max(0, min(t, T-1));
                x = max(0, min(x, W-1));
                y = max(0, min(y, H-1));
                z = max(0, min(z, D-1));
                break;
            case 1: // ZERO
                return 0.0;
            case 2: // TOROIDAL
                t = (t % T + T) % T;
                x = (x % W + W) % W;
                y = (y % H + H) % H;
                z = (z % D + D) % D;
                break;
        }
    }

    return grid[t * D * H * W + z * H * W + y * W + x];
}

// Trilinear interpolation of 8 corners
__device__ __forceinline__ double trilinear_interp(
    double c000, double c001, double c010, double c011,
    double c100, double c101, double c110, double c111,
    double dx, double dy, double dz
) {
    // Interpolate along z
    double c00 = c000 * (1-dz) + c001 * dz;
    double c01 = c010 * (1-dz) + c011 * dz;
    double c10 = c100 * (1-dz) + c101 * dz;
    double c11 = c110 * (1-dz) + c111 * dz;

    // Interpolate along y
    double c0 = c00 * (1-dy) + c01 * dy;
    double c1 = c10 * (1-dy) + c11 * dy;

    // Interpolate along x
    return c0 * (1-dx) + c1 * dx;
}

// =============================================================================
// Standard Trilinear Interpolation
// =============================================================================

extern "C" __global__ void trilinear_f64(
    double* __restrict__ output,
    const double* __restrict__ grid,    // [D, H, W]
    const double* __restrict__ coords,  // [N, 3] (x, y, z)
    int D, int H, int W, int N,
    int boundary_mode  // 0=CLAMP, 1=ZERO, 2=TOROIDAL, 3=REFLECT
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double x = coords[i*3 + 0];
    double y = coords[i*3 + 1];
    double z = coords[i*3 + 2];

    // Convert to grid coordinates (assume coords in [0,W) x [0,H) x [0,D))
    int x0 = floor(x);
    int y0 = floor(y);
    int z0 = floor(z);

    double dx = x - x0;
    double dy = y - y0;
    double dz = z - z0;

    // Sample 8 corners
    double c000 = sample_grid_3d(grid, x0,   y0,   z0,   D, H, W, boundary_mode);
    double c001 = sample_grid_3d(grid, x0,   y0,   z0+1, D, H, W, boundary_mode);
    double c010 = sample_grid_3d(grid, x0,   y0+1, z0,   D, H, W, boundary_mode);
    double c011 = sample_grid_3d(grid, x0,   y0+1, z0+1, D, H, W, boundary_mode);
    double c100 = sample_grid_3d(grid, x0+1, y0,   z0,   D, H, W, boundary_mode);
    double c101 = sample_grid_3d(grid, x0+1, y0,   z0+1, D, H, W, boundary_mode);
    double c110 = sample_grid_3d(grid, x0+1, y0+1, z0,   D, H, W, boundary_mode);
    double c111 = sample_grid_3d(grid, x0+1, y0+1, z0+1, D, H, W, boundary_mode);

    output[i] = trilinear_interp(c000, c001, c010, c011, c100, c101, c110, c111, dx, dy, dz);
}

// =============================================================================
// SRT-Specific Spatial Operations
// =============================================================================

// Toroidal trilinear (W⁴ periodic boundaries)
extern "C" __global__ void trilinear_toroidal_f64(
    double* __restrict__ output,
    const double* __restrict__ grid,
    const double* __restrict__ coords,
    int D, int H, int W, int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double x = coords[i*3 + 0];
    double y = coords[i*3 + 1];
    double z = coords[i*3 + 2];

    // Wrap coordinates for toroidal topology
    x = fmod(x, (double)W);
    if (x < 0) x += W;
    y = fmod(y, (double)H);
    if (y < 0) y += H;
    z = fmod(z, (double)D);
    if (z < 0) z += D;

    int x0 = floor(x);
    int y0 = floor(y);
    int z0 = floor(z);

    double dx = x - x0;
    double dy = y - y0;
    double dz = z - z0;
    double c000 = sample_grid_3d(grid, x0,   y0,   z0,   D, H, W, 2);
    double c001 = sample_grid_3d(grid, x0,   y0,   z0+1, D, H, W, 2);
    double c010 = sample_grid_3d(grid, x0,   y0+1, z0,   D, H, W, 2);
    double c011 = sample_grid_3d(grid, x0,   y0+1, z0+1, D, H, W, 2);
    double c100 = sample_grid_3d(grid, x0+1, y0,   z0,   D, H, W, 2);
    double c101 = sample_grid_3d(grid, x0+1, y0,   z0+1, D, H, W, 2);
    double c110 = sample_grid_3d(grid, x0+1, y0+1, z0,   D, H, W, 2);
    double c111 = sample_grid_3d(grid, x0+1, y0+1, z0+1, D, H, W, 2);

    output[i] = trilinear_interp(c000, c001, c010, c011, c100, c101, c110, c111, dx, dy, dz);
}

// φ-weighted trilinear (golden ratio decay)
extern "C" __global__ void trilinear_phi_weighted_f64(
    double* __restrict__ output,
    const double* __restrict__ grid,
    const double* __restrict__ coords,
    int D, int H, int W, int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double x = coords[i*3 + 0];
    double y = coords[i*3 + 1];
    double z = coords[i*3 + 2];

    int x0 = floor(x);
    int y0 = floor(y);
    int z0 = floor(z);

    double dx = x - x0;
    double dy = y - y0;
    double dz = z - z0;

    // Sample corners with φ-weighted distances
    double c000 = sample_grid_3d(grid, x0,   y0,   z0,   D, H, W, 0);
    double c001 = sample_grid_3d(grid, x0,   y0,   z0+1, D, H, W, 0);
    double c010 = sample_grid_3d(grid, x0,   y0+1, z0,   D, H, W, 0);
    double c011 = sample_grid_3d(grid, x0,   y0+1, z0+1, D, H, W, 0);
    double c100 = sample_grid_3d(grid, x0+1, y0,   z0,   D, H, W, 0);
    double c101 = sample_grid_3d(grid, x0+1, y0,   z0+1, D, H, W, 0);
    double c110 = sample_grid_3d(grid, x0+1, y0+1, z0,   D, H, W, 0);
    double c111 = sample_grid_3d(grid, x0+1, y0+1, z0+1, D, H, W, 0);

    // Apply φ weights based on distance from corner
    double w000 = pow(PHI_INV_F64, sqrt(0.0 + 0.0 + 0.0));  // (0,0,0)
    double w001 = pow(PHI_INV_F64, sqrt(0.0 + 0.0 + 1.0));  // (0,0,1)
    double w010 = pow(PHI_INV_F64, sqrt(0.0 + 1.0 + 0.0));  // (0,1,0)
    double w011 = pow(PHI_INV_F64, sqrt(0.0 + 1.0 + 1.0));  // (0,1,1)
    double w100 = pow(PHI_INV_F64, sqrt(1.0 + 0.0 + 0.0));  // (1,0,0)
    double w101 = pow(PHI_INV_F64, sqrt(1.0 + 0.0 + 1.0));  // (1,0,1)
    double w110 = pow(PHI_INV_F64, sqrt(1.0 + 1.0 + 0.0));  // (1,1,0)
    double w111 = pow(PHI_INV_F64, sqrt(1.0 + 1.0 + 1.0));  // (1,1,1)

    // Normalize weights
    double total_w = w000 + w001 + w010 + w011 + w100 + w101 + w110 + w111;
    w000 /= total_w; w001 /= total_w; w010 /= total_w; w011 /= total_w;
    w100 /= total_w; w101 /= total_w; w110 /= total_w; w111 /= total_w;

    // Weighted interpolation
    double result = c000 * w000 + c001 * w001 + c010 * w010 + c011 * w011 +
                    c100 * w100 + c101 * w101 + c110 * w110 + c111 * w111;

    output[i] = result;
}

// Golden decay trilinear (e^{-r²/φ})
extern "C" __global__ void trilinear_golden_decay_f64(
    double* __restrict__ output,
    const double* __restrict__ grid,
    const double* __restrict__ coords,
    int D, int H, int W, int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double x = coords[i*3 + 0];
    double y = coords[i*3 + 1];
    double z = coords[i*3 + 2];

    int x0 = floor(x);
    int y0 = floor(y);
    int z0 = floor(z);

    // Sample corners
    double c000 = sample_grid_3d(grid, x0,   y0,   z0,   D, H, W, 0);
    double c001 = sample_grid_3d(grid, x0,   y0,   z0+1, D, H, W, 0);
    double c010 = sample_grid_3d(grid, x0,   y0+1, z0,   D, H, W, 0);
    double c011 = sample_grid_3d(grid, x0,   y0+1, z0+1, D, H, W, 0);
    double c100 = sample_grid_3d(grid, x0+1, y0,   z0,   D, H, W, 0);
    double c101 = sample_grid_3d(grid, x0+1, y0,   z0+1, D, H, W, 0);
    double c110 = sample_grid_3d(grid, x0+1, y0+1, z0,   D, H, W, 0);
    double c111 = sample_grid_3d(grid, x0+1, y0+1, z0+1, D, H, W, 0);

    // Golden decay weights from sample point to each corner
    double dx = x - (x0 + 0.5);
    double dy = y - (y0 + 0.5);
    double dz = z - (z0 + 0.5);

    double w000 = exp(-PHI_INV_F64 * (dx*dx + dy*dy + dz*dz));
    double w001 = exp(-PHI_INV_F64 * (dx*dx + dy*dy + (dz-1)*(dz-1)));
    double w010 = exp(-PHI_INV_F64 * (dx*dx + (dy-1)*(dy-1) + dz*dz));
    double w011 = exp(-PHI_INV_F64 * (dx*dx + (dy-1)*(dy-1) + (dz-1)*(dz-1)));
    double w100 = exp(-PHI_INV_F64 * ((dx-1)*(dx-1) + dy*dy + dz*dz));
    double w101 = exp(-PHI_INV_F64 * ((dx-1)*(dx-1) + dy*dy + (dz-1)*(dz-1)));
    double w110 = exp(-PHI_INV_F64 * ((dx-1)*(dx-1) + (dy-1)*(dy-1) + dz*dz));
    double w111 = exp(-PHI_INV_F64 * ((dx-1)*(dx-1) + (dy-1)*(dy-1) + (dz-1)*(dz-1)));

    // Weighted sum
    double result = c000 * w000 + c001 * w001 + c010 * w010 + c011 * w011 +
                    c100 * w100 + c101 * w101 + c110 * w110 + c111 * w111;

    // Normalize by sum of weights
    double total_w = w000 + w001 + w010 + w011 + w100 + w101 + w110 + w111;
    output[i] = result / total_w;
}

// =============================================================================
// Temporal/Causal Operations (4D grids: [T, D, H, W])
// =============================================================================

// Causal trilinear (forward time arrow: past → present)
extern "C" __global__ void trilinear_causal_f64(
    double* __restrict__ output,
    const double* __restrict__ grid,    // [T, D, H, W]
    const double* __restrict__ coords,  // [N, 4] (t, x, y, z)
    int T, int D, int H, int W, int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double t = coords[i*4 + 0];
    double x = coords[i*4 + 1];
    double y = coords[i*4 + 2];
    double z = coords[i*4 + 3];

    int t0 = floor(t);
    double dt = t - t0;

    // Only sample from past: t0 and max(0, t0-1)
    int t_past = max(0, t0 - 1);

    // Sample at t0
    double val_t0 = sample_grid_4d(grid, t0, floor(x), floor(y), floor(z),
                                  T, D, H, W, 0);
    // Sample at t_past
    double val_t_past = sample_grid_4d(grid, t_past, floor(x), floor(y), floor(z),
                                      T, D, H, W, 0);

    // Linear interpolation in time (causal)
    output[i] = val_t_past * (1-dt) + val_t0 * dt;
}

// Retrocausal trilinear (backward: future → present)
extern "C" __global__ void trilinear_retrocausal_f64(
    double* __restrict__ output,
    const double* __restrict__ grid,
    const double* __restrict__ coords,
    int T, int D, int H, int W, int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double t = coords[i*4 + 0];
    double x = coords[i*4 + 1];
    double y = coords[i*4 + 2];
    double z = coords[i*4 + 3];

    int t0 = floor(t);
    double dt = t - t0;

    // Sample from future: t0 and min(T-1, t0+1)
    int t_future = min(T-1, t0 + 1);

    double val_t0 = sample_grid_4d(grid, t0, floor(x), floor(y), floor(z),
                                  T, D, H, W, 0);
    double val_t_future = sample_grid_4d(grid, t_future, floor(x), floor(y), floor(z),
                                        T, D, H, W, 0);

    // Retrocausal weighting with Lucas shadow
    double w_present = pow(PHI_F64, dt);  // φ^dt (golden amplification)
    double w_future = pow(-PHI_INV_F64, 1-dt);  // (1-φ)^{1-dt} (Lucas shadow)

    double total_w = w_present + w_future;
    output[i] = (val_t0 * w_present + val_t_future * w_future) / total_w;
}

// Symmetric trilinear (bidirectional: pre-crystallization)
extern "C" __global__ void trilinear_symmetric_f64(
    double* __restrict__ output,
    const double* __restrict__ grid,
    const double* __restrict__ coords,
    int T, int D, int H, int W, int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double t = coords[i*4 + 0];
    double x = coords[i*4 + 1];
    double y = coords[i*4 + 2];
    double z = coords[i*4 + 3];

    int t0 = floor(t);
    double dt = t - t0;

    // Sample both directions
    int t_past = max(0, t0 - 1);
    int t_future = min(T-1, t0 + 1);

    double val_past = sample_grid_4d(grid, t_past, floor(x), floor(y), floor(z),
                                    T, D, H, W, 0);
    double val_present = sample_grid_4d(grid, t0, floor(x), floor(y), floor(z),
                                       T, D, H, W, 0);
    double val_future = sample_grid_4d(grid, t_future, floor(x), floor(y), floor(z),
                                      T, D, H, W, 0);

    // Symmetric weighting: equal contribution from past/future
    double w_past = (1-dt) * 0.5;
    double w_present = 0.5;
    double w_future = dt * 0.5;

    output[i] = val_past * w_past + val_present * w_present + val_future * w_future;
}

// Acausal trilinear (instantaneous: wave-like)
extern "C" __global__ void trilinear_acausal_f64(
    double* __restrict__ output,
    const double* __restrict__ grid,
    const double* __restrict__ coords,
    int T, int D, int H, int W, int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double t = coords[i*4 + 0];
    double x = coords[i*4 + 1];
    double y = coords[i*4 + 2];
    double z = coords[i*4 + 3];

    int t_center = round(t);
    const int KERNEL_RADIUS = 3;  // ±3 time steps

    double sum = 0.0;
    double total_weight = 0.0;

    // Sum over temporal neighborhood with golden decay
    for (int dt = -KERNEL_RADIUS; dt <= KERNEL_RADIUS; dt++) {
        int t_sample = t_center + dt;
        if (t_sample >= 0 && t_sample < T) {
            double val = sample_grid_4d(grid, t_sample, floor(x), floor(y), floor(z),
                                       T, D, H, W, 0);
            double weight = exp(-abs(dt) * PHI_INV_F64);
            sum += val * weight;
            total_weight += weight;
        }
    }

    output[i] = sum / total_weight;
}

// =============================================================================
// 2D Bilinear Variants
// =============================================================================

extern "C" __global__ void bilinear_f64(
    double* __restrict__ output,
    const double* __restrict__ grid,    // [H, W]
    const double* __restrict__ coords,  // [N, 2] (x, y)
    int H, int W, int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double x = coords[i*2 + 0];
    double y = coords[i*2 + 1];

    int x0 = floor(x);
    int y0 = floor(y);
    double dx = x - x0;
    double dy = y - y0;

    // Sample 4 corners
    double c00 = sample_grid_3d(grid, x0,   y0,   0, 1, H, W, 0);
    double c01 = sample_grid_3d(grid, x0,   y0+1, 0, 1, H, W, 0);
    double c10 = sample_grid_3d(grid, x0+1, y0,   0, 1, H, W, 0);
    double c11 = sample_grid_3d(grid, x0+1, y0+1, 0, 1, H, W, 0);

    // Bilinear interpolation
    double c0 = c00 * (1-dy) + c01 * dy;
    double c1 = c10 * (1-dy) + c11 * dy;
    output[i] = c0 * (1-dx) + c1 * dx;
}

// =============================================================================
// Launcher Functions
// =============================================================================

extern "C" {

// Standard operations
void launch_trilinear_f64(cudaStream_t stream, double* output, const double* grid,
                         const double* coords, int D, int H, int W, int N, int boundary_mode) {
    dim3 block(256);
    dim3 grid_dim((N + 255) / 256);
    trilinear_f64<<<grid_dim, block, 0, stream>>>(output, grid, coords, D, H, W, N, boundary_mode);
}

void launch_bilinear_f64(cudaStream_t stream, double* output, const double* grid,
                        const double* coords, int H, int W, int N) {
    dim3 block(256);
    dim3 grid_dim((N + 255) / 256);
    bilinear_f64<<<grid_dim, block, 0, stream>>>(output, grid, coords, H, W, N);
}

// SRT spatial operations
void launch_trilinear_toroidal_f64(cudaStream_t stream, double* output, const double* grid,
                                  const double* coords, int D, int H, int W, int N) {
    dim3 block(256);
    dim3 grid_dim((N + 255) / 256);
    trilinear_toroidal_f64<<<grid_dim, block, 0, stream>>>(output, grid, coords, D, H, W, N);
}

void launch_trilinear_phi_weighted_f64(cudaStream_t stream, double* output, const double* grid,
                                      const double* coords, int D, int H, int W, int N) {
    dim3 block(256);
    dim3 grid_dim((N + 255) / 256);
    trilinear_phi_weighted_f64<<<grid_dim, block, 0, stream>>>(output, grid, coords, D, H, W, N);
}

void launch_trilinear_golden_decay_f64(cudaStream_t stream, double* output, const double* grid,
                                      const double* coords, int D, int H, int W, int N) {
    dim3 block(256);
    dim3 grid_dim((N + 255) / 256);
    trilinear_golden_decay_f64<<<grid_dim, block, 0, stream>>>(output, grid, coords, D, H, W, N);
}

// SRT temporal operations
void launch_trilinear_causal_f64(cudaStream_t stream, double* output, const double* grid,
                                const double* coords, int T, int D, int H, int W, int N) {
    dim3 block(256);
    dim3 grid_dim((N + 255) / 256);
    trilinear_causal_f64<<<grid_dim, block, 0, stream>>>(output, grid, coords, T, D, H, W, N);
}

void launch_trilinear_retrocausal_f64(cudaStream_t stream, double* output, const double* grid,
                                     const double* coords, int T, int D, int H, int W, int N) {
    dim3 block(256);
    dim3 grid_dim((N + 255) / 256);
    trilinear_retrocausal_f64<<<grid_dim, block, 0, stream>>>(output, grid, coords, T, D, H, W, N);
}

void launch_trilinear_symmetric_f64(cudaStream_t stream, double* output, const double* grid,
                                   const double* coords, int T, int D, int H, int W, int N) {
    dim3 block(256);
    dim3 grid_dim((N + 255) / 256);
    trilinear_symmetric_f64<<<grid_dim, block, 0, stream>>>(output, grid, coords, T, D, H, W, N);
}

void launch_trilinear_acausal_f64(cudaStream_t stream, double* output, const double* grid,
                                 const double* coords, int T, int D, int H, int W, int N) {
    dim3 block(256);
    dim3 grid_dim((N + 255) / 256);
    trilinear_acausal_f64<<<grid_dim, block, 0, stream>>>(output, grid, coords, T, D, H, W, N);
}

}
