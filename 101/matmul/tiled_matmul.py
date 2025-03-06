cuda_source = r'''
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
constexpr int TILE_SIZE=16;

__global__ void tiled_matmul_kernel(float* m, float* n, int h, int w, int k) {
    __shared__ float M_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float N_tile[TILE_SIZE][TILE_SIZE];

    // index into tile;
    int tile_r_idx = theadIdx.y;
    int tile_c_idx = threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    // note : cannot just exit if we want to do padding!

    float res = 0.0f;
    for (int k_tile_idx = 0; k_tile_idx < (k + TILE_SIZE - 1) / TILE_SIZE; ++k_tile_size ) {
        // note how threadIdx.x is the fastest moving bit --> coalesced memory access
        M_tile[tile_r_idx][tile_c_idx] = (((r < h) && (k_tile_idx * TILE_SIZE + ic < k)) ? 
                                        M[r*k + k_tile_idx + TILE_SIZE + ic] : 0.f);
        N_tile[tile_r_idx][tile_c_idx] = ((((k_tile_idx * TILE_SIZE + tile_r_idx) < k ) && (c < w)) ? 
                                        N[k_tile_idx * TILE_SIZE + tile_r_idx) * w + c : 0.f)
        
        //M_tile[ir][ic] = M[r * k + K_tileidx * TILE_SIZE + ic];
        //N_tile[ir][ic] = N[(K_tileidx * TILE_SIZE + ir) * w + c];

        __syncthreads();

        for (int idx = 0; idx < TILE_SIZE ; idx ++) {
            res += M_tile[tile_r_idx][idx] * N_tile[idx][tile_c_idx];
        }
        __syncthreads(); // import, make sure above for-loop is finished accross all thread;
    }

    if ((r < h) && (c < w)) {
        out [r * w + c] = res;
    }
}


// cpp wrapper

torch::Tensor tiled_matmul(torch::Tensor &m, torch::Tensor &n) {
    int h = m.size(0);
    int k = m.size(1);
    int w = n.size(1);

    dim3 tile=(16, 16);


}

'''

cpp_source = r'''

'''

import torch
from torch.utils.cpp_extension import load_inline

tiled_matmul_module = load_inline(
    name="tiled_matmul_module",
    functions=['tiled_matmul'],
    cpp_sources=[cpp_source,],
    cuda_sources=[cuda_source,],
    extra_cuda_cflags=['-O2'],
)


m = torch.randn((10, 1024), device='cuda')
n = torch.randn((1024, 5), device='cuda')

with torch.profiler.profile() as prof:
    for i in range(1000):
        output = tiled_matmul_module.tiled_matmul(m, n)
        torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))