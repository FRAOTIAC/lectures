cuda_source=r'''
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

int cdiv(int n, int divisor) {
    return (n + divisor - 1) / divisor;
}

__global__ void simple_matmul_cuda(float* m, float* n, float* output, int h, int w, int k) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    // h and w are result matrix dimensions;
    if (r >= h || c >= w) { return;}
    int element_idx = r * w + c;
    float sum = 0;

    // this cuda thread loop through k dimension, output one element at [r][c] in the output matrix
    // m[r * k + k_idx] 表示矩阵 A 的第 r 行第 k_idx 列的元素
    // n[k_idx * w + c] 表示矩阵 B 的第 k_idx 行第 c 列的元素
    for (int k_idx = 0; k_idx < k; ++k_idx) {
        sum += m[r * k + k_idx] * n[k_idx * w + c];
    }
    output[element_idx] = sum;
}

torch::Tensor simple_matmul(torch::Tensor &m, torch::Tensor &n) {
    CHECK_INPUT(m);
    CHECK_INPUT(n);

    # m：指向矩阵 A 的指针，假设 A 的尺寸为 h × k，以行优先（row-major）存储。
    # n：指向矩阵 B 的指针，假设 B 的尺寸为 k × w，以行优先存储。
    # output：指向结果矩阵 C 的指针，C 的尺寸为 h × w。
    # h, w, k：分别表示结果矩阵 C 的行数 h、列数 w，以及内积计算的公共维度 k。

    int h  = m.size(0);
    int k  = m.size(1);
    int w = n.size(1);
    
    TORCH_CHECK(k==n.size(0), "matrix dimensions must match");

    auto output = torch::empty({h, w}, m.options());

    dim3 tiles_per_block(16, 16);
    dim3 blocks(cdiv(w, tiles_per_block.x), cdiv(h, tiles_per_block.y));

    simple_matmul_cuda<<<blocks, tiles_per_block>>>(
                            m.data_ptr<float>(), 
                            n.data_ptr<float>(), 
                            output.data_ptr<float>(), 
                            h, w, k);

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}

'''

cpp_source=r"torch::Tensor simple_matmul(torch::Tensor &m, torch::Tensor &n);"


import torch
from torch.utils.cpp_extension import load_inline

simple_matmul_module = load_inline(
    name="simple_matmul_module",
    functions=['simple_matmul'],
    cpp_sources=[cpp_source,],
    cuda_sources=[cuda_source,],
    extra_cuda_cflags=['-O2'],
)


m = torch.randn((10, 1024), device='cuda')
n = torch.randn((1024, 5), device='cuda')


output = simple_matmul_module.simple_matmul(m, n)
torch.cuda.synchronize()

print(torch.allclose(torch.matmul(m, n), output))





