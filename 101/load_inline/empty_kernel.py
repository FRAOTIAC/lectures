
cuda_source = r'''
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
int  cdiv(int n, int divisor) {
  return (n + divisor - 1) / divisor;
}

// empty kernel
__global__ void empty_cuda_kernel(float * input, float * output, int n) {

}

// cpp wrapper
void empty_kernel(torch::Tensor &input, torch::Tensor &output) {
  auto n = input.numel();
  auto threads_per_block = 256;
  auto blocks = cdiv(n, threads_per_block);
  empty_cuda_kernel<<<blocks, threads_per_block>>>(input.data_ptr<float>(),output.data_ptr<float>(), n);
}
'''

cpp_source=r"void empty_kernel(torch::Tensor &input, torch::Tensor &output);"

import torch
from torch.utils.cpp_extension import load_inline

empty_kernel = load_inline(
    name = 'empty_kernel',
    functions = ['empty_kernel'],
    cpp_sources = [cpp_source,],
    cuda_sources = [cuda_source,],
    extra_cflags=['-O2'],
)

input = torch.randn((3, 1024, 1024), device='cuda')
output = torch.randn((3, 1024, 1024), device='cuda')

empty_kernel.empty_kernel(input, output)

with torch.profiler.profile() as prof:
  for i in range(1000):
    empty_kernel.empty_kernel(input, output)
    torch.cuda.synchronize()

print(prof.key_averages().table())

