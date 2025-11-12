import numpy as np
import cuda_bindings as cuda

def float32_to_bfloat16(f32_array):
    f32_as_uint32 = f32_array.view(np.uint32)
    bfloat16_uint16 = (f32_as_uint32 >> 16).astype(np.uint16)
    return bfloat16_uint16

def bfloat16_to_float32(bf16_uint16):
    f32_as_uint32 = (bf16_uint16.astype(np.uint32) << 16)
    return f32_as_uint32.view(np.float32)

# 假设我们把 compute_bf16gemm_async_copy.cu 作为字符串嵌入
cuda_source = r"""
#include <cuda_bf16.h>
#include <mma.h>
using namespace nvcuda;

#define M 16
#define N 16
#define K 16

extern "C" __global__
void simple_wmma_bf16gemm(const __nv_bfloat16 *A,
                          const __nv_bfloat16 *B,
                          float *C,
                          int M_GLOBAL,
                          int N_GLOBAL,
                          int K_GLOBAL,
                          float alpha,
                          float beta)
{
#if __CUDA_ARCH__ >= 800
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M_GLOBAL && col < N_GLOBAL) {
        float value = 0.0f;
        for (int k = 0; k < K_GLOBAL; ++k) {
            value += __bfloat162float(A[row * K_GLOBAL + k]) *
                     __bfloat162float(B[k * N_GLOBAL + col]);
        }
        C[row * N_GLOBAL + col] = alpha * value + beta * C[row * N_GLOBAL + col];
    }
#endif
}
"""

# -------------------------------------------------------------------------
# 初始化 CUDA 上下文
# -------------------------------------------------------------------------
dev = cuda.Device(0)
ctx = dev.make_context()

# 编译 CUDA 代码为 PTX
nvrtc = cuda.nvrtc.Program(cuda_source, "bf16gemm.cu", [])
ptx = nvrtc.compile()

# 加载 PTX 模块
module = cuda.Module.load_data(ptx)
kernel = module.get_function("simple_wmma_bf16gemm")

# -------------------------------------------------------------------------
# 创建矩阵并分配显存
# -------------------------------------------------------------------------
M_GLOBAL, N_GLOBAL, K_GLOBAL = 128, 128, 128
alpha, beta = 1.1, 1.2

A_h = np.random.randint(0, 3, (M_GLOBAL, K_GLOBAL)).astype(np.float32)
B_h = np.random.randint(0, 3, (K_GLOBAL, N_GLOBAL)).astype(np.float32)
C_h = np.random.randint(0, 3, (M_GLOBAL, N_GLOBAL)).astype(np.float32)

# 转换为 bfloat16
A_bf16 = A_h.astype(np.bfloat16)
B_bf16 = B_h.astype(np.bfloat16)

# 分配 GPU 内存
A_d = cuda.mem_alloc(A_bf16.nbytes)
B_d = cuda.mem_alloc(B_bf16.nbytes)
C_d = cuda.mem_alloc(C_h.nbytes)

cuda.memcpy_htod(A_d, A_bf16)
cuda.memcpy_htod(B_d, B_bf16)
cuda.memcpy_htod(C_d, C_h)

# -------------------------------------------------------------------------
# 启动 CUDA 内核
# -------------------------------------------------------------------------
block = (16, 16, 1)
grid = ((N_GLOBAL + 15)//16, (M_GLOBAL + 15)//16, 1)

kernel.launch(grid, block, (
    A_d, B_d, C_d,
    np.int32(M_GLOBAL), np.int32(N_GLOBAL), np.int32(K_GLOBAL),
    np.float32(alpha), np.float32(beta)
))

# -------------------------------------------------------------------------
# 取回结果
# -------------------------------------------------------------------------
C_result = np.empty_like(C_h)
cuda.memcpy_dtoh(C_result, C_d)

print("Result matrix sample:")
print(C_result[:4, :4])

# -------------------------------------------------------------------------
# 清理资源
# -------------------------------------------------------------------------
A_d.free()
B_d.free()
C_d.free()
ctx.pop()
