import sys
import numpy as np
import math

from cuda.bindings import driver
from common.helper_cuda import checkCudaErrors, findCudaDevice
from common.helper_string import sdkFindFilePath


# =========================
# BF16 <-> FP32 转换工具
# =========================
def float32_to_bfloat16(f32_array: np.ndarray) -> np.ndarray:
    """
    float32 -> bfloat16 (uint16 表示)
    """
    assert f32_array.dtype == np.float32
    f32_as_u32 = f32_array.view(np.uint32)
    bf16_u16 = (f32_as_u32 >> 16).astype(np.uint16)
    return bf16_u16


def bfloat16_to_float32(bf16_u16: np.ndarray) -> np.ndarray:
    """
    bfloat16 (uint16) -> float32
    """
    assert bf16_u16.dtype == np.uint16
    f32_as_u32 = bf16_u16.astype(np.uint32) << 16
    return f32_as_u32.view(np.float32)


# =========================
# 主程序
# =========================
def main():
    # ------------------------------------------------------------
    # 1. 初始化 CUDA
    # ------------------------------------------------------------
    dev = findCudaDevice()

    # ------------------------------------------------------------
    # 2. 加载 PTX 并获取 kernel
    # ------------------------------------------------------------
    ptx_path = sdkFindFilePath("kernels.ptx", sys.argv[0])
    if ptx_path is None:
        print("Error: kernels.ptx not found!")
        sys.exit(1)
        
    with open(ptx_path, "rb") as f:
        ptx_data = f.read()

    module = checkCudaErrors(driver.cuModuleLoadData(ptx_data))

    # 使用 simple_wmma_bf16gemm
    try:
        kernel = checkCudaErrors(
            driver.cuModuleGetFunction(module, b"simple_wmma_bf16gemm")
        )
    except Exception as e:
        print(f"Error: Could not find kernel 'simple_wmma_bf16gemm' in PTX file: {e}")
        print("Available functions in the module:")
        # This is a bit hacky but helps debug missing kernel issues
        try:
            # Try to list all functions (not directly supported in CUDA driver API)
            print(" - simple_wmma_bf16gemm (not found)")
            print("Note: You may need to recompile the PTX with the correct kernel name")
        except:
            pass
        sys.exit(1)

    # ------------------------------------------------------------
    # 3. 主机侧数据准备（真实 BF16 流程）
    # ------------------------------------------------------------
    M_GLOBAL, N_GLOBAL, K_GLOBAL = 1024, 1024, 1024
    alpha, beta = 1.1, 1.2

    # 使用 float32 生成
    A_f32 = np.random.rand(M_GLOBAL, K_GLOBAL).astype(np.float32)
    B_f32 = np.random.rand(K_GLOBAL, N_GLOBAL).astype(np.float32)

    # 转 BF16（uint16）
    A_h = float32_to_bfloat16(A_f32)
    B_h = float32_to_bfloat16(B_f32)

    # C / D 使用 float32
    C_h = np.random.rand(M_GLOBAL, N_GLOBAL).astype(np.float32)
    D_h = np.zeros((M_GLOBAL, N_GLOBAL), dtype=np.float32)

    # （可选）验证 BF16 精度
    A_recon = bfloat16_to_float32(A_h)
    print("BF16 A max abs error:", np.max(np.abs(A_f32 - A_recon)))

    # ------------------------------------------------------------
    # 4. 分配设备内存
    # ------------------------------------------------------------
    dA = checkCudaErrors(driver.cuMemAlloc(A_h.nbytes))
    dB = checkCudaErrors(driver.cuMemAlloc(B_h.nbytes))
    dC = checkCudaErrors(driver.cuMemAlloc(C_h.nbytes))
    dD = checkCudaErrors(driver.cuMemAlloc(D_h.nbytes))

    stream = checkCudaErrors(driver.cuStreamCreate(0))

    # ------------------------------------------------------------
    # 5. 拷贝数据到设备
    # ------------------------------------------------------------
    checkCudaErrors(driver.cuMemcpyHtoDAsync(dA, A_h.ctypes.data, A_h.nbytes, stream))
    checkCudaErrors(driver.cuMemcpyHtoDAsync(dB, B_h.ctypes.data, B_h.nbytes, stream))
    checkCudaErrors(driver.cuMemcpyHtoDAsync(dC, C_h.ctypes.data, C_h.nbytes, stream))
    checkCudaErrors(driver.cuMemsetD32Async(dD, 0, D_h.nbytes // 4, stream))

    # ------------------------------------------------------------
    # 6. kernel 参数（严格匹配 simple_wmma_bf16gemm）
    #
    # __global__ void simple_wmma_bf16gemm(
    #   __nv_bfloat16 *a,
    #   __nv_bfloat16 *b,
    #   float *c,
    #   float *d,
    #   int m_ld,
    #   int n_ld,
    #   int k_ld,
    #   float alpha,
    #   float beta)
    # ------------------------------------------------------------
    args = [
        np.array([int(dA)], dtype=np.uint64),
        np.array([int(dB)], dtype=np.uint64),
        np.array([int(dC)], dtype=np.uint64),
        np.array([int(dD)], dtype=np.uint64),
        np.array([M_GLOBAL], dtype=np.int32),
        np.array([N_GLOBAL], dtype=np.int32),
        np.array([K_GLOBAL], dtype=np.int32),
        np.array([alpha], dtype=np.float32),
        np.array([beta], dtype=np.float32),
    ]

    kernel_args = np.array([arg.ctypes.data for arg in args], dtype=np.uint64)

    # ------------------------------------------------------------
    # 7. grid / block 设置（严格匹配官方示例）
    # ------------------------------------------------------------
    block_dim = (128, 4, 1)  # 16 warps
    
    # 计算每个 block 处理的元素数量
    block_elements_x = (16 * block_dim[0]) // 32  # = 64
    block_elements_y = 16 * block_dim[1]          # = 64
    
    # 正确的向上取整计算（匹配 NVIDIA C++ 示例）
    def ceil_div(a, b):
        return (a + b - 1) // b
    
    grid_dim = (
        ceil_div(M_GLOBAL, block_elements_x),
        ceil_div(N_GLOBAL, block_elements_y),
        1,
    )
    
    print(f"Matrix dimensions: M={M_GLOBAL}, N={N_GLOBAL}, K={K_GLOBAL}")
    print(f"Block dimensions: {block_dim} (processes {block_elements_x}x{block_elements_y} elements per block)")
    print(f"Grid dimensions: {grid_dim} (total blocks: {grid_dim[0] * grid_dim[1]})")
    
    # ------------------------------------------------------------
    # 8. 启动 kernel
    # ------------------------------------------------------------
    print("Launching kernel...")
    checkCudaErrors(
        driver.cuLaunchKernel(
            kernel,
            grid_dim[0], grid_dim[1], grid_dim[2],
            block_dim[0], block_dim[1], block_dim[2],
            0,                      # sharedMemBytes
            stream,
            kernel_args.ctypes.data,
            0
        )
    )

    # 同步并检查错误
    print("Waiting for kernel to complete...")
    checkCudaErrors(driver.cuStreamSynchronize(stream))
    
    # ------------------------------------------------------------
    # 9. 拷贝结果回主机
    # ------------------------------------------------------------
    print("Copying results back to host...")
    checkCudaErrors(
        driver.cuMemcpyDtoHAsync(D_h.ctypes.data, dD, D_h.nbytes, stream)
    )
    checkCudaErrors(driver.cuStreamSynchronize(stream))

    print("Computation finished.")
    print("Example D[0,0] =", D_h[0, 0])
    
    # 检查是否全为零
    is_all_zero = np.allclose(D_h, 0.0)
    print("Is D all zero?", is_all_zero)
    
    if is_all_zero:
        print("ERROR: Result matrix is all zeros!")
        print("This usually indicates a problem with grid/block configuration or kernel parameters.")
        print("Please check the grid dimensions and ensure they match the matrix size.")
        sys.exit(1)
    
    # 打印更多统计信息
    print(f"Mean of D: {np.mean(D_h)}")
    print(f"Max of D: {np.max(D_h)}")
    print(f"Min of D: {np.min(D_h)}")

    # ------------------------------------------------------------
    # 10. 释放资源
    # ------------------------------------------------------------
    print("Cleaning up resources...")
    checkCudaErrors(driver.cuMemFree(dA))
    checkCudaErrors(driver.cuMemFree(dB))
    checkCudaErrors(driver.cuMemFree(dC))
    checkCudaErrors(driver.cuMemFree(dD))
    checkCudaErrors(driver.cuStreamDestroy(stream))
    
    print("Done.")


if __name__ == "__main__":
    main()
