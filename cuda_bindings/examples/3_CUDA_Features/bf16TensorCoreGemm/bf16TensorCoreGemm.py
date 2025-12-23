import sys
import numpy as np

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

    kernel = checkCudaErrors(
        driver.cuModuleGetFunction(module, b"simple_wmma_bf16gemm")
    )

    # ------------------------------------------------------------
    # 3. 主机侧数据准备（严格匹配 kernel 假设）
    # ------------------------------------------------------------
    M_GLOBAL, N_GLOBAL, K_GLOBAL = 1024, 1024, 1024
    alpha, beta = 1.1, 1.2

    # A：row-major（正确）
    A_f32 = np.random.rand(M_GLOBAL, K_GLOBAL).astype(np.float32)

    # B：必须是 column-major
    # 关键修正点：N x K -> 转置 -> copy
    B_f32 = np.random.rand(N_GLOBAL, K_GLOBAL).astype(np.float32).T.copy()

    # 转 BF16（uint16）
    A_h = float32_to_bfloat16(A_f32)
    B_h = float32_to_bfloat16(B_f32)

    # C / D：FP32，row-major
    C_h = np.random.rand(M_GLOBAL, N_GLOBAL).astype(np.float32)
    D_h = np.zeros((M_GLOBAL, N_GLOBAL), dtype=np.float32)

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
    # 6. kernel 参数（顺序必须严格一致）
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
    # 7. grid / block 设置（完全对齐 C++ 示例）
    # ------------------------------------------------------------
    block_dim = (128, 4, 1)   # 16 warps

    block_elements_x = (16 * block_dim[0]) // 32   # 64
    block_elements_y = 16 * block_dim[1]           # 64

    def ceil_div(a, b):
        return (a + b - 1) // b

    grid_dim = (
        ceil_div(M_GLOBAL, block_elements_x),
        ceil_div(N_GLOBAL, block_elements_y),
        1,
    )

    print(f"Grid: {grid_dim}, Block: {block_dim}")

    # ------------------------------------------------------------
    # 8. 启动 kernel
    # ------------------------------------------------------------
    checkCudaErrors(
        driver.cuLaunchKernel(
            kernel,
            grid_dim[0], grid_dim[1], grid_dim[2],
            block_dim[0], block_dim[1], block_dim[2],
            0,          # sharedMemBytes
            stream,
            kernel_args.ctypes.data,
            0
        )
    )

    checkCudaErrors(driver.cuStreamSynchronize(stream))

    # ------------------------------------------------------------
    # 9. 拷贝结果回主机
    # ------------------------------------------------------------
    checkCudaErrors(
        driver.cuMemcpyDtoHAsync(D_h.ctypes.data, dD, D_h.nbytes, stream)
    )
    checkCudaErrors(driver.cuStreamSynchronize(stream))

    # ------------------------------------------------------------
    # 10. 简单正确性检查（左上角 16x16）
    # ------------------------------------------------------------
    D_ref = alpha * (A_f32[:16, :16] @ B_f32[:16, :16]) + beta * C_h[:16, :16]
    err = np.max(np.abs(D_ref - D_h[:16, :16]))

    print("Example D[0,0] =", D_h[0, 0])
    print("Max abs error (16x16 tile):", err)

    # ------------------------------------------------------------
    # 11. 清理资源
    # ------------------------------------------------------------
    checkCudaErrors(driver.cuMemFree(dA))
    checkCudaErrors(driver.cuMemFree(dB))
    checkCudaErrors(driver.cuMemFree(dC))
    checkCudaErrors(driver.cuMemFree(dD))
    checkCudaErrors(driver.cuStreamDestroy(stream))

    print("Done.")


if __name__ == "__main__":
    main()
