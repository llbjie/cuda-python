import sys
from cuda.bindings import driver, nvrtc
from common import common
from common.helper_cuda import checkCudaErrors, findCudaDevice
from common.helper_string import sdkFindFilePath
import numpy as np

def float32_to_bfloat16(f32_array):
    f32_as_uint32 = f32_array.view(np.uint32)
    bfloat16_uint16 = (f32_as_uint32 >> 16).astype(np.uint16)
    return bfloat16_uint16

def bfloat16_to_float32(bf16_uint16):
    f32_as_uint32 = (bf16_uint16.astype(np.uint32) << 16)
    return f32_as_uint32.view(np.float32)

def main():
    dev = findCudaDevice()
    
    # 1. 加载函数
    kernel_path = sdkFindFilePath("kernels.ptx", sys.argv[0])
    with open(kernel_path, "rb") as f:
        # kernels = f.read()
        ptx_data = f.read()
    module = checkCudaErrors(driver.cuModuleLoadData(ptx_data))
    kernel_1 = checkCudaErrors(driver.cuModuleGetFunction(module, "compute_bf16gemm".encode()))

    
    # kernelHelper = common.KernelHelper(kernels.decode('utf-8'), dev)    
    # kernel_1 = kernelHelper.getFunction(b"compute_bf16gemm")
    # kernel_2 = kernelHelper.getFunction(b"simple_wmma_bf16gemm")
    # kernel_3 = kernelHelper.getFunction(b"compute_bf16gemm_async_copy")

    # 2. 初始化主机数据
    M_GLOBAL, N_GLOBAL, K_GLOBAL = 1024, 1024, 1024  # 示例大小
    alpha, beta = 1.1, 1.2

    # 使用numpy分配主机内存，bfloat16可以用uint16表示（cuda_bindings没自带__nv_bfloat16类型）
    A_h = np.random.randint(0, 65535, size=(M_GLOBAL, K_GLOBAL), dtype=np.uint16)
    B_h = np.random.randint(0, 65535, size=(K_GLOBAL, N_GLOBAL), dtype=np.uint16)
    C_h = np.random.rand(M_GLOBAL, N_GLOBAL).astype(np.float32)
    D_h = np.zeros((M_GLOBAL, N_GLOBAL), dtype=np.float32)

    # 3. 分配设备内存
    size_A = A_h.nbytes
    size_B = B_h.nbytes
    size_C = C_h.nbytes
    size_D = D_h.nbytes

    dA = checkCudaErrors(driver.cuMemAlloc(size_A))
    dB = checkCudaErrors(driver.cuMemAlloc(size_B))
    dC = checkCudaErrors(driver.cuMemAlloc(size_C))
    dD = checkCudaErrors(driver.cuMemAlloc(size_D))

    # 4. 拷贝数据到设备
    stream = checkCudaErrors(driver.cuStreamCreate(0))
    checkCudaErrors(driver.cuMemcpyHtoDAsync(dA, A_h.ctypes.data, size_A, stream))
    checkCudaErrors(driver.cuMemcpyHtoDAsync(dB, B_h.ctypes.data, size_B, stream))
    checkCudaErrors(driver.cuMemcpyHtoDAsync(dC, C_h.ctypes.data, size_C, stream))
    checkCudaErrors(driver.cuMemsetD32Async(dD, 0, size_D // 4, stream))  # float为4字节

    # 5. 设置kernel启动参数
    # 注意，__nv_bfloat16对应uint16_t，这里指针传递即可，大小在内核中正确处理
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
    # 打包为指针数组
    kernel_args = np.array([arg.ctypes.data for arg in args], dtype=np.uint64)

    # 6. 设置线程块和网格尺寸
    threads_per_block = 256
    blocks_per_grid = (M_GLOBAL * N_GLOBAL + threads_per_block - 1) // threads_per_block

    # 7. 启动内核
    checkCudaErrors(driver.cuLaunchKernel(
        kernel_1,
        blocks_per_grid, 1, 1,           # gridDim
        threads_per_block, 1, 1,         # blockDim
        0,                              # sharedMemBytes
        stream,
        kernel_args.ctypes.data,
        0
    ))

    checkCudaErrors(driver.cuStreamSynchronize(stream))

    # 7. 拷贝结果回主机
    checkCudaErrors(driver.cuMemcpyDtoHAsync(D_h.ctypes.data, dD, size_D, stream))
    checkCudaErrors(driver.cuStreamSynchronize(stream))

    print("Computation finished, example D[0,0] =", D_h[0, 0])

    # 8. 释放资源
    checkCudaErrors(driver.cuMemFree(dA))
    checkCudaErrors(driver.cuMemFree(dB))
    checkCudaErrors(driver.cuMemFree(dC))
    checkCudaErrors(driver.cuMemFree(dD))
    checkCudaErrors(driver.cuStreamDestroy(stream))



if __name__ == '__main__':
    main()
