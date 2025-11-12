from cuda.bindings import driver, nvrtc
import numpy as np

# def checkCudaErrors(result):
#     if result[0].value:
#         raise RuntimeError("CUDA error code: {}".format(result[0].value))
#     if len(result) == 1:
#         return None
#     elif len(result) == 2:
#         return result[1]
#     else:
#         return result[1:]

# 这里请替换成你实际的CUDA核函数源码字符串（或PTX）
bf16_gemm_cuda_src = r'''
extern "C" __global__
void bf16_gemm_kernel(
    const __nv_bfloat16 *A, const __nv_bfloat16 *B,
    const float *C, float *D,
    int M_GLOBAL, int N_GLOBAL, int K_GLOBAL,
    float alpha, float beta)
{
    // ... 内核实现
}
'''

def main():
    # 1. 初始化CUDA
    checkCudaErrors(driver.cuInit(0))
    cuDevice = checkCudaErrors(driver.cuDeviceGet(0))

    major = checkCudaErrors(driver.cuDeviceGetAttribute(
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice))
    minor = checkCudaErrors(driver.cuDeviceGetAttribute(
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice))
    arch_arg = bytes(f'--gpu-architecture=compute_{major}{minor}', 'ascii')

    # 2. 编译CUDA程序
    prog = checkCudaErrors(nvrtc.nvrtcCreateProgram(str.encode(bf16_gemm_cuda_src), b'bf16_gemm.cu', 0, [], []))
    opts = [arch_arg]
    checkCudaErrors(nvrtc.nvrtcCompileProgram(prog, len(opts), opts))

    ptx_size = checkCudaErrors(nvrtc.nvrtcGetPTXSize(prog))
    ptx = bytearray(ptx_size)
    checkCudaErrors(nvrtc.nvrtcGetPTX(prog, ptx))

    # 3. 创建上下文
    context = checkCudaErrors(driver.cuCtxCreate(None, 0, cuDevice))

    # 4. 加载模块和内核函数
    module = checkCudaErrors(driver.cuModuleLoadData(ptx))
    kernel = checkCudaErrors(driver.cuModuleGetFunction(module, b'bf16_gemm_kernel'))

    # 5. 初始化主机数据（根据你的程序参数替换）
    M_GLOBAL, N_GLOBAL, K_GLOBAL = 1024, 1024, 1024  # 示例大小
    alpha, beta = 1.1, 1.2

    # 使用numpy分配主机内存，bfloat16可以用uint16表示（cuda_bindings没自带__nv_bfloat16类型）
    A_h = np.random.randint(0, 65535, size=(M_GLOBAL, K_GLOBAL), dtype=np.uint16)
    B_h = np.random.randint(0, 65535, size=(K_GLOBAL, N_GLOBAL), dtype=np.uint16)
    C_h = np.random.rand(M_GLOBAL, N_GLOBAL).astype(np.float32)
    D_h = np.zeros((M_GLOBAL, N_GLOBAL), dtype=np.float32)

    # 6. 分配设备内存
    size_A = A_h.nbytes
    size_B = B_h.nbytes
    size_C = C_h.nbytes
    size_D = D_h.nbytes

    dA = checkCudaErrors(driver.cuMemAlloc(size_A))
    dB = checkCudaErrors(driver.cuMemAlloc(size_B))
    dC = checkCudaErrors(driver.cuMemAlloc(size_C))
    dD = checkCudaErrors(driver.cuMemAlloc(size_D))

    # 7. 拷贝数据到设备
    stream = checkCudaErrors(driver.cuStreamCreate(0))
    checkCudaErrors(driver.cuMemcpyHtoDAsync(dA, A_h.ctypes.data, size_A, stream))
    checkCudaErrors(driver.cuMemcpyHtoDAsync(dB, B_h.ctypes.data, size_B, stream))
    checkCudaErrors(driver.cuMemcpyHtoDAsync(dC, C_h.ctypes.data, size_C, stream))
    checkCudaErrors(driver.cuMemsetD32Async(dD, 0, size_D // 4, stream))  # float为4字节

    # 8. 设置kernel启动参数
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

    # 9. 设置线程块和网格尺寸（示例，需根据内核调整）
    threads_per_block = 256
    blocks_per_grid = (M_GLOBAL * N_GLOBAL + threads_per_block - 1) // threads_per_block

    # 10. 启动内核
    checkCudaErrors(driver.cuLaunchKernel(
        kernel,
        blocks_per_grid, 1, 1,           # gridDim
        threads_per_block, 1, 1,         # blockDim
        0,                              # sharedMemBytes
        stream,
        kernel_args.ctypes.data,
        0
    ))

    checkCudaErrors(driver.cuStreamSynchronize(stream))

    # 11. 拷贝结果回主机
    checkCudaErrors(driver.cuMemcpyDtoHAsync(D_h.ctypes.data, dD, size_D, stream))
    checkCudaErrors(driver.cuStreamSynchronize(stream))

    print("Computation finished, example D[0,0] =", D_h[0, 0])

    # 12. 释放资源
    checkCudaErrors(driver.cuMemFree(dA))
    checkCudaErrors(driver.cuMemFree(dB))
    checkCudaErrors(driver.cuMemFree(dC))
    checkCudaErrors(driver.cuMemFree(dD))
    checkCudaErrors(driver.cuStreamDestroy(stream))
    checkCudaErrors(driver.cuModuleUnload(module))
    checkCudaErrors(driver.cuCtxDestroy(context))


if __name__ == '__main__':
    main()
