from cuda.bindings import driver, nvrtc
import numpy as np
import ctypes

def checkCudaErrors(result):
    if result[0].value != 0:
        err_name = driver.cuGetErrorName(result[0])[1]
        raise RuntimeError(f'CUDA error {result[0].value} ({err_name})')
    return result[1] if len(result) > 1 else None

kernel_code = r"""
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

extern "C" __global__
void oddEvenCountAndSumCG(int *inputArr, int *numOfOdds, int *sumOfOddAndEvens, unsigned int size)
{
    cg::thread_block          cta    = cg::this_thread_block();
    cg::grid_group            grid   = cg::this_grid();
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    for (int i = grid.thread_rank(); i < size; i += grid.size()) {
        int  elem    = inputArr[i];
        auto subTile = cg::binary_partition(tile32, elem & 1);
        if (elem & 1) // Odd numbers group
        {
            int oddGroupSum = cg::reduce(subTile, elem, cg::plus<int>());

            if (subTile.thread_rank() == 0) {
                atomicAdd(numOfOdds, subTile.size());
                atomicAdd(&sumOfOddAndEvens[0], oddGroupSum);
            }
        }
        else // Even numbers group
        {
            int evenGroupSum = cg::reduce(subTile, elem, cg::plus<int>());

            if (subTile.thread_rank() == 0) {
                atomicAdd(&sumOfOddAndEvens[1], evenGroupSum);
            }
        }
        cg::sync(tile32);
    }
}
"""

def main():
    # 初始化 CUDA
    checkCudaErrors(driver.cuInit(0))

    # 选择设备 0
    cuDevice = checkCudaErrors(driver.cuDeviceGet(0))

    # 获取架构版本
    major = checkCudaErrors(driver.cuDeviceGetAttribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice))
    minor = checkCudaErrors(driver.cuDeviceGetAttribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice))
    arch_flag = bytes(f'--gpu-architecture=compute_{major}{minor}', 'ascii')

    # 创建 nvrtc 程序
    prog = checkCudaErrors(nvrtc.nvrtcCreateProgram(kernel_code.encode('utf-8'), b"oddEven.cu", 0, [], []))

    # 编译
    opts = [arch_flag]
    checkCudaErrors(nvrtc.nvrtcCompileProgram(prog, len(opts), opts))

    # 获取 PTX
    ptx_size = checkCudaErrors(nvrtc.nvrtcGetPTXSize(prog))
    ptx = b' ' * ptx_size
    checkCudaErrors(nvrtc.nvrtcGetPTX(prog, ptx))

    # 创建上下文
    context = checkCudaErrors(driver.cuCtxCreate(None, 0, cuDevice))

    # 加载模块
    module = checkCudaErrors(driver.cuModuleLoadData(ctypes.c_char_p(ptx)))

    # 获取函数句柄
    kernel = checkCudaErrors(driver.cuModuleGetFunction(module, b"oddEvenCountAndSumCG"))

    # 准备数据
    arr_size = 1024 * 100
    host_input = np.random.randint(0, 50, size=arr_size).astype(np.int32)
    host_num_of_odds = np.zeros(1, dtype=np.int32)
    host_sum_of_odd_even = np.zeros(2, dtype=np.int32)

    buf_size = host_input.nbytes
    # 申请设备内存
    d_input = checkCudaErrors(driver.cuMemAlloc(buf_size))
    d_num_of_odds = checkCudaErrors(driver.cuMemAlloc(host_num_of_odds.nbytes))
    d_sum_of_odd_even = checkCudaErrors(driver.cuMemAlloc(host_sum_of_odd_even.nbytes))

    # 创建流
    stream = checkCudaErrors(driver.cuStreamCreate(0))

    # 传输数据到设备
    checkCudaErrors(driver.cuMemcpyHtoDAsync(d_input, host_input.ctypes.data, buf_size, stream))
    checkCudaErrors(driver.cuMemsetD32Async(d_num_of_odds, 0, 1, stream))
    checkCudaErrors(driver.cuMemsetD32Async(d_sum_of_odd_even, 0, 2, stream))

    # 准备内核参数，传入指针和大小
    # 注意：内核参数是指针数组，传入指针的地址（uint64）
    args = (ctypes.c_void_p * 4)()
    args[0] = ctypes.c_void_p(d_input)
    args[1] = ctypes.c_void_p(d_num_of_odds)
    args[2] = ctypes.c_void_p(d_sum_of_odd_even)
    args[3] = ctypes.c_void_p(ctypes.c_uint(arr_size))

    # 计算启动参数（你也可以自己指定）
    blocks_per_grid = 0
    threads_per_block = 0
    # 计算最大潜在线程数
    blocks_ptr = ctypes.c_int()
    threads_ptr = ctypes.c_int()
    checkCudaErrors(driver.cuOccupancyMaxPotentialBlockSize(
        ctypes.byref(blocks_ptr),
        ctypes.byref(threads_ptr),
        kernel,
        0,
        0))
    blocks_per_grid = blocks_ptr.value
    threads_per_block = threads_ptr.value

    # 启动核函数
    checkCudaErrors(driver.cuLaunchKernel(
        kernel,
        blocks_per_grid, 1, 1,
        threads_per_block, 1, 1,
        0,
        stream,
        ctypes.byref(args),
        None
    ))

    # 复制结果回主机
    checkCudaErrors(driver.cuMemcpyDtoHAsync(host_num_of_odds.ctypes.data, d_num_of_odds, host_num_of_odds.nbytes, stream))
    checkCudaErrors(driver.cuMemcpyDtoHAsync(host_sum_of_odd_even.ctypes.data, d_sum_of_odd_even, host_sum_of_odd_even.nbytes, stream))

    # 等待完成
    checkCudaErrors(driver.cuStreamSynchronize(stream))

    print(f"Array size = {arr_size}")
    print(f"Number of odd elements = {host_num_of_odds[0]}")
    print(f"Sum of odd elements = {host_sum_of_odd_even[0]}")
    print(f"Sum of even elements = {host_sum_of_odd_even[1]}")

    # 释放资源
    checkCudaErrors(driver.cuStreamDestroy(stream))
    checkCudaErrors(driver.cuMemFree(d_input))
    checkCudaErrors(driver.cuMemFree(d_num_of_odds))
    checkCudaErrors(driver.cuMemFree(d_sum_of_odd_even))
    checkCudaErrors(driver.cuModuleUnload(module))
    checkCudaErrors(driver.cuCtxDestroy(context))

if __name__ == "__main__":
    main()
