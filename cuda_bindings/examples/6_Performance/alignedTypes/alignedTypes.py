from cuda.bindings import driver, nvrtc
import numpy as np
import ctypes

def checkCudaErrors(result):
    # 省略，使用你给的check函数
    ...

# 这里是cuda核函数示例(只示意，非完整)
kernel_code = r"""
template <typename T>
__global__ void testKernel(T* d_odata, T* d_idata, int numElements) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int numThreads = blockDim.x * gridDim.x;
    for (int pos = tid; pos < numElements; pos += numThreads) {
        d_odata[pos] = d_idata[pos];
    }
}
"""

def runTest(dtype, packedElementSize, memory_size):
    # 初始化设备、编译内核(省略，类似你的saxpy例子)
    # 计算元素个数
    numElements = memory_size // np.dtype(dtype).itemsize

    # 分配设备内存
    d_idata = checkCudaErrors(driver.cuMemAlloc(memory_size))
    d_odata = checkCudaErrors(driver.cuMemAlloc(memory_size))

    # 在主机上准备输入数据
    h_idata = (np.arange(memory_size) % 256).astype(dtype)
    h_odata = np.zeros_like(h_idata)

    # 复制输入到设备
    checkCudaErrors(driver.cuMemcpyHtoD(d_idata, h_idata.ctypes.data, memory_size))

    # 配置线程
    block_size = 256
    grid_size = 64

    # 准备内核参数
    numElements_ct = ctypes.c_int(numElements)
    args = (ctypes.c_void_p * 3)()
    args[0] = ctypes.c_void_p(d_odata)
    args[1] = ctypes.c_void_p(d_idata)
    args[2] = ctypes.cast(ctypes.pointer(numElements_ct), ctypes.c_void_p)

    # 启动内核
    checkCudaErrors(driver.cuLaunchKernel(
        kernel_function_handle,
        grid_size, 1, 1,
        block_size, 1, 1,
        0, None,
        ctypes.cast(args, ctypes.POINTER(ctypes.c_void_p)),
        None,
    ))

    # 拷贝结果回主机
    checkCudaErrors(driver.cuMemcpyDtoH(h_odata.ctypes.data, d_odata, memory_size))

    # 验证结果
    if not np.array_equal(h_odata, h_idata):
        print("Test failure")
        return False
    else:
        print("Test success")
        return True

# 主流程，初始化CUDA，创建context，编译内核，获取kernel handle
# 调用runTest

# 细节需根据实际改写完善，比如nvrtc编译内核代码，获取kernel handle等
