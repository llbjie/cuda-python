from common.helper_cuda import findCudaDevice, checkCudaErrors
from cuda.bindings import driver, runtime
import sys
import numpy as np
from common import common
import ctypes, time
from common.helper_string import sdkFindFilePath

def convertSmVerToArchName(major: int, minor: int) -> str:
    """
    根据 GPU 的计算能力版本（主版本和次版本）返回对应的架构名称

    参数:
        major: 计算能力主版本 (如 8)
        minor: 计算能力次版本 (如 9)

    返回:
        架构名称字符串 (如 "Ada")
    """
    smToArch = [
        (0x30, "Kepler"),
        (0x32, "Kepler"),
        (0x35, "Kepler"),
        (0x37, "Kepler"),
        (0x50, "Maxwell"),
        (0x52, "Maxwell"),
        (0x53, "Maxwell"),
        (0x60, "Pascal"),
        (0x61, "Pascal"),
        (0x62, "Pascal"),
        (0x70, "Volta"),
        (0x72, "Xavier"),
        (0x75, "Turing"),
        (0x80, "Ampere"),
        (0x86, "Ampere"),
        (0x87, "Ampere"),
        (0x89, "Ada"),
        (0x90, "Hopper"),
        (0xA0, "Blackwell"),
        (0xA1, "Blackwell"),
        (0xA3, "Blackwell"),
        (0xB0, "Blackwell"),
        (0xC0, "Blackwell"),
        (0xC1, "Blackwell"),
        (-1, "Graphics Device"),
    ]

    combinedVersion = (major << 4) + minor

    for smVersion, archName in smToArch:
        if smVersion == combinedVersion:
            return archName

    return "Graphics Device"


incrementKernel = """\
extern "C" __global__ void increment_kernel(int *g_data, int inc_value)
{
    int idx     = blockIdx.x * blockDim.x + threadIdx.x;
    g_data[idx] = g_data[idx] + inc_value;
}
"""


def correctOutput(data: list, n: int, incValue: int) -> bool:
    for i in range(n):
        if data[i] != incValue:
            print(f"Error! data[{i}] = {data[i]}, ref = {incValue}")
            return False
    return True


def main():
    print(f"[{sys.argv[0]}] - Starting...\n")
    checkCudaErrors(driver.cuInit(0))

    device = findCudaDevice()

    major = checkCudaErrors(
        driver.cuDeviceGetAttribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            device,
        )
    )
    minor = checkCudaErrors(
        driver.cuDeviceGetAttribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            device,
        )
    )

    print(
        f'GPU Device {device}: "{convertSmVerToArchName(major, minor)}" with compute capability {major}.{minor}'
    )

    name = checkCudaErrors(driver.cuDeviceGetName(100, device))
    print(f"CUDA device  [{name.decode('utf-8').rstrip()}]")

    # 通过 .cu 文件保存核函数
    # kernel_path = sdkFindFilePath("increment_kernel.cu", sys.argv[0])
    # if kernel_path is None:
    #   print("The file 'increment_kernel.cu' could not be found in the executable directory")
    #   sys.exit(1)
    # with open(kernel_path, 'r') as f:
    #     increment_kernel = f.read() 
    # kernelHelper = common.KernelHelper(increment_kernel, device)
    
    # 通过字符串的方式
    # kernelHelper = common.KernelHelper(incrementKernel, device)

    # 
    # kernel = kernelHelper.getFunction(b"increment_kernel")

    # 通过.ptx文件的方式获取和函数
    kernel_path = sdkFindFilePath("increment_kernel.ptx", sys.argv[0])
    with open(kernel_path, 'rb') as f:
        ptx_data = f.read()
    err, module = driver.cuModuleLoadData(ptx_data)
    err, kernel = driver.cuModuleGetFunction(module, "increment_kernel".encode())

    n = 16 * 1024 * 1024
    numBytes = np.dtype(np.int32).itemsize * n
    incValue = np.array(26, dtype=np.int32)

    hostPtr = checkCudaErrors(driver.cuMemAllocHost(numBytes))
    buffer = (ctypes.c_byte * numBytes).from_address(hostPtr)
    hostArray = np.frombuffer(buffer, dtype=np.int32, count=n)
    hostArray.fill(0)

    devicePtr = checkCudaErrors(driver.cuMemAlloc(numBytes))
    checkCudaErrors(runtime.cudaMemset(devicePtr, 255, numBytes))

    numThreads = 512
    numBlocks = n // numThreads

    startEvent = checkCudaErrors(driver.cuEventCreate(0))
    stopEvent = checkCudaErrors(driver.cuEventCreate(0))

    timeBegin = time.perf_counter()
    checkCudaErrors(runtime.cudaDeviceSynchronize())

    gpuTime = 0.0
    checkCudaErrors(runtime.cudaProfilerStart())
    runtime.cudaEventRecord(startEvent, 0)
    runtime.cudaMemcpyAsync(
        devicePtr, hostPtr, numBytes, runtime.cudaMemcpyKind.cudaMemcpyHostToDevice, 0
    )

    kernelArgs = [np.array([int(devicePtr)], dtype=np.uint64), incValue]
    kernelArgs = np.array([arg.ctypes.data for arg in kernelArgs], dtype=np.uint64)

    checkCudaErrors(
        driver.cuLaunchKernel(
            kernel,
            numBlocks,
            1,
            1,
            numThreads,
            1,
            1,
            0,
            0,
            kernelArgs.ctypes.data,
            0,
        )
    )

    runtime.cudaMemcpyAsync(
        hostPtr, devicePtr, numBytes, runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost, 0
    )
    runtime.cudaEventRecord(stopEvent, 0)
    timeEnd = time.perf_counter()
    checkCudaErrors(runtime.cudaProfilerStop())

    counter = 0
    while runtime.cudaEventQuery(stopEvent)[0] == runtime.cudaError_t.cudaErrorNotReady:
        counter = counter + 1

    gpuTime = checkCudaErrors(runtime.cudaEventElapsedTime(startEvent, stopEvent))
    print(f"time spent executing by the GPU: {gpuTime:.2f}")
    print(f"time spent by CPU in CUDA calls: {(timeEnd - timeBegin) * 1000:.2f}")
    print(f"CPU executed {counter} iterations while waiting for GPU to finish")

    result = correctOutput(hostArray, n, incValue)

    checkCudaErrors(driver.cuMemFreeHost(hostPtr))
    checkCudaErrors(driver.cuMemFree(devicePtr))
    checkCudaErrors(driver.cuEventDestroy(startEvent))
    checkCudaErrors(driver.cuEventDestroy(stopEvent))

    sys.exit(result)


if __name__ == "__main__":
    main()
