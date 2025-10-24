from common.helper_cuda import findCudaDevice, checkCudaErrors
from cuda.bindings import driver, runtime
import sys
import numpy as np
from common import common
import ctypes, time
from common.helper_string import sdkFindFilePath


def convertSmVerToArchName(major: int, minor: int) -> str:
    """
    根据 GPU 的计算能力版本返回对应的架构名称
    """
    sm_arch_map = [
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

    combined_ver = (major << 4) + minor

    for sm_ver, arch_name in sm_arch_map:
        if sm_ver == combined_ver:
            return arch_name

    return "Graphics Device"


INCREMENT_KERNEL = """\
extern "C" __global__ void increment_kernel(int *g_data, int inc_value)
{
    int idx     = blockIdx.x * blockDim.x + threadIdx.x;
    g_data[idx] = g_data[idx] + inc_value;
}
"""


def verify_output(data: list, n: int, inc_val: int) -> bool:
    """验证输出数据是否正确"""
    for i in range(n):
        if data[i] != inc_val:
            print(f"Error! data[{i}] = {data[i]}, expected = {inc_val}")
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

    device_name = checkCudaErrors(driver.cuDeviceGetName(100, device))
    print(f"CUDA device  [{device_name.decode('utf-8').rstrip()}]")

    # 通过字符串的方式创建核函数
    kernel_helper = common.KernelHelper(INCREMENT_KERNEL, device)
    kernel_func = kernel_helper.getFunction(b"increment_kernel")

    # 配置参数
    n = 16 * 1024 * 1024
    byte_size = np.dtype(np.int32).itemsize * n
    inc_val = np.array(26, dtype=np.int32)

    # 分配主机内存
    host_ptr = checkCudaErrors(driver.cuMemAllocHost(byte_size))
    host_buf = (ctypes.c_byte * byte_size).from_address(host_ptr)
    host_arr = np.frombuffer(host_buf, dtype=np.int32, count=n)
    host_arr.fill(0)

    # 分配设备内存
    dev_ptr = checkCudaErrors(driver.cuMemAlloc(byte_size))
    checkCudaErrors(runtime.cudaMemset(dev_ptr, 255, byte_size))

    # 配置核函数执行参数
    block_size = 512
    grid_size = n // block_size

    # 创建事件用于计时
    start_evt = checkCudaErrors(driver.cuEventCreate(0))
    stop_evt = checkCudaErrors(driver.cuEventCreate(0))

    # 开始计时
    cpu_start = time.perf_counter()
    checkCudaErrors(runtime.cudaDeviceSynchronize())

    gpu_time = 0.0
    checkCudaErrors(runtime.cudaProfilerStart())
    runtime.cudaEventRecord(start_evt, 0)

    # 数据传输：主机到设备
    runtime.cudaMemcpyAsync(
        dev_ptr, host_ptr, byte_size, runtime.cudaMemcpyKind.cudaMemcpyHostToDevice, 0
    )

    # 准备核函数参数
    kernel_args = [np.array([int(dev_ptr)], dtype=np.uint64), inc_val]
    kernel_args_arr = np.array(
        [arg.ctypes.data for arg in kernel_args], dtype=np.uint64
    )

    # 启动核函数
    checkCudaErrors(
        driver.cuLaunchKernel(
            kernel_func,
            grid_size, 1, 1,
            block_size, 1, 1,
            0,
            0,
            kernel_args_arr.ctypes.data,
            0,
        )
    )

    # 数据传输：设备到主机
    runtime.cudaMemcpyAsync(
        host_ptr, dev_ptr, byte_size, runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost, 0
    )
    runtime.cudaEventRecord(stop_evt, 0)
    cpu_end = time.perf_counter()
    checkCudaErrors(runtime.cudaProfilerStop())

    # 等待GPU完成
    wait_count = 0
    while runtime.cudaEventQuery(stop_evt)[0] == runtime.cudaError_t.cudaErrorNotReady:
        wait_count = wait_count + 1

    # 计算执行时间
    gpu_time = checkCudaErrors(runtime.cudaEventElapsedTime(start_evt, stop_evt))
    print(f"GPU execution time: {gpu_time:.2f} ms")
    print(f"CPU CUDA call time: {(cpu_end - cpu_start) * 1000:.2f} ms")
    print(f"CPU wait iterations: {wait_count}")

    # 验证结果
    result_ok = verify_output(host_arr, n, inc_val)

    # 清理资源
    checkCudaErrors(driver.cuMemFreeHost(host_ptr))
    checkCudaErrors(driver.cuMemFree(dev_ptr))
    checkCudaErrors(driver.cuEventDestroy(start_evt))
    checkCudaErrors(driver.cuEventDestroy(stop_evt))

    sys.exit(0 if result_ok else 1)


if __name__ == "__main__":
    main()
