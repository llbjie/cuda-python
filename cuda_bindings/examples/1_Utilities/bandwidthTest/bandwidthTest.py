#!/usr/bin/env python3

import sys
import argparse
import time
import ctypes
from enum import Enum
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from cuda.bindings import runtime as rt, driver
from common.helper_cuda import findCudaDevice, checkCudaErrors


class TestMode(Enum):
    QUICK_MODE = "quick"
    RANGE_MODE = "range"
    SHMOO_MODE = "shmoo"


class MemcpyKind(Enum):
    DEVICE_TO_HOST = 0
    HOST_TO_DEVICE = 1
    DEVICE_TO_DEVICE = 2


class PrintMode(Enum):
    USER_READABLE = 0
    CSV = 1


class MemoryMode(Enum):
    PAGEABLE = 0
    PINNED = 1
    NONEED = 3


# 常量定义
MEMCOPY_ITERATIONS = 100
DEFAULT_SIZE = 32 * 1024 * 1024  # 32 MB
DEFAULT_INCREMENT = 4 * 1024 * 1024  # 4 MB
CACHE_CLEAR_SIZE = 16 * 1024 * 1024  # 16 MB
FLUSH_SIZE = 256 * 1024 * 1024  # 256 MB

# Shmoo模式常量
SHMOO_MEMSIZE_MAX = 64 * 1024 * 1024  # 64 MB
SHMOO_MEMSIZE_START = 1024  # 1 KB
SHMOO_INCREMENT_1KB = 1024  # 1 KB
SHMOO_INCREMENT_2KB = 2 * 1024  # 2 KB
SHMOO_INCREMENT_10KB = 10 * 1024  # 10 KB
SHMOO_INCREMENT_100KB = 100 * 1024  # 100 KB
SHMOO_INCREMENT_1MB = 1024 * 1024  # 1 MB
SHMOO_INCREMENT_2MB = 2 * 1024 * 1024  # 2 MB
SHMOO_INCREMENT_4MB = 4 * 1024 * 1024  # 4 MB
SHMOO_LIMIT_20KB = 20 * 1024  # 20 KB
SHMOO_LIMIT_50KB = 50 * 1024  # 50 KB
SHMOO_LIMIT_100KB = 100 * 1024  # 100 KB
SHMOO_LIMIT_1MB = 1024 * 1024  # 1 MB
SHMOO_LIMIT_16MB = 16 * 1024 * 1024  # 16 MB
SHMOO_LIMIT_32MB = 32 * 1024 * 1024  # 32 MB

# 全局变量
bDontUseGPUTiming = False
flush_buf = None

# 字符串常量
sMemoryCopyKind = ["Device to Host", "Host to Device", "Device to Device"]
sMemoryMode = ["PAGEABLE", "PINNED"]


class CUDABandwidthTester:
    def __init__(self):
        self.device_count = 0
        self._initialize_cuda()

    def _transfer_time(self, dst, src, mem_mode: MemoryMode, mem_size: int) -> float:
        """执行内存拷贝测试并计算带宽"""
        if mem_mode == MemoryMode.PINNED and not bDontUseGPUTiming:
            # 使用GPU事件计时
            start_event = checkCudaErrors(driver.cuEventCreate(0))
            stop_event = checkCudaErrors(driver.cuEventCreate(0))
            rt.cudaEventRecord(start_event, 0)
            for _ in range(MEMCOPY_ITERATIONS):
                driver.cuMemcpy(dst, src, mem_size)
            rt.cudaEventRecord(stop_event, 0)
            rt.cudaDeviceSynchronize()
            elapsed_time_in_ms = checkCudaErrors(
                driver.cuEventElapsedTime(start_event, stop_event)
            )
            driver.cuEventDestroy(start_event)
            driver.cuEventDestroy(stop_event)
        else:
            # 使用CPU计时
            start_time = time.time()
            for _ in range(MEMCOPY_ITERATIONS):
                driver.cuMemcpy(dst, src, mem_size)
            rt.cudaDeviceSynchronize()
            elapsed_time_in_ms = (time.time() - start_time) * 1000

        # 计算带宽 (GB/s)
        time_s = elapsed_time_in_ms / 1000.0
        bandwidth_in_gbs = (mem_size * MEMCOPY_ITERATIONS) / 1e9 / time_s

        return bandwidth_in_gbs

    def _initialize_cuda(self):
        """初始化CUDA环境"""
        driver.cuInit(0)
        result, self.device_count = driver.cuDeviceGetCount()
        if result != driver.CUresult.CUDA_SUCCESS:
            print("!!!!!No devices found!!!!!")
            sys.exit(1)

    def get_device_info(self, device_id: int) -> Dict[str, Any]:
        """获取设备信息"""
        prop = checkCudaErrors(rt.cudaGetDeviceProperties(device_id))
        attr = checkCudaErrors(
            rt.cudaDeviceGetAttribute(
                rt.cudaDeviceAttr.cudaDevAttrComputeMode, device_id
            )
        )

        return {
            "name": prop.name.decode("utf-8"),
            "compute_mode": rt.cudaComputeMode(attr),
        }

    def print_device_info(self, start_device: int, end_device: int):
        """打印设备信息"""
        print("Running on...\n")

        for device_id in range(start_device, end_device + 1):
            info = self.get_device_info(device_id)
            if info:
                print(f" Device {device_id}: {info['name']}")
                if info["compute_mode"] == rt.cudaComputeMode.cudaComputeModeProhibited:
                    print("Error: device is running in <Compute Mode Prohibited>, no threads can use cudaSetDevice().")
                    sys.exit(1)

    def _get_pointer(self, obj, mem_mode: MemoryMode) -> int:
        """获取对象的内存指针"""
        if mem_mode == MemoryMode.PAGEABLE and hasattr(obj, 'ctypes'):
            return obj.ctypes.data
        return obj

    def test_device_to_host_transfer(
        self, mem_size: int, mem_mode: MemoryMode = MemoryMode.PINNED, wc: bool = False
    ) -> float:
        """测试设备到主机的传输带宽"""
        # 分配主机内存
        if mem_mode == MemoryMode.PINNED:
            h_idata_ptr = checkCudaErrors(driver.cuMemAllocHost(mem_size))
            h_odata_ptr = checkCudaErrors(driver.cuMemAllocHost(mem_size))
            h_idata = np.frombuffer((ctypes.c_byte * mem_size).from_address(h_idata_ptr), dtype=np.uint8)
            h_odata = np.frombuffer((ctypes.c_byte * mem_size).from_address(h_odata_ptr), dtype=np.uint8)
        else:
            h_idata = np.empty(mem_size, dtype=np.uint8)
            h_odata = np.empty(mem_size, dtype=np.uint8)
            h_idata_ptr = h_idata.ctypes.data
            h_odata_ptr = h_odata.ctypes.data

        # 初始化主机内存
        h_idata[:] = np.arange(mem_size, dtype=np.uint8) 

        # 分配设备内存
        d_idata = checkCudaErrors(driver.cuMemAlloc(mem_size))

        # 复制数据到设备
        driver.cuMemcpy(d_idata, h_idata_ptr, mem_size)

        # 测试传输带宽
        bandwidth = self._transfer_time(h_odata_ptr, d_idata, mem_mode, mem_size)

        # 清理内存
        driver.cuMemFree(d_idata)
        if mem_mode == MemoryMode.PINNED:
            driver.cuMemFreeHost(h_idata_ptr)
            driver.cuMemFreeHost(h_odata_ptr)

        return bandwidth

    def test_host_to_device_transfer(
        self, mem_size: int, mem_mode: MemoryMode, wc: bool = False
    ) -> float:
        """测试主机到设备的传输带宽"""
        # 分配主机内存
        if mem_mode == MemoryMode.PINNED:
            h_odata_ptr = checkCudaErrors(driver.cuMemAllocHost(mem_size))
            h_odata = np.frombuffer((ctypes.c_byte * mem_size).from_address(h_odata_ptr), dtype=np.uint8)
        else:
            h_odata = np.empty(mem_size, dtype=np.uint8)
            h_odata_ptr = h_odata.ctypes.data

        # 初始化主机内存
        h_odata[:] = np.arange(mem_size, dtype=np.uint8)

        # 分配设备内存
        d_idata = checkCudaErrors(driver.cuMemAlloc(mem_size))

        # 测试传输带宽
        bandwidth = self._transfer_time(d_idata, h_odata_ptr, mem_mode, mem_size)

        # 清理内存
        driver.cuMemFree(d_idata)
        if mem_mode == MemoryMode.PINNED:
            driver.cuMemFreeHost(h_odata_ptr)

        return bandwidth

    def test_device_to_device_transfer(self, mem_size: int) -> float:
        """测试设备到设备的传输带宽"""
        # 分配主机内存用于初始化
        h_idata = np.empty(mem_size, dtype=np.uint8)
        h_idata[:] = np.arange(mem_size, dtype=np.uint8) 

        # 分配设备内存
        d_idata = checkCudaErrors(driver.cuMemAlloc(mem_size))
        d_odata = checkCudaErrors(driver.cuMemAlloc(mem_size))

        # 初始化设备内存
        driver.cuMemcpy(d_idata, h_idata.ctypes.data, mem_size)

        # 测试传输带宽
        bandwidth = 2 * self._transfer_time(d_odata, d_idata, MemoryMode.NONEED, mem_size)

        # 清理内存
        driver.cuMemFree(d_idata)
        driver.cuMemFree(d_odata)

        return bandwidth

    def test_bandwidth_quick(
        self,
        size: int,
        kind: MemcpyKind,
        printmode: PrintMode,
        mem_mode: MemoryMode,
        start_device: int,
        end_device: int,
        wc: bool,
    ):
        """快速模式带宽测试"""
        self.test_bandwidth_range(
            size, size, DEFAULT_INCREMENT, kind, printmode, mem_mode,
            start_device, end_device, wc
        )

    def test_bandwidth_range(
        self,
        start: int,
        end: int,
        increment: int,
        kind: MemcpyKind,
        printmode: PrintMode,
        mem_mode: MemoryMode,
        start_device: int,
        end_device: int,
        wc: bool,
    ):
        """范围模式带宽测试"""
        # 计算测试数量
        count = 1 + ((end - start) // increment)
        mem_sizes = [start + i * increment for i in range(count)]
        bandwidths = [0.0] * count

        # 在每个设备上运行测试
        for current_device in range(start_device, end_device + 1):
            rt.cudaSetDevice(current_device)

            for i, size in enumerate(mem_sizes):
                if kind == MemcpyKind.DEVICE_TO_HOST:
                    bandwidths[i] += self.test_device_to_host_transfer(size, mem_mode, wc)
                elif kind == MemcpyKind.HOST_TO_DEVICE:
                    bandwidths[i] += self.test_host_to_device_transfer(size, mem_mode, wc)
                elif kind == MemcpyKind.DEVICE_TO_DEVICE:
                    bandwidths[i] += self.test_device_to_device_transfer(size)

        # 打印结果
        if printmode == PrintMode.CSV:
            self.print_results_csv(mem_sizes, bandwidths, count, kind, mem_mode,
                                1 + end_device - start_device, wc)
        else:
            self.print_results_readable(mem_sizes, bandwidths, count, kind, mem_mode,
                                     1 + end_device - start_device, wc)

    def test_bandwidth_shmoo(
        self,
        kind: MemcpyKind,
        printmode: PrintMode,
        mem_mode: MemoryMode,
        start_device: int,
        end_device: int,
        wc: bool,
    ):
        """Shmoo模式带宽测试"""
        mem_sizes = []
        current_size = SHMOO_MEMSIZE_START

        # 生成测试大小序列
        while current_size <= SHMOO_MEMSIZE_MAX:
            mem_sizes.append(current_size)
            if current_size < SHMOO_LIMIT_20KB:
                current_size += SHMOO_INCREMENT_1KB
            elif current_size < SHMOO_LIMIT_50KB:
                current_size += SHMOO_INCREMENT_2KB
            elif current_size < SHMOO_LIMIT_100KB:
                current_size += SHMOO_INCREMENT_10KB
            elif current_size < SHMOO_LIMIT_1MB:
                current_size += SHMOO_INCREMENT_100KB
            elif current_size < SHMOO_LIMIT_16MB:
                current_size += SHMOO_INCREMENT_1MB
            elif current_size < SHMOO_LIMIT_32MB:
                current_size += SHMOO_INCREMENT_2MB
            else:
                current_size += SHMOO_INCREMENT_4MB

        count = len(mem_sizes)
        bandwidths = [0.0] * count

        # 在每个设备上运行测试
        for current_device in range(start_device, end_device + 1):
            rt.cudaSetDevice(current_device)

            for i, size in enumerate(mem_sizes):
                if kind == MemcpyKind.DEVICE_TO_HOST:
                    bandwidths[i] += self.test_device_to_host_transfer(size, mem_mode, wc)
                elif kind == MemcpyKind.HOST_TO_DEVICE:
                    bandwidths[i] += self.test_host_to_device_transfer(size, mem_mode, wc)
                elif kind == MemcpyKind.DEVICE_TO_DEVICE:
                    bandwidths[i] += self.test_device_to_device_transfer(size)
                print(".", end="", flush=True)

        # 打印结果
        print()
        if printmode == PrintMode.CSV:
            self.print_results_csv(mem_sizes, bandwidths, count, kind, mem_mode,
                                1 + end_device - start_device, wc)
        else:
            self.print_results_readable(mem_sizes, bandwidths, count, kind, mem_mode,
                                     1 + end_device - start_device, wc)

    def test_bandwidth(
        self,
        start: int,
        end: int,
        increment: int,
        mode: TestMode,
        kind: MemcpyKind,
        printmode: PrintMode,
        mem_mode: MemoryMode,
        start_device: int,
        end_device: int,
        wc: bool,
    ):
        """运行带宽测试"""
        if mode == TestMode.QUICK_MODE:
            self.test_bandwidth_quick(DEFAULT_SIZE, kind, printmode, mem_mode,
                                    start_device, end_device, wc)
        elif mode == TestMode.RANGE_MODE:
            self.test_bandwidth_range(start, end, increment, kind, printmode, mem_mode,
                                    start_device, end_device, wc)
        elif mode == TestMode.SHMOO_MODE:
            self.test_bandwidth_shmoo(kind, printmode, mem_mode, start_device, end_device, wc)

    def print_results_readable(
        self,
        mem_sizes,
        bandwidths,
        count: int,
        kind: MemcpyKind,
        mem_mode: MemoryMode,
        num_devs: int,
        wc: bool,
    ):
        """以可读格式打印结果"""
        print(f" {sMemoryCopyKind[kind.value]} Bandwidth, {num_devs} Device(s)")
        print(f" {sMemoryMode[mem_mode.value]} Memory Transfers")

        if wc:
            print(" Write-Combined Memory Writes are Enabled")

        print("   Transfer Size (Bytes)\tBandwidth(GB/s)")

        for i in range(count):
            tab = "\t" if mem_sizes[i] < 10000 else ""
            print(f"   {mem_sizes[i]}\t\t\t{tab}{bandwidths[i]:.1f}")

        print()

    def print_results_csv(
        self,
        mem_sizes,
        bandwidths,
        count: int,
        kind: MemcpyKind,
        mem_mode: MemoryMode,
        num_devs: int,
        wc: bool,
    ):
        """以CSV格式打印结果"""
        # 构建配置字符串
        if kind == MemcpyKind.DEVICE_TO_DEVICE:
            s_config = "D2D"
        else:
            if kind == MemcpyKind.DEVICE_TO_HOST:
                s_config = "D2H"
            elif kind == MemcpyKind.HOST_TO_DEVICE:
                s_config = "H2D"

            if mem_mode == MemoryMode.PAGEABLE:
                s_config += "-Paged"
            elif mem_mode == MemoryMode.PINNED:
                s_config += "-Pinned"
                if wc:
                    s_config += "-WriteCombined"

        # 打印结果
        for i in range(count):
            d_seconds = mem_sizes[i] / (bandwidths[i] * 1e9)
            print(
                f"bandwidthTest-{s_config}, Bandwidth = {bandwidths[i]:.1f} GB/s, "
                f"Time = {d_seconds:.5f} s, Size = {mem_sizes[i]} bytes, "
                f"NumDevsUsed = {num_devs}"
            )


def main():
    """主函数"""
    print("CUDA Bandwidth Test - Starting...")

    parser = argparse.ArgumentParser(description="CUDA Bandwidth Test", add_help=False)
    parser.add_argument("--help", action="store_true", help="Display this help menu")
    parser.add_argument("--csv", action="store_true", help="Print results as a CSV")
    parser.add_argument("--memory", type=str, choices=["pageable", "pinned"], help="Memory mode")
    parser.add_argument("--device", type=str, help="Device to use")
    parser.add_argument("--mode", type=str, choices=["quick", "range", "shmoo"], help="Test mode")
    parser.add_argument("--htod", action="store_true", help="Host to device transfers")
    parser.add_argument("--dtoh", action="store_true", help="Device to host transfers")
    parser.add_argument("--dtod", action="store_true", help="Device to device transfers")
    parser.add_argument("--wc", action="store_true", help="Write-combined memory")
    parser.add_argument("--cputiming", action="store_true", help="CPU-based timing")
    parser.add_argument("--start", type=int, help="Starting size in bytes")
    parser.add_argument("--end", type=int, help="Ending size in bytes")
    parser.add_argument("--increment", type=int, help="Increment size in bytes")

    args = parser.parse_args()

    if args.help:
        print_help()
        return 0

    result = run_test(args)

    if result == 0:
        print("Result = PASS")
    else:
        print("Result = FAIL")

    print("\nNOTE: The CUDA Samples are not meant for performance measurements. "
          "Results may vary when GPU Boost is enabled.")

    return result


def run_test(args):
    """运行测试主函数"""
    global bDontUseGPUTiming, flush_buf

    # 参数解析
    printmode = PrintMode.CSV if args.csv else PrintMode.USER_READABLE
    mem_mode = MemoryMode.PAGEABLE if args.memory == "pageable" else MemoryMode.PINNED

    # 设备设置
    tester = CUDABandwidthTester()

    if args.device:
        if args.device == "all":
            start_device, end_device = 0, tester.device_count - 1
        else:
            device_id = int(args.device)
            if 0 <= device_id < tester.device_count:
                start_device = end_device = device_id
            else:
                print(f"Invalid GPU number {device_id}, using device 0")
                start_device = end_device = 0
    else:
        start_device = end_device = 0

    # 打印设备信息
    tester.print_device_info(start_device, end_device)

    # 模式设置
    mode = TestMode(args.mode) if args.mode else TestMode.QUICK_MODE

    # 传输方向
    htod = args.htod or (not args.htod and not args.dtoh and not args.dtod)
    dtoh = args.dtoh or (not args.htod and not args.dtoh and not args.dtod)
    dtod = args.dtod or (not args.htod and not args.dtoh and not args.dtod)

    # 全局设置
    bDontUseGPUTiming = args.cputiming

    # 分配刷新缓冲区
    global flush_buf
    try:
        flush_buf = np.empty(FLUSH_SIZE, dtype=np.uint8)
    except:
        flush_buf = None

    # 运行测试
    start = args.start or DEFAULT_SIZE
    end = args.end or DEFAULT_SIZE
    increment = args.increment or DEFAULT_INCREMENT

    if htod:
        tester.test_bandwidth(start, end, increment, mode, MemcpyKind.HOST_TO_DEVICE,
                            printmode, mem_mode, start_device, end_device, args.wc)

    if dtoh:
        tester.test_bandwidth(start, end, increment, mode, MemcpyKind.DEVICE_TO_HOST,
                            printmode, mem_mode, start_device, end_device, args.wc)

    if dtod:
        tester.test_bandwidth(start, end, increment, mode, MemcpyKind.DEVICE_TO_DEVICE,
                            printmode, mem_mode, start_device, end_device, args.wc)

    # 重置所有设备
    for nDevice in range(start_device, end_device + 1):
        try:
            rt.setDevice(nDevice)
        except:
            pass

    return 0


def print_help():
    """打印帮助信息"""
    help_text = """
Usage: bandwidthTest [OPTION]...
Test the bandwidth for device to host, host to device, and device to device transfers

Options:
--help\tDisplay this help menu
--csv\tPrint results as a CSV
--device=[deviceno]\tSpecify the device to be used
  all - compute cumulative bandwidth on all the devices
  0,1,2,...,n - Specify any particular device to be used
--memory=[MEMMODE]\tSpecify which memory mode to use
  pageable - pageable memory
  pinned   - non-pageable system memory
--mode=[MODE]\tSpecify the mode to use
  quick - performs a quick measurement
  range - measures a user-specified range of values
  shmoo - performs an intense shmoo of a large range of values
--htod\tMeasure host to device transfers
--dtoh\tMeasure device to host transfers
--dtod\tMeasure device to device transfers
--wc\tAllocate pinned memory as write-combined
--cputiming\tForce CPU-based timing always

Range mode options
--start=[SIZE]\tStarting transfer size in bytes
--end=[SIZE]\tEnding transfer size in bytes
--increment=[SIZE]\tIncrement size in bytes
"""
    print(help_text)


if __name__ == "__main__":
    sys.exit(main())