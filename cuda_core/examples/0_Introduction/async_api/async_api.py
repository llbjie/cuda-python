import cupy as cp
from cuda.core.experimental import Device, LaunchConfig, Program, ProgramOptions, launch

def convertSmVerToArchName(major, minor) -> str:
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

    major = int(major)
    minor = int(minor)
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

import sys
import time
if __name__ == "__main__":

    print(f"[{sys.argv[0]}] - Starting...\n")

    dev = Device()
    dev.set_current()
    s = dev.create_stream()

    arch = float(dev.arch)
    print(f"GPU Device {dev._id}: {convertSmVerToArchName(arch // 10, arch % 10)} with compute capability {arch}")
    print(f"CUDA device  [{dev.name}]")

    prog_options = ProgramOptions(std="c++17", arch=f"sm_{dev.arch}")
    prog = Program(INCREMENT_KERNEL, code_type="c++", options=prog_options)
    mod = prog.compile("cubin")
    kernel = mod.get_kernel("increment_kernel")

    dtype = cp.int32
    n = 16 << 20 
    data = cp.zeros(n, dtype=dtype)
    dev.sync()

    block_size = 512
    grid_size = n // block_size
    config = LaunchConfig(grid=grid_size, block=block_size)

    cpu_start = time.perf_counter()
    gpu_time = 0.0

    e1 = dev.create_event({"enable_timing": True})
    e2 = dev.create_event({"enable_timing": True})
    s.record(e1)

    launch(s, config, kernel, data.data.ptr, 26)
    cpu_end = time.perf_counter()

    s.record(e2)
    e2.sync()
    s.sync()


    print(f"time spent executing by the GPU: {(e2 - e1):.2f} ms")
    print(f"CPU CUDA call time: {(cpu_end - cpu_start) * 1000:.2f} ms")

    assert cp.allclose(data, cp.full(n, 26))