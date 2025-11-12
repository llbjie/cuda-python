import cupy as cp
from cuda.core.experimental import Device, Program, ProgramOptions, LaunchConfig, launch
import sys

kernel_code = r"""
template<typename T>
__global__ void testKernel(T* d_odata, T* d_idata, int numElements) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int numThreads = blockDim.x * gridDim.x;
    for (int pos = tid; pos < numElements; pos += numThreads) {
        d_odata[pos] = d_idata[pos];
    }
}
"""

def run_test(dtype, packed_element_size, memory_size):
    dev = Device()
    dev.set_current()
    s = dev.create_stream()

    # 编译kernel
    opts = ProgramOptions(std="c++11", arch=f"sm_{dev.arch}")
    prog = Program(kernel_code, code_type="c++", options=opts)
    mod = prog.compile("cubin", logs=sys.stdout, name_expressions=(f"testKernel<{dtype.__name__}>",))

    ker = mod.get_kernel(f"testKernel<{dtype.__name__}>")

    # 分配输入输出cupy数组
    num_elements = memory_size // dtype().nbytes
    h_idata = cp.arange(memory_size, dtype=dtype) % 256
    d_idata = cp.array(h_idata)
    d_odata = cp.zeros_like(d_idata)

    # 配置launch
    block_size = 256
    grid_size = 64
    config = LaunchConfig(grid=grid_size, block=block_size)

    # 参数
    ker_args = (d_odata.data.ptr, d_idata.data.ptr, num_elements)

    # 启动kernel
    launch(s, config, ker, *ker_args)
    s.sync()

    # 验证
    h_odata = d_odata.get()
    if not (h_odata == h_idata.get()).all():
        print("Test failure")
        return False
    else:
        print("Test success")
        return True

# 主程序调用示例
if __name__ == "__main__":
    MEM_SIZE = 50_000_000
    run_test(cp.uint8, 1, MEM_SIZE)
    run_test(cp.uint16, 2, MEM_SIZE)
    # 其他类型需要自己定义相应cupy dtype结构体或使用numpy结构体映射，稍微复杂
