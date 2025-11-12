import numpy as np
import ctypes
from cuda_core import device, nvrtc, memory, module, stream

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
    # 1. 选择设备并创建上下文
    dev = device.Device(0)
    ctx = dev.create_context()

    # 2. 编译内核
    prog = nvrtc.Program(kernel_code, "oddEven.cu")
    arch_flag = f"--gpu-architecture=compute_{dev.compute_capability_major}{dev.compute_capability_minor}"
    prog.compile([arch_flag])
    ptx = prog.get_ptx()

    # 3. 加载模块和内核函数
    mod = module.Module(ptx)
    kernel = mod.get_function("oddEvenCountAndSumCG")

    # 4. 准备数据
    arr_size = 1024 * 100
    host_input = np.random.randint(0, 50, arr_size).astype(np.int32)
    host_num_of_odds = np.zeros(1, dtype=np.int32)
    host_sum_of_odd_even = np.zeros(2, dtype=np.int32)

    # 5. 分配设备内存
    d_input = memory.DeviceAllocation(host_input.nbytes)
    d_num_of_odds = memory.DeviceAllocation(host_num_of_odds.nbytes)
    d_sum_of_odd_even = memory.DeviceAllocation(host_sum_of_odd_even.nbytes)

    # 6. 创建流
    stream_obj = stream.Stream()

    # 7. 拷贝输入数据到设备，初始化计数器为0
    d_input.copy_from_host_async(host_input, stream=stream_obj)
    d_num_of_odds.memset_async(0, stream=stream_obj)
    d_sum_of_odd_even.memset_async(0, stream=stream_obj)

    # 8. 设置内核参数
    # 注意：cuda_core 的内核参数传入方式是按顺序传递python对象，会自动转换
    args = (d_input, d_num_of_odds, d_sum_of_odd_even, np.uint32(arr_size))

    # 9. 计算合适的线程块和线程数
    max_blocks, max_threads = kernel.get_occupancy_max_potential_block_size()

    # 10. 启动内核
    kernel.launch(
        grid_dim=(max_blocks, 1, 1),
        block_dim=(max_threads, 1, 1),
        args=args,
        stream=stream_obj
    )

    # 11. 拷贝结果回主机
    d_num_of_odds.copy_to_host_async(host_num_of_odds, stream=stream_obj)
    d_sum_of_odd_even.copy_to_host_async(host_sum_of_odd_even, stream=stream_obj)

    # 12. 等待流完成
    stream_obj.synchronize()

    print(f"Array size = {arr_size}")
    print(f"Number of odd elements = {host_num_of_odds[0]}")
    print(f"Sum of odd elements = {host_sum_of_odd_even[0]}")
    print(f"Sum of even elements = {host_sum_of_odd_even[1]}")

    # 13. 释放上下文（with 语句也可以自动释放）
    ctx.pop()

if __name__ == "__main__":
    main()
