from common.helper_string import sdkFindFilePath
import sys
from common.helper_cuda import findCudaDevice,checkCudaErrors
from cuda.bindings import driver, runtime
from common import common
import numpy as np

if __name__ == "__main__":

    path = sdkFindFilePath("warandpeace.txt", sys.argv[0])  
    if path is  None:  
        print("The file 'warandpeace.txt' could not be found in the executable directory")

    num_bytes = 16 * 1048576  # 16 MB
    with open(path, 'r') as f:
        data = f.read(num_bytes)
        print(f"Read {len(data)} bytes from file {path}") 

    # h_data = np.frombuffer(data.bytes, dtype=np.uint8)
    h_data = np.frombuffer(data.encode('utf-8'), dtype=np.uint8)

    dev_id = findCudaDevice()
    d_text = checkCudaErrors(driver.cuMemAlloc(len(data)))

    checkCudaErrors(driver.cuMemcpyHtoD(d_text, h_data.ctypes.data, len(data)))

    d_count = checkCudaErrors(driver.cuMemAlloc(4))
    checkCudaErrors(driver.cuMemsetD32(d_count, 0, 1))

    kernel_path = sdkFindFilePath("c++11_cuda.ptx", sys.argv[0])
    with open(kernel_path, 'rb') as f:
        ptx_data = f.read()
    err, module = driver.cuModuleLoadData(ptx_data)
    err, kernel = driver.cuModuleGetFunction(module, "xyzw_frequency".encode())

    d_a = np.array([int(d_count)], dtype=np.uint64)
    d_b = np.array([int(d_text)], dtype=np.uint64)
    d_c = np.array([len(data)], dtype=np.uint64)

    args = [d_a, d_b, d_c]
    args = np.array([arg.ctypes.data for arg in args], dtype=np.uint64)

    checkCudaErrors(
        driver.cuLaunchKernel(
            kernel,
            8, 1, 1,  
            256, 1, 1, 
            0,         # 共享内存大小
            0,         # 流
            args.ctypes.data,  # kernel arguments
            0,         # 额外参数
        )
    )
    # 同步等待核函数完成
    checkCudaErrors(driver.cuCtxSynchronize())

    h_count = np.zeros(1, dtype=np.int32)
    checkCudaErrors(driver.cuMemcpyDtoH(h_count.ctypes.data, d_count, 4))
    count = h_count[0]
    print(f'counted {count} instances of \'x\', \'y\', \'z\', or \'w\' in "{path}"')

    checkCudaErrors(driver.cuMemFree(d_text))
    checkCudaErrors(driver.cuMemFree(d_count))