from common.helper_string import sdkFindFilePath
from common.helper_cuda import findCudaDevice, checkCudaErrors
from cuda.bindings import driver
import numpy as np
import sys

if __name__ == "__main__":

    path = sdkFindFilePath("warandpeace.txt", sys.argv[0])
    if path is None:
        print(
            "The file 'warandpeace.txt' could not be found in the executable directory"
        )
        sys.exit(1)

    size = 16 * 1048576  # 16 MB
    with open(path, "r") as f:
        data = f.read(size)
        n = len(data)
        print(f"Read {n} bytes from file {path}")
    h_data = np.frombuffer(data.encode("utf-8"), dtype=np.uint8)

    dev = findCudaDevice()
    d_text = checkCudaErrors(driver.cuMemAlloc(n))
    checkCudaErrors(driver.cuMemcpyHtoD(d_text, h_data.ctypes.data, n))

    d_count = checkCudaErrors(driver.cuMemAlloc(4))
    checkCudaErrors(driver.cuMemsetD32(d_count, 0, 1))

    kernel_path = sdkFindFilePath("c++11_cuda.ptx", sys.argv[0])
    with open(kernel_path, "rb") as f:
        ptx = f.read()
    module = checkCudaErrors(driver.cuModuleLoadData(ptx))
    kernel = checkCudaErrors(
        driver.cuModuleGetFunction(module, "xyzw_frequency".encode())
    )

    args = [
        np.array([int(d_count)], dtype=np.uint64),
        np.array([int(d_text)], dtype=np.uint64),
        np.array([n], dtype=np.uint64),
    ]
    args = np.array([arg.ctypes.data for arg in args], dtype=np.uint64)
    checkCudaErrors(
        driver.cuLaunchKernel(
            kernel,
            8, 1, 1,
            256, 1, 1,
            0,
            0,
            args.ctypes.data,
            0,
        )
    )

    count = np.zeros(1, dtype=np.int32)
    checkCudaErrors(driver.cuMemcpyDtoH(count.ctypes.data, d_count, 4))
    print(f"counted {count[0]} instances of 'x', 'y', 'z', or 'w' in \"{path}\"")

    checkCudaErrors(driver.cuMemFree(d_text))
    checkCudaErrors(driver.cuMemFree(d_count))
