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

  kernel_path = sdkFindFilePath("c++11_cuda.cu", sys.argv[0])
  if kernel_path is None:
      print("The file 'c++11_cuda.cu' could not be found in the executable directory")
      sys.exit(1)
  with open(kernel_path, 'r') as f:
      kernel = f.read() 

  kernel_helper = common.KernelHelper(kernel, dev_id)
  kernel_addr_1 = common.kernelHelper.getFunction(b"xyzw_frequency")
  # kernel_addr_2 = common.kernelHelper.getFunction(b"xyzw_frequency_thrust_device")

  checkCudaErrors(
      driver.cuLaunchKernel(
          kernel_addr_1,
          256,
          1,
          1,
          256,
          1,
          1,
          0,
          0,
          (d_text, d_count),
          0,
      )
  )
  print("counted {count} instances of 'x', 'y', 'z', or 'w' in ")

  checkCudaErrors(driver.cudaFree(d_text))
