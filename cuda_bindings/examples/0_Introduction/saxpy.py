from cuda.bindings import driver, nvrtc
import numpy as np

def _cudaGetErrorEnum(error):
    if isinstance(error, driver.CUresult):
        err, name = driver.cuGetErrorName(error)
        return name if err == driver.CUresult.CUDA_SUCCESS else "<unknown>"
    elif isinstance(error, nvrtc.nvrtcResult):
        return nvrtc.nvrtcGetErrorString(error)[1]
    else:
        raise RuntimeError('Unknown error type: {}'.format(error))

def checkCudaErrors(result):
    if result[0].value:
        raise RuntimeError("CUDA error code={}({})".format(result[0].value, _cudaGetErrorEnum(result[0])))
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]
    
saxpy = """\
extern "C" __global__
void saxpy(float a, float *x, float *y, float *out, size_t n)
{
 size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
 if (tid < n) {
   out[tid] = a * x[tid] + y[tid];
 }
}
"""

def main():
  # Initialize CUDA Driver API
  checkCudaErrors(driver.cuInit(0))

  # Retrieve handle for device 0
  cuDevice = checkCudaErrors(driver.cuDeviceGet(0))

  # Derive target architecture for device 0
  major = checkCudaErrors(driver.cuDeviceGetAttribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice))
  minor = checkCudaErrors(driver.cuDeviceGetAttribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice))
  arch_arg = bytes(f'--gpu-architecture=compute_{major}{minor}', 'ascii')
  # arch_arg = bytes(f'--gpu-architecture=compute_{major}{minor}', 'ascii')

  # Create program
  prog = checkCudaErrors(nvrtc.nvrtcCreateProgram(str.encode(saxpy), b"saxpy.cu", 0, [], []))
  # prog = checkCudaErrors(nvrtc.nvrtcCreateProgram(str.encode(saxpy), b"saxpy.cu", 0, [], []))

  # Compile program
  opts = [b'--fmad=false', arch_arg]
  checkCudaErrors(nvrtc.nvrtcCompileProgram(prog, 2, opts))

  # Get PTX from compilation
  ptxSize = checkCudaErrors(nvrtc.nvrtcGetPTXSize(prog))
  ptx = b" " * ptxSize
  checkCudaErrors(nvrtc.nvrtcGetPTX(prog, ptx))

  # Create context
  context = checkCudaErrors(driver.cuCtxCreate(None, 0, cuDevice))
  # Create context
  # context = checkCudaErrors(driver.cuCtxCreate(0, cuDevice))
  
  # Load PTX as module data and retrieve function
  ptx = np.char.array(ptx)
  module = checkCudaErrors(driver.cuModuleLoadData(ptx.ctypes.data))
  kernel = checkCudaErrors(driver.cuModuleGetFunction(module, b'saxpy'))

  # Next, get all your data prepared and transferred to the GPU.
  NUM_THREADS = 512
  NUM_BLOCKS = 32768

  a = np.array([2.0],dtype=np.float32)
  n = np.array(NUM_THREADS * NUM_BLOCKS, dtype=np.uint32)
  bufferSize = n * a.itemsize

  hX = np.random.rand(n).astype(np.float32)
  hY = np.random.rand(n).astype(np.float32)
  hOut = np.zeros(n).astype(np.float32)

  # transfer data to device 
  dXclass = checkCudaErrors(driver.cuMemAlloc(bufferSize))
  dYclass = checkCudaErrors(driver.cuMemAlloc(bufferSize)) 
  dOutclass = checkCudaErrors(driver.cuMemAlloc(bufferSize))
  
  stream = checkCudaErrors(driver.cuStreamCreate(0))
  checkCudaErrors(driver.cuMemcpyHtoDAsync(dXclass, hX.ctypes.data, bufferSize, stream))
  checkCudaErrors(driver.cuMemcpyHtoDAsync(dYclass, hY.ctypes.data, bufferSize, stream))

  # retrieve device pointer
  dX = np.array([int(dXclass)], dtype=np.uint64)
  dY = np.array([int(dYclass)], dtype=np.uint64)
  dOut = np.array([int(dOutclass)], dtype=np.uint64)

  # prepare kernel arguments
  args = [a, dX, dY, dOut, n]
  args = np.array([arg.ctypes.data for arg in args], dtype=np.uint64)

  # launch kernel
  checkCudaErrors(driver.cuLaunchKernel(
      kernel,
      NUM_BLOCKS,
      1,
      1,
      NUM_THREADS,
      1,
      1,
      0,
      stream,
      args.ctypes.data,
      0,
  ))

  checkCudaErrors(driver.cuMemcpyDtoHAsync(hOut.ctypes.data, dOutclass, bufferSize, stream))
  checkCudaErrors(driver.cuStreamSynchronize(stream))

  # Assert values are same after running kernel
  hZ = a * hX + hY
  if not np.allclose(hOut, hZ):
   raise ValueError("Error outside tolerance for host-device vectors")
  
  checkCudaErrors(driver.cuStreamDestroy(stream))
  checkCudaErrors(driver.cuMemFree(dXclass))
  checkCudaErrors(driver.cuMemFree(dYclass))
  checkCudaErrors(driver.cuMemFree(dOutclass))
  checkCudaErrors(driver.cuModuleUnload(module))
  checkCudaErrors(driver.cuCtxDestroy(context))

if __name__ == "__main__":
    main()