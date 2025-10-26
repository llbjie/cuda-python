import cupy as cp
from cuda.core.experimental import Device, LaunchConfig, Program, ProgramOptions, launch

if __name__ == "__main__":
    device = Device()
    device.set_current()
    s = device.create_stream()

    path = "./c++11_cuda.ptx"
    with open(path, 'r') as f:
        code = f.read()
    
    prog = Program(code, code_type="ptx")
    mod = prog.compile("cubin")
    kernel = mod.get_kernel("xyzw_frequency")
    
    file_path = "./warandpeace.txt"
    with open(file_path, "r") as f:
        text = f.read()
  
    count = cp.zeros(1, dtype=cp.uint32) 
    text_bytes = text.encode("utf-8")
    n = len(text_bytes)  
    text = cp.frombuffer(text_bytes, dtype=cp.uint8)
    device.sync()

    config = LaunchConfig(grid=8, block=256)
    
    ker_args = (count.data.ptr, text.data.ptr, n)
    launch(s, config, kernel, *ker_args)
    s.sync()

    result = count[0]  
    print(f"counted {result} instances of 'x', 'y', 'z', or 'w' in \"{file_path}\"")