import numpy as np
import torch

if __name__ == "__main__":
  a = np.array([1, 2], dtype=np.float32)
  print(a.astype(torch.bfloat16))