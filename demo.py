import os
import torch

print(f"this is rank {os.environ['RANK']}")
num_gpus = torch.cuda.device_count()
print("Number of available GPUs: ", num_gpus)