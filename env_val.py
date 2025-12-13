import torch

print(f'CUDA: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0)} | Cap: {torch.cuda.get_device_capability(0)}')