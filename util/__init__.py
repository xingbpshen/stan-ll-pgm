import torch
import warnings


def gpu_compute_capability():
    if torch.cuda.is_available():
        return torch.cuda.get_device_capability(0)
    else:
        warnings.warn("No CUDA-compatible GPU found.", UserWarning)
        return None


def info(file_name, msg):
    print(f"\033[1;94m[{file_name}]\033[0m \033[94mINFO\033[0m {msg}")
