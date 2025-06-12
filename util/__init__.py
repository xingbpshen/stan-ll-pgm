import torch
import warnings


def gpu_compute_capability():
    if torch.cuda.is_available():
        return torch.cuda.get_device_capability(0)
    else:
        warnings.warn("No CUDA-compatible GPU found.", UserWarning)
        return None
