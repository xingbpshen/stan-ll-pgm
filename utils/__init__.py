import torch
import warnings


def get_gpu_compute_capability():
    if torch.cuda.is_available():
        return torch.cuda.get_device_capability(0)
    else:
        warnings.warn("No CUDA-compatible GPU found.", UserWarning)
        return None


def get_vllm_dtype():
    gpu_compute_capability = get_gpu_compute_capability()
    if gpu_compute_capability is not None:
        if gpu_compute_capability[0] >= 8:  # for GPUs with compute capability >= 8.0, can use bfloat16
            return "bfloat16"
        else:  # for older GPUs, must use float16
            return "half"


def info(file_name, msg):
    print(f"\033[1;94m[{file_name}]\033[0m \033[94mINFO\033[0m {msg}")
