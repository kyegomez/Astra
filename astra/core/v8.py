# !pip install cloud-tpu-client==0.10 torch==2.0.0 torchvision==0.15.1 https://storage.googleapis.com/tpu-pytorch/wheels/colab/torch_xla-2.0-cp310-cp310-linux_x86_64.whl
# !git clone https://github.com/NVIDIA/apex
# %cd apex
# # if pip >= 23.1 (ref: https://pip.pypa.io/en/stable/news/#v23-1) which supports multiple `--config-settings` with the same key... 
# !pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

import torch

# Check for available backends
try:
    import torch_xla.core.xla_model as xm
    _XLA_AVAILABLE = True
except ImportError:
    _XLA_AVAILABLE = False

try:
    import jax
    from jax import jit, grad
    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False

try:
    import triton
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False

try:
    from apex import amp
    _APEX_AVAILABLE = True
except ImportError:
    _APEX_AVAILABLE = False

def astra(mixed_precision=False, backend="auto"):
    def decorator(original_func):
        def wrapper(*args, **kwargs):
            if backend == "jax" and not _JAX_AVAILABLE:
                raise ImportError("JAX backend selected but JAX is not installed.")
            if backend == "triton" and not _TRITON_AVAILABLE:
                raise ImportError("Triton backend selected but Triton is not installed.")
            
            # Check device availability and set device based on backend
            if not _XLA_AVAILABLE and not torch.cuda.is_available():
                raise RuntimeError("Neither XLA (TPU support) nor CUDA (GPU support) is available. Ensure one of them is installed.")
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # If backend is set to auto, determine the best backend
            if backend == "auto":
                if _XLA_AVAILABLE and xm.xla_device().type == "xla":
                    device = xm.xla_device()

            # Handle tensor precision based on device
            dtype = torch.float32
            if device.type == "cuda" and mixed_precision:
                if not _APEX_AVAILABLE:
                    raise ImportError("Mixed precision requested but Apex is not installed. Please install Apex for mixed precision support.")
                dtype = torch.float16
            elif device.type == "xla":
                dtype = torch.bfloat16

            args = [arg.to(device=device, dtype=dtype) if torch.is_tensor(arg) else arg for arg in args]
            
            # Convert PyTorch tensors to numpy for JAX
            if backend == "jax" and _JAX_AVAILABLE:
                args = [arg.detach().cpu().numpy() if isinstance(arg, torch.Tensor) else arg for arg in args]
                kwargs = {k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}

            result = original_func(*args, **kwargs)

            # If the result is a numpy array (from JAX), convert it back to a PyTorch tensor
            if backend == "jax" and _JAX_AVAILABLE and isinstance(result, np.ndarray):
                result = torch.tensor(result, device=device, dtype=dtype)

            # If using XLA, mark the step for synchronization
            if device.type == "xla" and _XLA_AVAILABLE:
                xm.mark_step()

            return result

        return wrapper
    return decorator

# Sample Usage
data = torch.randn(2, 3)

@astra(mixed_precision=True, backend="triton")
def forward(x):
    return x * 2  # Just doubling the data for demonstration

result = forward(data)
print(result)
