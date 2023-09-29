import torch
import numpy as np

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
    """
    Astra works by wrapping a function with a decorator that determines the device and precision to run the function on.
    
    Parameters
    ----------
    mixed_precision : bool, optional
        Whether to use mixed precision. This is only available on CUDA devices.
    
    Returns
    -------
    decorator
        The decorator that will wrap the function.
    
    Examples
    --------
    >>> from torch import nn
    >>> data = torch.randn(2, 3)
    >>> @astra(mixed_precision=True)
    ... def forward(x):
    ...     softmax = nn.Softmax(dim=1).to(x.device)
    ...     result = softmax(x)
    ...     return result
    >>> result = forward(data)
    >>> print(result)

    Future optimizations, features
    ------------------------------
    - [ ] Kernel fusion
    - [ ] JIT compilation
    - [ ] Graph optimization

    
    """
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


