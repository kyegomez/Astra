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

def astra(mixed_precision=False, backend="auto"):
    def decorator(original_func):
        def wrapper(*args, **kwargs):
            if backend == "jax" and not _JAX_AVAILABLE:
                raise ImportError("JAX backend selected but JAX is not installed.")
            if backend == "triton" and not _TRITON_AVAILABLE:
                raise ImportError("Triton backend selected but Triton is not installed.")

            # Convert PyTorch tensors to numpy for JAX
            if backend == "jax" and _JAX_AVAILABLE:
                args = tuple(arg.detach().cpu().numpy() if isinstance(arg, torch.Tensor) else arg for arg in args)
                kwargs = {k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}

            result = original_func(*args, **kwargs)

            # If the result is a numpy array (from JAX), convert it back to a PyTorch tensor
            if backend == "jax" and _JAX_AVAILABLE and isinstance(result, np.ndarray):
                result = torch.tensor(result)

            return result

        return wrapper
    return decorator

# Sample Usage
from torch import nn

data = torch.randn(2, 3)

@astra(mixed_precision=True, backend="xla")
def forward(x):
    # This is just an example and won't work directly with nn.Softmax
    # You'd typically have pure functions here for JAX
    return x * 2  # Just doubling the data for demonstration

result = forward(data)
print(result)
