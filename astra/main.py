import torch

# Check for available backends
try:
    import torch_xla.core.xla_model as xm
    _XLA_AVAILABLE = True
except ImportError:
    _XLA_AVAILABLE = False

# Check for Apex availability for mixed precision
try:
    from apex import amp
    _APEX_AVAILABLE = True
except ImportError:
    _APEX_AVAILABLE = False

def astra(mixed_precision=False):
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
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not _XLA_AVAILABLE and not torch.cuda.is_available():
                raise RuntimeError("Neither XLA (TPU support) nor CUDA (GPU support) is available. Ensure one of them is installed.")
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Check for TPU
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
                
            args = [arg.to(device=device, dtype=dtype) for arg in args if torch.is_tensor(arg)]

            # Run the function on the determined device and precision
            result = func(*args, **kwargs)

            if device.type == "xla":
                if not _XLA_AVAILABLE:
                    raise ImportError("XLA operations requested but torch_xla is not installed. Please install torch_xla for TPU support.")
                xm.mark_step()

            return result

        return wrapper
    return decorator

