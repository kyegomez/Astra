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

