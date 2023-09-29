import torch
import torch_xla
import torch_xla.core.xla_model as xm

def astra(mixed_precision=False):
    def decorator(func):
        def wrapper(*args, **kwargs):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Check for TPU
            if xm.xla_device().type == "xla":
                device = xm.xla_device()

            # Handle tensor precision based on device
            dtype = torch.float32
            if device.type == "cuda" and mixed_precision:
                dtype = torch.float16
            elif device.type == "xla":
                dtype = torch.bfloat16
                
            args = [arg.to(device=device, dtype=dtype) for arg in args if torch.is_tensor(arg)]

            # Run the function on the determined device and precision
            result = func(*args, **kwargs)

            if device.type == "xla":
                xm.mark_step()

            return result

        return wrapper
    return decorator

# Sample Usage
from torch import nn

data = torch.randn(2, 3)

@astra(mixed_precision=True)
def forward(x):
    softmax = nn.Softmax(dim=1).to(x.device)
    result = softmax(x)
    return result

result = forward(data)
print(result)
