import torch
import torch_xla.core.xla_model as xm

def astra(func):
    """
    Astra wrapper to enhance PyTorch functions using XLA for better GPU performance.
    """
    def wrapper(*args, **kwargs):
        # Convert input tensors to XLA tensors
        xla_args = [xm._convert_to_xla_tensor(arg) if torch.is_tensor(arg) else arg for arg in args]

        # Execute the original function with XLA tensors
        xla_result = func(*xla_args, **kwargs)

        # Convert the XLA tensor back to a PyTorch tensor
        result = xm._convert_to_cpu_tensor(xla_result)
        
        return result
    return wrapper

# Sample Usage

from torch import nn

data = torch.randn(2, 3)    

@astra  # Potentially 100x+ boost in performance and speed.
def forward(x):
    softmax = nn.Softmax(dim=1)
    result = softmax(x)
    return result

result = forward(data)
print(result)
