!pip install torch torchvision
!pip install cloud-tpu-client==0.10 torch==2.0.0 torchvision==0.15.1 https://storage.googleapis.com/tpu-pytorch/wheels/colab/torch_xla-2.0-cp310-cp310-linux_x86_64.whl
import torch

try:
    import torch_xla.core.xla_model as xm
    _XLA_AVAILABLE = True
except ImportError:
    _XLA_AVAILABLE = False

def astra(func):
    """
    Astra wrapper to enhance PyTorch functions using the optimal backend (CPU, GPU, or XLA for TPU).
    """
    def wrapper(*args, **kwargs):
        # Detect device
        device = "tpu" if _XLA_AVAILABLE and xm.xla_device().type == "xla" else (
            "gpu" if torch.cuda.is_available() else "cpu"
        )

        # Convert input tensors based on the device
        if device == "tpu":
            backend_args = [xm._convert_to_xla_tensor(arg) if torch.is_tensor(arg) else arg for arg in args]
        else:
            backend_args = args

        # Execute the original function with tensors compatible with the chosen backend
        result = func(*backend_args, **kwargs)

        # Convert the result back to a PyTorch tensor if needed
        if device == "tpu":
            result = xm._convert_to_cpu_tensor(result)

        return result

    return wrapper

# Sample Usage

from torch import nn

data = torch.randn(2, 3)    

@astra  # Potential boost in performance and speed.
def forward(x):
    softmax = nn.Softmax(dim=1)
    result = softmax(x)
    return result

result = forward(data)
print(result)
