# Sample Usage
import torch
from astra.main import astra 
from torch import nn

data = torch.randn(2, 3)

@astra(mixed_precision=True)
def forward(x):
    softmax = nn.Softmax(dim=1).to(x.device)
    result = softmax(x)
    return result

result = forward(data)
print(result)
