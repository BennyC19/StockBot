import torch
import torch.nn as nn
import torch.optim as optim
from transformer_colab import Transformer

transformer = Transformer().to('cpu')

total_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)

print(total_params)