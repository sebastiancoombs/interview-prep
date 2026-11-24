"""Exercise 4: make one optimizer step portable across CPU and CUDA."""

import torch
from torch import nn


device = torch.device("cuda")
model = nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 2)).to(device)
features = torch.randn(4, 3)
targets = torch.tensor([0, 1, 1, 0])

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
optimizer.zero_grad()
logits = model(features)
loss = nn.CrossEntropyLoss()(logits, targets)
loss.backward()
optimizer.step()

print("device:", device)
print("loss:", loss.item())
