import torch
from torch import nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 2)).to(device)
features = torch.randn(4, 3, device=device)
targets = torch.tensor([0, 1, 1, 0], device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
optimizer.zero_grad()
loss = nn.CrossEntropyLoss()(model(features), targets)
loss.backward()
optimizer.step()
print("device:", device)
print("loss:", loss.item())
