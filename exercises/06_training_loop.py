"""Exercise 6: repair API contracts and state in this training loop."""

import torch
from torch import nn


torch.manual_seed(7)
features = torch.tensor(
    [[-2.0, -1.0], [-1.0, -2.0], [-1.5, -0.5], [1.0, 2.0], [2.0, 1.0], [1.5, 0.5]]
)
targets = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.float32)

model = nn.Linear(2, 2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()

initial_loss = None
for _ in range(25):
    probabilities = torch.softmax(model(features), dim=1)
    loss = criterion(probabilities, targets)
    if initial_loss is None:
        initial_loss = loss.item()
    loss.backward()
    optimizer.step()

model.eval()
predictions = model(features).argmax(dim=1)
final_loss = criterion(model(features), targets).item()

assert initial_loss is not None
assert final_loss < initial_loss
print(f"initial loss: {initial_loss:.4f}")
print(f"final loss: {final_loss:.4f}")
print("predictions:", predictions.tolist())
