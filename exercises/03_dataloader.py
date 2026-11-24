"""Exercise 3: batch variable-length inputs and compute one loss."""

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class TokenDataset(Dataset):
    def __init__(self) -> None:
        self.rows = [
            {"tokens": [2, 4, 1], "label": 1},
            {"tokens": [3, 1], "label": 0},
            {"tokens": [8, 7, 6, 2], "label": 1},
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        return {
            "tokens": torch.tensor(row["tokens"]),
            "label": torch.tensor(float(row["label"])),
        }


loader = DataLoader(TokenDataset(), batch_size=3)
batch = next(iter(loader))

embedding = nn.Embedding(num_embeddings=10, embedding_dim=5, padding_idx=0)
classifier = nn.Linear(5, 2)
features = embedding(batch["tokens"]).mean(dim=1)
logits = classifier(features)
loss = nn.CrossEntropyLoss()(logits, batch["label"])

print("token shape:", tuple(batch["tokens"].shape))
print("label shape/dtype:", tuple(batch["label"].shape), batch["label"].dtype)
print("loss:", loss.item())
