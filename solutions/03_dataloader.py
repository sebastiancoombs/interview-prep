import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


class TokenDataset(Dataset):
    rows = [
        {"tokens": [2, 4, 1], "label": 1},
        {"tokens": [3, 1], "label": 0},
        {"tokens": [8, 7, 6, 2], "label": 1},
    ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return torch.tensor(row["tokens"]), torch.tensor(row["label"], dtype=torch.long)


def collate(rows):
    tokens, labels = zip(*rows)
    return {
        "tokens": pad_sequence(tokens, batch_first=True, padding_value=0),
        "label": torch.stack(labels),
    }


batch = next(iter(DataLoader(TokenDataset(), batch_size=3, collate_fn=collate)))
embedding = nn.Embedding(10, 5, padding_idx=0)
classifier = nn.Linear(5, 2)
logits = classifier(embedding(batch["tokens"]).mean(dim=1))
loss = nn.CrossEntropyLoss()(logits, batch["label"])
assert batch["tokens"].shape == (3, 4)
assert batch["label"].dtype == torch.int64
print("token shape:", tuple(batch["tokens"].shape))
print("label shape/dtype:", tuple(batch["label"].shape), batch["label"].dtype)
print("loss:", loss.item())
