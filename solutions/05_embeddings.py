import torch
from torch import nn


vocab = {"<PAD>": 0, "<UNK>": 1, "red": 2, "green": 3, "blue": 4}


def encode(text: str, length: int) -> list[int]:
    ids = [vocab.get(token, vocab["<UNK>"]) for token in text.split()]
    return (ids + [vocab["<PAD>"]] * length)[:length]


batch = torch.tensor([encode("red blue", 3), encode("green purple", 3)])
embedding = nn.Embedding(len(vocab), 4, padding_idx=vocab["<PAD>"])
output = embedding(batch)
assert output.shape == (2, 3, 4)
print("encoded batch:\n", batch)
print("embedding shape:", tuple(output.shape))
