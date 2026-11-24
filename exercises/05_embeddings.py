"""Exercise 5: repair vocabulary, padding, and unknown-token handling."""

import torch
from torch import nn


vocab = {"<PAD>": 0, "red": 1, "green": 2, "blue": 3}
unknown_id = len(vocab)


def encode(text: str, length: int) -> list[int]:
    token_ids = [vocab.get(token, unknown_id) for token in text.split()]
    return (token_ids + [vocab["<PAD>"]] * length)[:length]


batch = torch.tensor(
    [
        encode("red blue", length=3),
        encode("green purple", length=3),
    ]
)

embedding = nn.Embedding(num_embeddings=len(vocab), embedding_dim=4, padding_idx=0)
output = embedding(batch)

assert output.shape == (2, 3, 4)
print("encoded batch:\n", batch)
print("embedding shape:", tuple(output.shape))
