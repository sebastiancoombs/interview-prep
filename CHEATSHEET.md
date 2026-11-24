# Debugging Cheatsheet

Use this for preparation. If you want it open during the interview, first
confirm that personal notes are allowed under their open-book policy.

## First response to a failure

```text
1. Reproduce with the smallest command.
2. Read the final exception line.
3. Find the first traceback frame in code I control.
4. Inspect the relevant value and the API contract.
5. Change one thing.
6. Rerun and test silent correctness conditions.
```

## Python debugger

```python
breakpoint()
```

Common `pdb` commands:

- `n`: next line
- `s`: step into
- `c`: continue
- `w`: stack trace
- `u` / `d`: move up/down stack frames
- `l`: source around current line
- `p expression`: inspect a value
- `pp expression`: pretty-print a value
- `q`: quit

Launch without changing source:

```bash
python -m pdb script.py
python -m pdb -c continue script.py
```

## Inspect a tensor

```python
print(type(x))
print(x.shape, x.dtype, x.device)
print(x.min().item(), x.max().item())
print(torch.isnan(x).any().item())
```

## Frequent PyTorch contracts

- `nn.Embedding(n, d)` accepts integer IDs in `[0, n - 1]`.
- `CrossEntropyLoss` expects raw logits shaped `(N, C)` and class targets
  shaped `(N,)` with dtype `torch.long`.
- Do not apply softmax before `CrossEntropyLoss`.
- Model parameters and every tensor used in an operation must share a device.
- Training iteration: `zero_grad()` → forward → loss → `backward()` → `step()`.
- Evaluation: `model.eval()` plus `with torch.no_grad():`.
- Variable-length sequences need padding or a custom `collate_fn` before they
  can be stacked into a batch.

## Portable device selection

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
inputs = inputs.to(device)
targets = targets.to(device)
```

## JSON and CSV

JSON Lines/JSONL contains one JSON object per line:

```python
with open(path) as handle:
    rows = [json.loads(line) for line in handle if line.strip()]
```

CSV values start as strings, including IDs and missing values. Choose a
consistent join-key type and define how blanks should be represented.

## Split correctness

Check more than lengths:

```python
assert train_ids.isdisjoint(validation_ids)
assert len(train_ids | validation_ids) == len(all_ids)
```

Use a local seeded RNG when reproducibility matters:

```python
rng = random.Random(7)
rng.shuffle(records)
```

## Useful search phrases

When Google/Stack Overflow is allowed, search the exact error plus the smallest
relevant API name, for example:

```text
PyTorch stack expects each tensor to be equal size DataLoader
CrossEntropyLoss expected scalar type Long but found Float
PyTorch Expected all tensors to be on the same device
nn.Embedding index out of range
```
