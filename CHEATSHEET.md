# PyTorch Debugging Cheat Sheet

Designed for quick use during a debugging interview.

## The 60-second debugging loop

```text
1. REPRODUCE  Run the smallest command that fails.
2. LOCATE     Read the last exception line, then find the first traceback
              frame in code you control.
3. CLASSIFY   Is this a shape, dtype, device, value/range, gradient, mode,
              data, or environment problem?
4. INSPECT    Break immediately before the bad line and inspect its inputs.
5. CONTRACT   State what the operation expects and what it received.
6. FIX        Change one thing—the earliest point where the invariant breaks.
7. VERIFY     Rerun, then check correctness beyond “it did not crash.”
```

A strong interview explanation sounds like:

> “The final exception says the target must be `Long`. The first relevant
> frame is our loss call. `CrossEntropyLoss` expects class-index targets shaped
> `(N,)` with dtype `torch.long`. I’ll inspect the target, fix its construction,
> then verify its shape, dtype, and value range.”

## Fast decision guide

```text
Did Python raise an exception?
|
+-- Yes
|   |
|   +-- Import/file/name error? --------> Check environment, paths, spelling.
|   +-- Size/dimension/matmul error? ----> Inspect .shape at each boundary.
|   +-- Expected scalar type error? ----> Inspect .dtype.
|   +-- CPU/CUDA/MPS mismatch? ----------> Inspect .device for model + tensors.
|   +-- Index out of range? -------------> Inspect min/max vs valid range.
|   +-- Grad/backward/in-place error? ----> Inspect requires_grad, graph reuse,
|   |                                      detach(), no_grad(), and mutations.
|   +-- DataLoader worker error? --------> Retry with num_workers=0.
|
+-- No, but result is wrong
    |
    +-- Loss does not move? -------------> Check data, targets, logits/loss,
    |                                      zero_grad, backward, step, train mode.
    +-- Train good, validation bad? ------> Check leakage, mode, preprocessing,
    |                                      overfitting, and split correctness.
    +-- Results change each run? ---------> Check seeds, shuffle, workers, and
    |                                      nondeterministic GPU operations.
    +-- NaN/Inf? -------------------------> Check inputs, loss components,
    |                                      learning rate, division/log/sqrt,
    |                                      and gradient magnitude.
    +-- Too slow or out of memory? -------> Check tensor sizes, graph retention,
                                           batch size, dtype, and device use.
```

## Inspect before guessing

Put a breakpoint immediately before the failing operation:

```python
breakpoint()
```

For any important tensor, check the six basic facts:

```python
print("type:", type(x))
print("shape:", x.shape)
print("dtype:", x.dtype)
print("device:", x.device)
print("requires_grad:", x.requires_grad)
print("range:", x.min().item(), x.max().item())
print("finite:", torch.isfinite(x).all().item())
```

For a model:

```python
print(next(model.parameters()).device)
print(next(model.parameters()).dtype)
print(model.training)  # True after train(), False after eval()
```

Useful assertions near boundaries:

```python
assert x.ndim == 2
assert x.shape[0] == targets.shape[0]
assert x.device == targets.device
assert torch.isfinite(x).all()
```

## Read a traceback efficiently

1. Read the **final line** first: exception type and message.
2. Scan upward for the **first frame in your code**.
3. Inspect the values passed from that frame into PyTorch.
4. Do not edit a library file just because the exception was raised there.
5. After the first fix, rerun from the top; another error may have been hidden.

Run under `pdb` without editing source:

```bash
python -m pdb script.py
python -m pdb -c continue script.py  # stop at the unhandled exception
```

Useful `pdb` commands:

- `n`: next line
- `s`: step into
- `r`: run until current function returns
- `c`: continue
- `w`: show stack
- `u` / `d`: move up/down the stack
- `l` / `ll`: show source
- `p expr` / `pp expr`: inspect a value
- `a`: show function arguments
- `q`: quit

## Common mistakes and fixes

### 1. Matrix or linear-layer shape mismatch

Typical errors:

```text
RuntimeError: mat1 and mat2 shapes cannot be multiplied
RuntimeError: size mismatch
```

Inspect:

```python
print(x.shape)
print(model.fc.weight.shape)  # (out_features, in_features)
```

`nn.Linear(in_features, out_features)` expects the input's final dimension to
equal `in_features`. Trace shapes forward from the batch; do not blindly
reshape until the error disappears.

```python
x = x.flatten(start_dim=1)  # preserve batch dimension
self.fc = nn.Linear(actual_width, num_classes)
```

Avoid `x.view(-1, something)` unless you can explain every dimension.

### 2. Batch or target shape mismatch

Typical errors:

```text
Expected input batch_size (...) to match target batch_size (...)
0D or 1D target tensor expected
```

Inspect:

```python
print("logits", logits.shape)
print("targets", targets.shape)
```

For multiclass classification:

```text
logits:  (N, C)
targets: (N,)
```

Fix an accidental extra target dimension:

```python
targets = targets.squeeze(1)  # only when shape is (N, 1)
```

Do not use bare `squeeze()` when batch size can be 1; it may remove the batch
dimension too.

### 3. Wrong dtype

Typical errors:

```text
Expected scalar type Long but found Float
mat1 and mat2 must have the same dtype
```

Inspect:

```python
print(inputs.dtype, targets.dtype)
```

Common contracts:

```python
inputs = inputs.float()    # model inputs are usually floating point
targets = targets.long()   # class IDs for CrossEntropyLoss / NLLLoss
targets = targets.float()  # targets for BCEWithLogitsLoss
```

Fix dtype where the tensor is created or collated when possible.

### 4. CPU, CUDA, or MPS device mismatch

Typical error:

```text
Expected all tensors to be on the same device
```

Inspect model parameters, inputs, targets, masks, and new helper tensors:

```python
print(next(model.parameters()).device)
print(inputs.device, targets.device)
```

Portable setup:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

for inputs, targets in loader:
    inputs = inputs.to(device)
    targets = targets.to(device)
```

Create helpers on the correct device:

```python
mask = torch.ones_like(inputs, dtype=torch.bool)
positions = torch.arange(length, device=inputs.device)
```

Do not hard-code `.cuda()` in code that must run in a CPU Codespace.

### 5. `CrossEntropyLoss` used incorrectly

Correct contract:

```text
input:  raw logits, shape (N, C), floating point
target: class IDs,  shape (N,),   torch.long, values 0 through C-1
```

```python
logits = model(inputs)
loss = nn.CrossEntropyLoss()(logits, targets.long())
```

Common mistakes:

- applying `softmax` before the loss;
- passing one-hot targets instead of class indices;
- targets shaped `(N, 1)`;
- class IDs outside `[0, C - 1]`.

For evaluation only:

```python
probabilities = logits.softmax(dim=1)
predictions = logits.argmax(dim=1)
```

### 6. `BCEWithLogitsLoss` used incorrectly

Logits and targets must have the same shape and be floating point. Targets
commonly contain `0.0` and `1.0`.

```python
logits = model(inputs).squeeze(1)
targets = targets.float()
loss = nn.BCEWithLogitsLoss()(logits, targets)
```

Do not apply `sigmoid` before `BCEWithLogitsLoss`; it already includes a
numerically stable sigmoid. Use sigmoid only to interpret predictions.

### 7. Embedding index out of range

Typical error:

```text
IndexError: index out of range in self
```

Every ID passed to `nn.Embedding(num_embeddings, dim)` must be an integer in
`[0, num_embeddings - 1]`.

```python
print(token_ids.dtype)
print(token_ids.min().item(), token_ids.max().item())
print(embedding.num_embeddings)
```

Fix the vocabulary contract:

```python
PAD_ID = 0
UNK_ID = 1
token_id = vocabulary.get(token, UNK_ID)
embedding = nn.Embedding(len(vocabulary), dim, padding_idx=PAD_ID)
```

Check for off-by-one IDs and reserve explicit `<PAD>` and `<UNK>` rows.

### 8. Variable-length sequences cannot be stacked

Typical error:

```text
stack expects each tensor to be equal size
```

The default `DataLoader` collator stacks examples. Pad inside a custom
`collate_fn`:

```python
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    sequences, labels = zip(*batch)
    padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return padded, labels

loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
```

Carry lengths or an attention mask if the model must ignore padding.

### 9. Gradients accumulate unexpectedly

PyTorch adds new gradients to existing `.grad` values. Standard order:

```python
optimizer.zero_grad()
logits = model(inputs)
loss = criterion(logits, targets)
loss.backward()
optimizer.step()
```

If `zero_grad()` is missing, gradients grow across batches. Intentional
gradient accumulation is different and should be explicit.

### 10. Tensor does not require gradients

Typical error:

```text
element 0 of tensors does not require grad and does not have a grad_fn
```

Inspect:

```python
print(loss.requires_grad, loss.grad_fn)
print(torch.is_grad_enabled())
```

Common causes:

- forward pass occurred inside `torch.no_grad()`;
- tensor was detached with `.detach()`;
- tensor was rebuilt with `torch.tensor(existing_tensor)`;
- loss became a Python number via `.item()` before `backward()`.

```python
loss.backward()
logged_loss = loss.item()  # only for logging
```

### 11. Backward through the graph twice

Typical error:

```text
Trying to backward through the graph a second time
```

A tensor from a prior iteration may still hold its graph. Keep values, not
graphs, when logging or saving outputs:

```python
running_loss += loss.item()
saved_output = output.detach()
```

`retain_graph=True` is rarely the right first fix.

### 12. In-place operation breaks autograd

Typical error:

```text
one of the variables needed for gradient computation has been modified by an inplace operation
```

Look for operations ending in `_` and slice assignments:

```python
x.relu_()
x += residual
x[:, 0] = 0
```

Try out-of-place operations:

```python
x = torch.relu(x)
x = x + residual
```

For a better traceback during investigation:

```python
torch.autograd.set_detect_anomaly(True)
```

Anomaly detection is slow; remove it after debugging.

### 13. Training and evaluation modes are mixed up

`model.eval()` changes dropout and batch-normalization behavior.
`torch.no_grad()` disables gradient recording. They solve different problems.

```python
model.train()
for inputs, targets in train_loader:
    ...

model.eval()
with torch.no_grad():
    for inputs, targets in validation_loader:
        ...
```

Call `model.train()` again when returning to training after validation.

### 14. NaN or infinite loss

Find the first non-finite value:

```python
assert torch.isfinite(inputs).all()
assert torch.isfinite(logits).all()
assert torch.isfinite(loss)
```

Common causes:

- invalid inputs to `log`, `sqrt`, or division;
- excessive learning rate;
- exploding gradients;
- extreme, unnormalized inputs;
- unstable manual probability calculations.

Inspect gradient size:

```python
for name, parameter in model.named_parameters():
    if parameter.grad is not None:
        print(name, parameter.grad.norm().item())
```

Potential fixes depend on the cause: add an epsilon to a denominator, use a
stable built-in loss, lower the learning rate, or clip gradients:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Do not hide NaNs with `nan_to_num` before finding their source.

### 15. CUDA out of memory

```python
print(torch.cuda.memory_summary())
```

Common causes and fixes:

- batch too large → reduce batch size;
- stored graph-connected tensors → store `.detach()` or `.item()`;
- missing `no_grad()` during evaluation → add it;
- duplicated model/inputs on GPU → remove copies;
- gradients not cleared → use `zero_grad()` each iteration.

`torch.cuda.empty_cache()` releases unused cached blocks. It does not free live
tensors or fix graph retention.

### 16. NumPy conversion fails

Errors often mention CUDA tensors or tensors requiring gradients:

```python
array = tensor.detach().cpu().numpy()
```

### 17. Checkpoint loads on the wrong device

```python
checkpoint = torch.load(path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
```

If keys have an unexpected `module.` prefix, check whether the checkpoint came
from `DataParallel`. Confirm architecture and key names before rewriting them.

### 18. DataLoader hides the real exception

Worker processes can obscure tracebacks. Reproduce synchronously:

```python
loader = DataLoader(dataset, batch_size=..., num_workers=0)
```

Find the exact failing example:

```python
for index in range(len(dataset)):
    try:
        dataset[index]
    except Exception:
        print("failed index:", index)
        raise
```

Restore workers only after the data path works.

## When code runs but learning is wrong

Check in this order:

1. **One batch:** can the model overfit one tiny batch?
2. **Inputs:** plausible values, shapes, and preprocessing?
3. **Targets:** correct examples, dtype, shape, range, and alignment?
4. **Output:** raw logits with the expected number of classes?
5. **Loss:** correct for the task and target representation?
6. **Gradients:** finite and nonzero?
7. **Update:** are `zero_grad`, `backward`, and `step` executed?
8. **Modes:** `train()` for training; `eval()` + `no_grad()` for validation?
9. **Split:** no duplicate IDs, leakage, or validation-fitted preprocessing?
10. **Metric:** right dimension and reset each epoch?

Quick gradient check:

```python
loss.backward()
for name, parameter in model.named_parameters():
    if parameter.grad is not None:
        print(name, parameter.grad.abs().mean().item())
        break
```

If a tiny model cannot overfit one tiny batch, suspect the pipeline or training
loop before tuning hyperparameters.

## Data pipeline checks

JSON Lines/JSONL contains one object per line:

```python
with open(path) as handle:
    rows = [json.loads(line) for line in handle if line.strip()]
```

CSV values begin as strings. Explicitly choose types for IDs, numbers, and
missing values. Confirm join keys use the same type and normalization.

Check split membership, not only lengths:

```python
assert train_ids.isdisjoint(validation_ids)
assert len(train_ids | validation_ids) == len(all_ids)
```

Use a local seeded random generator:

```python
rng = random.Random(7)
rng.shuffle(records)
```

Fit tokenizers, scalers, vocabularies, and other learned preprocessing on the
training split only.

## Minimal reproducibility setup

```python
import random
import numpy as np
import torch

seed = 7
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
```

Seeding does not guarantee every GPU operation is deterministic. During
debugging, make the failing example small and CPU-based when possible.

## Search effectively

Search the exact final error plus the smallest relevant API:

```text
PyTorch mat1 and mat2 shapes cannot be multiplied nn.Linear
PyTorch CrossEntropyLoss expected scalar type Long found Float
PyTorch expected all tensors to be on the same device
PyTorch embedding index out of range num_embeddings
PyTorch backward through graph a second time
PyTorch DataLoader stack expects each tensor equal size
```

Prefer official PyTorch docs for API contracts. Use Stack Overflow to generate
hypotheses, then verify them against the actual tensor values.

## Think-aloud template

```text
Observed:       “The failure is ... at file:line.”
Interpretation: “This operation expects ..., but appears to have received ...”
Inspection:     “I’ll check shape, dtype, device, range, and gradient state.”
Hypothesis:     “My current hypothesis is ... because ...”
Small fix:      “I’ll change only ... first.”
Verification:   “The error is gone; now I’ll verify ...”
Prevention:     “An assertion/test for ... would catch this earlier.”
```

## Final cleanup

Remove temporary debugger hooks:

```bash
rg -n 'breakpoint\(|pdb\.set_trace|debugpy\.' --type py
```

Rerun the original command without the debugger, then inspect:

```bash
git diff
git status
```
