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

## Understanding tensor shapes

### Shape vocabulary

A tensor's **rank** is its number of dimensions (axes). Its **shape** gives the
size of each dimension.

```python
x = torch.zeros(32, 10, 128)
print(x.ndim)       # 3 dimensions
print(x.shape)      # torch.Size([32, 10, 128])
print(x.shape[0])   # 32 items along dimension 0
print(x.shape[-1])  # 128 items along the final dimension
```

Common letters:

| Letter | Meaning | Example |
|---|---|---|
| `B` or `N` | batch size: number of examples processed together | `32` sentences |
| `F` | number of input features | `12` columns per customer |
| `C` | channels or classes, depending on context | `3` RGB channels or `10` classes |
| `H`, `W` | image height and width | `224 × 224` pixels |
| `L` or `T` | sequence length or time steps | `50` tokens |
| `D` | embedding, hidden, or feature width | `128` values per token |
| `V` | vocabulary size | `30,000` possible token IDs |

The letters describe the **meaning** of each axis. The numbers are the current
sizes. For example, `(B, L, D) = (32, 50, 128)` means 32 examples, 50 tokens
per example, and 128 learned features per token.

### Common input and output shapes

| Data or tensor | Typical shape | What each dimension means |
|---|---|---|
| One tabular example | `(F,)` | `F` feature values |
| Tabular batch | `(B, F)` | examples × features |
| One RGB image | `(C, H, W)` | channels × height × width |
| Image batch | `(B, C, H, W)` | examples × channels × height × width |
| Token IDs | `(B, L)` | examples × token positions |
| Token embeddings | `(B, L, D)` | examples × token positions × features |
| Sequence mask | `(B, L)` | whether each token position is valid |
| Multiclass logits | `(B, C)` | one score per class for each example |
| Multiclass labels | `(B,)` | one integer class ID per example |
| Binary logits | `(B,)` or `(B, 1)` | one score per example |
| Segmentation logits | `(B, C, H, W)` | class scores at every pixel |
| Segmentation labels | `(B, H, W)` | one class ID at every pixel |

`C` is overloaded: near image input it usually means image channels; at the
classifier output it usually means classes. Use variable names and surrounding
operations to determine which meaning applies.

### The batch dimension

Dimension 0 is usually the batch dimension. A model may expect a batch even
when you have only one example.

```python
image.shape                  # (3, 224, 224): one image
image = image.unsqueeze(0)
image.shape                  # (1, 3, 224, 224): batch of one
```

If a `DataLoader` uses `batch_size=32`, it normally adds the leading `B=32`
dimension. The final batch can be smaller unless `drop_last=True`.

Never hard-code the batch size into a reshape:

```python
x = x.reshape(x.shape[0], -1)  # preserves the actual batch size
```

### What common layers do to shapes

#### `nn.Linear`

`nn.Linear(D_in, D_out)` transforms the **last** dimension:

```text
(B, D_in)    -> (B, D_out)
(B, L, D_in) -> (B, L, D_out)
```

It keeps every leading dimension. The required contract is:

```python
x.shape[-1] == linear.in_features
```

The weight is stored as `(D_out, D_in)`, which can look reversed when printed.

#### `nn.Embedding`

An embedding turns each integer ID into a vector:

```text
token IDs:  (B, L)    integer IDs
embedding:  (V, D)    lookup table
output:     (B, L, D) one D-wide vector per token
```

The embedding adds a feature dimension; it does not remove the sequence
dimension.

#### `nn.Conv2d`

PyTorch convolution expects channel-first image batches:

```text
input:  (B, C_in,  H_in,  W_in)
output: (B, C_out, H_out, W_out)
```

`C_in` must match the layer's `in_channels`. `C_out` is set by `out_channels`.
Stride, padding, dilation, and kernel size determine the new height and width.

If image data arrives as `(B, H, W, C)`, reorder axes—do not reshape:

```python
x = x.permute(0, 3, 1, 2)  # BHWC -> BCHW
```

#### Pooling and flattening

Global/adaptive pooling can reduce spatial dimensions:

```text
(B, C, H, W) -> AdaptiveAvgPool2d((1, 1)) -> (B, C, 1, 1)
```

Then flatten features while preserving the batch:

```python
x = torch.flatten(x, start_dim=1)  # (B, C, 1, 1) -> (B, C)
```

#### RNNs and transformers

Many sequence layers use one of two layouts:

```text
batch_first=True:  (B, L, D)
batch_first=False: (L, B, D)
```

Check the layer configuration before swapping dimensions. A tensor can have
the right rank but silently use the wrong axis order.

### End-to-end shape examples

Text classification:

```text
token IDs                 (32, 50)      B=32, L=50
nn.Embedding(30000, 128)  (32, 50, 128) adds D=128
mean over dim=1           (32, 128)     removes token axis L
nn.Linear(128, 4)         (32, 4)       four class logits
targets                   (32,)         one class ID per example
```

```python
embedded = embedding(token_ids)  # (B, L, D)
pooled = embedded.mean(dim=1)    # average tokens -> (B, D)
logits = classifier(pooled)      # (B, C)
loss = criterion(logits, targets)
```

Image classification:

```text
RGB images                (32, 3, 224, 224)
convolution block         (32, 64, 56, 56)
adaptive average pool     (32, 64, 1, 1)
flatten from dim 1        (32, 64)
nn.Linear(64, 10)         (32, 10) ten class logits
targets                   (32,)    one class ID per image
```

Tabular classification:

```text
input features            (32, 12) 12 columns per row
nn.Linear(12, 64)         (32, 64)
nn.Linear(64, 3)          (32, 3)  three class logits
targets                   (32,)
```

### Shape-changing operations

| Operation | Meaning | Example |
|---|---|---|
| `unsqueeze(dim)` | add a size-1 axis | `(C,H,W)` → `(1,C,H,W)` |
| `squeeze(dim)` | remove one known size-1 axis | `(B,1)` → `(B,)` |
| `flatten(start_dim=1)` | combine dimensions | `(B,C,H,W)` → `(B,C×H×W)` |
| `reshape(...)` | regroup the same elements | `(B,L,D)` → `(B,L×D)` |
| `permute(...)` | reorder axes | `(B,H,W,C)` → `(B,C,H,W)` |
| `transpose(a,b)` | swap two axes | `(B,L,D)` → `(B,D,L)` |
| `mean(dim=k)` | reduce/remove axis `k` | `(B,L,D)` → `(B,D)` for `k=1` |
| `stack(..., dim=k)` | add a new axis | two `(B,D)` tensors → `(2,B,D)` |
| `cat(..., dim=k)` | extend an existing axis | two `(B,D)` tensors → `(B,2D)` |

`reshape` changes grouping, while `permute` changes axis order. They are not
interchangeable. The number of elements must stay constant across a reshape:

```python
assert x.numel() == reshaped.numel()
```

Use a specific dimension with `squeeze(dim)` so a batch of size 1 does not
accidentally lose its batch axis.

### A repeatable shape-debugging method

1. Write the intended shape above each important line.
2. Break immediately before the failing operation.
3. Print the actual input shapes and the layer's expected attributes.
4. Find the **first** line where actual and intended shapes diverge.
5. Fix the semantic operation at that line—not just the final reshape.
6. Test with batch size 1 and a normal batch to catch unsafe `squeeze()` code.

```python
print("raw batch:", inputs.shape)
x = embedding(inputs)
print("after embedding:", x.shape)
x = x.mean(dim=1)
print("after pooling:", x.shape)
print("classifier expects:", classifier.in_features)
logits = classifier(x)
print("logits:", logits.shape, "targets:", targets.shape)
```

For a larger model, temporary forward hooks show the input and output shapes of
each leaf module:

```python
def show_shape(name):
    def hook(module, inputs, output):
        input_shapes = [getattr(value, "shape", None) for value in inputs]
        output_shape = getattr(output, "shape", None)
        print(name, input_shapes, "->", output_shape)
    return hook

handles = []
for name, module in model.named_modules():
    if not list(module.children()):
        handles.append(module.register_forward_hook(show_shape(name)))

model(sample_batch)

for handle in handles:
    handle.remove()
```

Remove hooks after debugging so they do not keep printing during training.

### Decode common shape errors

```text
mat1 and mat2 shapes cannot be multiplied (B×K and I×O)
```

The input's final size `K` does not match the linear layer's `in_features=I`.
Inspect the operation that produced `K`.

```text
Expected 4-dimensional input for 4-dimensional weight
```

`Conv2d` expected `(B,C,H,W)`. You may be missing the batch or channel axis,
or the data may be in `(B,H,W,C)` order.

```text
Dimension out of range
```

Code referenced an axis that does not exist. Print `x.ndim` and remember that
valid positive indices are `0` through `x.ndim - 1`.

```text
Sizes of tensors must match except in dimension ...
```

For `torch.cat`, every dimension except the concatenation dimension must match.
Print all input shapes and decide which semantic axis should grow.

```text
permute: number of dimensions ... does not match
```

`permute` needs exactly one index for each existing axis. Print `x.shape` before
choosing the new order.

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
See **Understanding tensor shapes** above for complete layer-by-layer examples.

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
