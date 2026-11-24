# ML Debugging Interview Practice Kit

This kit targets the interview described: practical Python/PyTorch debugging,
not model tuning or exploratory analysis.

Keep [`CHEATSHEET.md`](CHEATSHEET.md) open for the symptom-to-fix decision
guide, tensor inspection commands, and common PyTorch failure patterns.

## Recommended setup

```bash
cd interview-prep
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

In a GitHub Codespace, open this folder in VS Code and select the `.venv`
interpreter.

## How to practice

For each exercise:

1. Start a 20-minute timer.
2. Run the script unchanged and read the complete traceback.
3. Say out loud what the exception means and identify the first frame in code
   you control.
4. Form one hypothesis before editing.
5. Use `breakpoint()`, the VS Code debugger, or a temporary print to inspect
   the relevant value, shape, dtype, or device.
6. Make the smallest defensible change.
7. Rerun the script and check the stated definition of done.
8. Explain the root cause, the fix, and one regression check.

Do not open `solutions/` until the timer ends.

## Exercises

### 1. JSONL + CSV pipeline

```bash
python exercises/01_data_pipeline.py
```

Debug a small join between line-delimited JSON and CSV data.

Definition of done:
- script exits successfully;
- writes `exercises/data/processed.csv`;
- output contains four users in the same order as the events file;
- missing numeric values are represented consistently rather than crashing.

### 2. Dataset split leakage

```bash
python exercises/02_split.py
```

The script runs but violates its own split checks. Diagnose the logic error.

Definition of done:
- 8 training examples and 2 validation examples;
- no ID appears in both sets;
- repeated runs produce the same split.

### 3. Variable-length batches

```bash
python exercises/03_dataloader.py
```

Fix batching and loss computation for variable-length token sequences.

Definition of done:
- batch token shape is `(3, 4)`;
- label shape is `(3,)` and label dtype is `torch.int64`;
- one forward/loss computation succeeds.

### 4. CPU/GPU device handling

```bash
python exercises/04_device.py
```

Make the script work on a CPU-only Codespace and on a CUDA host.

Definition of done:
- no hard requirement for CUDA;
- model, input, and target are on the same selected device;
- one optimizer step succeeds.

### 5. Custom embedding indices

```bash
python exercises/05_embeddings.py
```

Trace an index error caused by the vocabulary/OOV contract.

Definition of done:
- known tokens keep stable IDs;
- unknown tokens map to a valid `<UNK>` row;
- padding has its own valid row;
- output shape is `(2, 3, 4)`.

### 6. Training-loop correctness

```bash
python exercises/06_training_loop.py
```

Fix a minimal classifier training loop. Focus on API contracts and state,
not hyperparameter tuning.

Definition of done:
- target dtype is valid for `CrossEntropyLoss`;
- the model provides logits to the loss;
- gradients are cleared each iteration;
- evaluation does not track gradients;
- final training loss is below initial training loss.

## Codespaces/VS Code speed drill (10 minutes)

Practice these without a mouse where possible:

- open terminal: ``Ctrl+` ``;
- global text search: `Ctrl+Shift+F` / `Cmd+Shift+F`;
- go to definition: `F12`;
- find references: `Shift+F12`;
- command palette: `Ctrl+Shift+P` / `Cmd+Shift+P`;
- set a breakpoint by clicking the gutter or pressing `F9`;
- start/continue debugger: `F5`;
- step over: `F10`; step into: `F11`;
- inspect Variables, Watch, Call Stack, and Debug Console;
- run `git diff` before declaring the fix complete.

## Think-aloud template

Use this exact rhythm in the interview:

1. **Observed:** “The failure is `...` at `file:line`.”
2. **Interpretation:** “That usually means the API expected `...`, but got
   `...`.”
3. **Inspection:** “I want to inspect this variable's type/shape/dtype/device.”
4. **Hypothesis:** “My current hypothesis is `...`.”
5. **Small fix:** “I’ll change only `...` first.”
6. **Verification:** “The original error is gone; now I’ll verify `...`.”
7. **Prevention:** “A small assertion/test for `...` would prevent this.”

See `MOCK_INTERVIEW.md` for a timed rehearsal and `solutions/` only after
attempting each exercise.
