# Preparation Plan

## If you have three days

### Day 1 — Python data + debugging mechanics (75 minutes)

- 10 min: Codespaces/VS Code speed drill from `README.md`.
- 20 min: Exercise 1 (JSONL + CSV).
- 20 min: Exercise 2 (split leakage).
- 15 min: rerun both with `breakpoint()` or `python -m pdb`.
- 10 min: explain each root cause aloud without looking at notes.

### Day 2 — PyTorch contracts (90 minutes)

- 10 min: review tensor shape/dtype/device checks in `CHEATSHEET.md`.
- 25 min: Exercise 3 (DataLoader).
- 20 min: Exercise 4 (device handling).
- 20 min: Exercise 5 (embeddings).
- 15 min: redo the slowest exercise from a clean copy.

### Day 3 — Full rehearsal (90 minutes)

- 25 min: Exercise 6 (training loop).
- 60 min: mock interview using `MOCK_INTERVIEW.md`.
- 5 min: write down the two habits you forgot under time pressure.

## If the interview is tomorrow (90 minutes total)

- 10 min: debugger and VS Code shortcuts.
- 15 min: Exercise 1.
- 15 min: Exercise 3.
- 15 min: Exercise 4.
- 15 min: Exercise 5.
- 15 min: Exercise 6.
- 5 min: rehearse the think-aloud template.

## Final readiness test

You are ready when you can do all of the following without prompting:

- summarize a traceback before touching code;
- inspect Python type plus tensor shape, dtype, and device;
- distinguish a crash from a silent-correctness bug;
- explain the expected contract of `Embedding`, `DataLoader`, and
  `CrossEntropyLoss`;
- make one scoped edit and state how you will verify it;
- accept an interviewer hint without becoming defensive or abandoning your
  current evidence.
