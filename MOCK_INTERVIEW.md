# 60-Minute Mock Interview

Have a partner choose two exercises without telling you which ones in advance.
If practicing alone, pick them randomly.

## Timing

- **0–5 min:** inspect the repository and explain how you will run the code.
- **5–27 min:** first debugging task.
- **27–49 min:** second debugging task.
- **49–56 min:** explain both root causes and regression checks.
- **56–60 min:** questions for the interviewer.

## Interviewer prompts

Use these only if the candidate stalls:

- “What does the traceback say, in your own words?”
- “Which is the first stack frame in code you control?”
- “What are the value's Python type, tensor shape, dtype, and device?”
- “What contract does this API expect?”
- “Could the script run successfully while still being wrong?”
- “How would this behave on a CPU-only machine?”
- “What would you assert or test after the fix?”
- “Is this the smallest change that addresses the root cause?”

## Scoring (20 points)

- **4 — Reproduction and traceback reading:** runs the smallest failing command,
  locates the relevant frame, and does not guess blindly.
- **4 — Inspection:** checks the right runtime evidence (type, contents, shape,
  dtype, device, indices, or split membership).
- **4 — Root-cause reasoning:** explains the violated data/API contract.
- **4 — Fix and verification:** makes a scoped fix and tests the actual success
  condition, including silent correctness bugs.
- **4 — Communication:** thinks aloud, states uncertainty honestly, responds to
  hints, and gives concise technical explanations.

Strong performance: **16–20**. Ready but inconsistent: **12–15**. Below **12**:
repeat the same two exercises after a day, then try two new ones.

## Questions to ask at the end

- “How representative are the debugging tasks of failures the team sees in
  production training pipelines?”
- “How does the team divide responsibility for data quality, training code,
  and infrastructure?”
- “What does strong debugging and technical communication look like on this
  team?”
