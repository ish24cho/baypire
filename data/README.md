# BayPIRE Data

This folder is intentionally separated into raw, external, processed, and mask
outputs.

The code should convert benchmark-specific files into `EvaluationMatrix` objects
before calling any statistical estimator.

## Current Local Data

PromptEval MMLU correctness chunks:

```text
data/external/prompteval_mmlu/chunks/
```

Copied OlympiadMath DeepSeek p50 run:

```text
data/raw/olympiad/olym_math_deepseek_p50/
```

The Olympiad prompt-item matrix used by `scripts/run_olympiad.py` is:

```text
data/raw/olympiad/olym_math_deepseek_p50/matrices/Y_pilot.pkl
```

Official PromptEval repository copy for PE-Rasch:

```text
vendor/prompteval/
```
