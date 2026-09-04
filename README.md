# BayPIRE

BayPIRE is the research code for uncertainty-aware sparse prompt evaluation.

The code mirrors the paper's finite prompt-item target:

```text
raw benchmark data -> EvaluationMatrix -> ObservationMask -> estimator
                  -> observed-retention reconstruction -> metrics
```

Benchmark-specific logic ends once an `EvaluationMatrix` has been created.
MMLU, OlympiadBench, and later benchmarks should then use the same masks,
estimators, recovery rule, and metrics.

## Quick Start

From `/Users/jayhung/M2R_LLM_Eval`:

```bash
cd /Users/jayhung/M2R_LLM_Eval/baypire

../.venv/bin/python scripts/run_mmlu_subject_batch.py \
  --results-dir results/mmlu_predictive \
  --model-id {model_id} \
  --subjects abstract_algebra anatomy astronomy business_ethics clinical_knowledge college_biology college_chemistry college_computer_science college_mathematics college_physics \
  --mask-types balanced random \
  --budgets 200,400,800,1600 \
  --n-masks 10 \
  --methods Avg PE-Rasch Bayes-MAP BayPIRE \
  --draws 1000 \
  --tune 1000 \
  --chains 4 \
  --target-accept 0.97
```

The default MMLU input points to the copied PromptEval chunks:

```text
data/external/prompteval_mmlu/chunks
```

Results are written to one folder per model, subject, and mask type:

```text
baypire/results/mmlu/{model_id}/{subject}/{mask_type}/
```

For example:

```text
results/mmlu/{model_id}/abstract_algebra/balanced/balanced.csv
results/mmlu/{model_id}/abstract_algebra/balanced/balanced_summary.csv
results/mmlu/{model_id}/abstract_algebra/balanced/balanced.json
```

Budgets follow the PromptEval convention by default:

```text
200, 400, 800, 1600 observed prompt-item cells
```

For quick robustness checks, fractional budgets are still accepted. For example,
`--budgets 0.1,0.2` means 10% and 20% of the prompt-item matrix.

Active scripts:

```text
scripts/download_prompteval_mmlu.py  # import released PromptEval MMLU data
scripts/run_mmlu.py                  # run BayPIRE vs PromptEval experiments
scripts/make_figures.py              # make plots from saved result CSVs
```

## Inspecting Results

Use the notebook for reading tables and making paper figures:

```text
notebooks/01_inspect_mmlu_abstract_algebra.ipynb
```

Install notebook dependencies when needed:

```bash
python -m pip install -r requirements-notebook.txt
```

Then open Jupyter from this folder:

```bash
jupyter lab
```

The copied OlympiadMath raw run is preserved as reference data:

```text
data/raw/olympiad/olym_math_deepseek_p50/
```

The Olympiad-specific config, scripts, processed folder, and result folder are
archived outside the active MMLU workflow:

```text
archive/olympiad/
```

## Importing More MMLU Subjects

To fetch the remaining PromptEval MMLU subjects from Hugging Face for the
current model split:

```bash
PYTHONPATH=src ../.venv/bin/python scripts/download_prompteval_mmlu.py \
  --models meta_llama_llama_3_8b
```

The importer is resumable. Existing subject chunks are skipped unless `--force`
is supplied.

To inspect remote subject and model split names first:

```bash
PYTHONPATH=src ../.venv/bin/python scripts/download_prompteval_mmlu.py --list-remote
```

To combine all local subject chunks into one processed parquet:

```bash
PYTHONPATH=src ../.venv/bin/python scripts/download_prompteval_mmlu.py \
  --models meta_llama_llama_3_8b \
  --combine
```

## Package Layout

```text
src/baypire/
  data/          EvaluationMatrix and benchmark adapters
  design/        Observation masks and mask diagnostics
  models/        Avg, PE-Rasch, Bayes-MAP, BayPIRE
  recovery/      observed-retention reconstruction and prompt scores
  metrics/       recovery, predictive, and calibration metrics
  diagnostics/   MCMC and identifiability summaries
  experiments/   benchmark-agnostic runner
```

## Methods

`Avg` uses observed prompt means.

`PE` uses the official PromptEval `ExtendedRaschModel` when a local PromptEval
repo is available; otherwise it uses the local weakly regularized Rasch
implementation with the same additive mean structure.

`Bayes-MAP` is a MAP additive Rasch estimator with explicit Gaussian shrinkage.

`BayPIRE` is the Bayesian additive IRT model:

```text
Y_ij ~ Bernoulli(mu_ij)
logit(mu_ij) = alpha + prompt_effect_i - item_difficulty_j
```

All methods use the same observed-retention reconstruction rule:

```text
observed cells   -> keep Y_ij exactly
hidden cells     -> use fitted probability mu_hat_ij
```
