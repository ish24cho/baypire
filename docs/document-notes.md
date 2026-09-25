# Notes on the Uploaded Documents

The paper and slides are copies of the supplied PDFs under shorter filenames. Their content has not been edited. The README incorporates these clarifications identified while preparing the repository.

## Presentation Corrections

- Slide 9: for scores `[0.40, 0.55, 0.60, 0.75, 0.90]`, the empirical CDF at 0.60 is **3/5 = 0.6**, because three scores are less than or equal to 0.60.
- Slide 14: when discussing informativeness, check whether interval width **decreases**, rather than increases, as more cells are observed.
- Slide 11: the local official PromptEval reference code uses scikit-learn logistic regression with `C=100`. The description of PE-Rasch as unregularised maximum likelihood should be revised; it is a regularised point estimator.
- Slide 18: the comparison does not establish that shrinkage alone, or posterior averaging alone, causes the improvement. Isolating these contributions requires a controlled ablation.

## Paper and Implementation Clarifications

The PE-Rasch regularisation clarification also applies to the paper's maximum-likelihood description. The local upstream reference revision is `c238d6473f2e7f956418b87ccd200e3165f9eb62`; its `prompteval/methods.py` defines `LogisticRegression(reg=1e2)` and passes this value as scikit-learn's `C`. This identifies the inspected local reference checkout; it is not independent proof of the revision used for every historical fit.

The wrapper has a fallback when the official checkout is absent. Reproduction instructions therefore explicitly install the upstream source. Check diagnostic metadata when establishing which backend generated a historical run.

BayPIRE retains 1000 draws per chain across four chains. The code subsamples at most 2000 pooled draws for reconstruction and uses the full retained trace for convergence diagnostics.

The line-plot shading exported from paper assets represents one standard deviation across model-subject summaries after averaging mask replicates. It is not a standard error or a posterior interval.

At budget 1600, the reported convergence flag rate is 0.934. The 95% prompt intervals have aggregate coverage 0.981, indicating conservatism. Neither result is proof of convergence or universally exact calibration.
