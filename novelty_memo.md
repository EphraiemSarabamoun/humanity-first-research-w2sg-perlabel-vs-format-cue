# Novelty memo (hand-written; Semantic Scholar still throttled 429 on the gate re-run 2026-07-09)

Semantic Scholar returned HTTP 429 on every query for both the original gate run and the
2026-07-09 re-run, so `related_work.json` has no auto-fetched neighbors. Nearest-neighbor
analysis below is hand-curated; every reference is real and cross-checked. references.bib
holds 12 hand-built, verified entries (no fabricated arXiv ids).

## Nearest prior work (named neighbors + our delta against each)
- **Burns et al. 2023, "Weak-to-Strong Generalization" (arXiv:2312.09390)** — the seminal W2SG setup
  and PGR metric. A strong student trained on a weak supervisor's per-item labels recovers part of
  the weak->strong gap; their quality lever is a student-side confidence (auxiliary) loss. Labels are
  DENSE and REAL. They never ask whether the per-item correspondence of the labels is load-bearing.
  **Delta:** we hold format fixed and destroy per-item signal (shuffle/random/constant) to isolate it;
  Burns never runs that ablation.
- **Min et al. 2022, "Rethinking the Role of Demonstrations" (arXiv:2202.12837)** — the conceptual
  ancestor: in in-context learning, replacing gold demonstration labels with RANDOM labels barely
  hurts; the label SPACE and FORMAT do most of the work, not the per-item input->label mapping.
  **Delta:** this is an ICL result and comes out the OPPOSITE way in our finetuning W2SG regime —
  destroying per-item signal collapses the student to chance (~0.51-0.56) while real weak labels reach
  ~0.954. The Min-style decomposition does NOT transfer from ICL to W2SG finetuning.
- **Elicitation-vs-imitation (KB scalable-oversight open problem)** — when a strong model trained on
  weak labels beats the weak labeler, is it recovering latent knowledge or imitating the labeler?
  Our result speaks to this: the student needs genuine per-item weak-label information, i.e. it is
  learning FROM the labels, not merely being format-cued into eliciting its own latent competence.
- **Zhang et al. 2017, "Understanding Deep Learning Requires Rethinking Generalization" (arXiv:1611.03530)**
  — nets can fit random labels (memorization). **Delta:** our random-label condition does NOT reach
  train-set-driven test accuracy; the LoRA-adapted student stays at chance on held-out data, so the
  format-only signal carries no generalizing information — consistent with memorization-not-generalization.
- **Irving et al. 2018 (debate, arXiv:1805.00899) & Bowman et al. 2022 (measuring scalable oversight,
  arXiv:2211.03540)** — situate W2SG inside the scalable-oversight agenda. **Delta:** we contribute a
  clean negative control for one mechanistic question within that agenda rather than a new oversight
  protocol.

## Also cited (methodological / benchmark anchors, not novelty competitors)
- Hu et al. 2021 (LoRA, arXiv:2106.09685) — the finetuning method for the student.
- Socher et al. 2013 (SST/SST-2, EMNLP 2013) and Wang et al. 2018 (GLUE, arXiv:1804.07461) — the task.
- Yang et al. 2024 (Qwen2.5 report, arXiv:2412.15115) — the student/weak-labeler model family.
- Northcutt et al. 2021 (Confident Learning, arXiv:1911.00068) and Hinton et al. 2015 (distillation,
  arXiv:1503.02531) — noisy-label / label-transfer framing for the discussion.

## Our delta (explicit contribution sentence)
Prior W2SG trains the strong student on the weak labeler's real per-item labels. We decompose that
training signal into two components — the PER-ITEM label information and the mere TASK-FORMAT cue of
being finetuned in the task's prompt/label format — by finetuning the strong student on conditions
that PRESERVE format while DESTROYING per-item signal (label-shuffle, random-label, constant-majority),
against the real-weak-label condition and a gold ceiling, all size- and format-matched. Empirically the
format-only conditions recover NOTHING above chance (~0.51-0.56) while real weak labels reach ~0.954
(gold ceiling 0.962): per-item weak-label signal is REQUIRED; task-format cueing alone is insufficient.
This is the finetuning-regime analog of Min et al., previously unstudied in W2SG, and it inverts their
ICL conclusion.

## Verdict used: incremental->novel (proceed; delta above is the contribution sentence).
Re-run novelty_search before the write stage if S2 recovers.
