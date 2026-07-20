# Incident: LLM-judge accuracy masked a majority-class collapse

## Background: what this system is

This project studies how a protein language model (ESM-2) represents the effects
of genetic variants. As part of that work, we built a separate **evaluation
harness** to answer a side question: can an off-the-shelf LLM, given only basic
information about a missense variant, predict the *molecular mechanism* by which
that variant causes disease?

Each variant in our dataset is labelled with one of three mechanism classes:

- **LOF** — loss of function: the variant breaks or weakens the protein.
- **GOF** — gain of function: the variant gives the protein a new or
  overactive behaviour.
- **DN** — dominant-negative: the broken protein actively interferes with the
  working copies.

The harness uses an LLM as a "judge": for each variant it is shown the gene, the
amino-acid substitution (e.g. `A216D`), and one structural feature — the FoldX
ΔΔG, a computed estimate of how much the substitution destabilizes the protein's
fold — and must return one of the three classes. We then score the LLM's
predictions against the known labels. None of this touches the ESM-2 model
itself; it is a standalone benchmark of the LLM.

## Summary

The system here is an evaluation harness that uses an LLM as a "judge": for each
genetic variant it is given the gene, the amino-acid substitution, and one
structural feature (a computed protein-stability score), and it must predict
which of three disease mechanisms the variant acts through — loss of function
(LOF), gain of function (GOF), or dominant-negative (DN). The harness runs the
LLM over a batch of variants whose true mechanism is already known and scores the
predictions, so it is a benchmark of the LLM, not of any model in the wider
project.

On a batch of 200 labelled variants the judge reported **83.5% accuracy**, which
looked like a working system. It was not. The model had quietly collapsed onto
the single most common answer — **170 of its 200 predictions were LOF** — and the
high accuracy was an artifact of the dataset itself being 80.5% LOF, so that a
trivial classifier which *always* answers LOF already scores 80.5%. The judge was
beating that baseline by only three points and was near-random on the two
minority classes (GOF, DN), which are the scientifically interesting ones. The
collapse was invisible in the headline metric and only surfaced because of the
specific observability the harness was built with: the batch runner does not stop
at accuracy, it also emits a `prediction_counts` field — a histogram of how many
times each class was predicted — and every individual model call is wrapped in
Langfuse's `@observe` decorator, which captures the predicted class, the true
label, latency, retry count and token usage as a structured trace. The
`prediction_counts` histogram is what flagged the skew at a glance; the per-call
traces are what let the skew be diagnosed rather than merely noticed. From that
one field the diagnosis was a short chain of checks against data the tooling had
already captured — compare the prediction histogram to the label distribution
(revealing the 80.5% majority baseline), recompute accuracy *per class* from the
stored prediction/truth pairs (LOF 0.93 vs GOF 0.52, DN 0.33), and read the
model's own rationales in the Langfuse traces for the misclassified GOF/DN cases
(which showed it anchoring on the one structural feature in the prompt, ΔΔG,
instead of reasoning about mechanism). The fix follows directly: stop trusting a
single aggregate metric on imbalanced data, make a class-balanced macro score and
the prediction histogram the primary verdict, report the always-majority baseline
beside any accuracy, and promote the histogram skew to an automatic guardrail
that fails a run loudly instead of letting a flattering number through.

## System under test

- **Task:** per-variant mechanism classification. For each variant the model is
  shown the gene, the substitution (e.g. `A216D`) and its FoldX ΔΔG, and must
  return exactly one of `GOF` / `DN` / `LOF`.
- **Model:** `claude-haiku-4-5`, called through the Anthropic SDK with a *forced
  tool call* (`tool_choice` pinned to a `report_mechanism` tool whose schema
  `enum`s the three classes) so the output is always structured.
- **Orchestration:** a batched runner fans the judge out across 8 concurrent
  workers, streams each prediction to a JSONL file as it completes, and writes a
  scored summary at the end.
- **Observability:** every model call is wrapped in the Langfuse `@observe`
  decorator, so latency, token usage, retry count, the predicted class and the
  ground-truth label are recorded per call and viewable per trace. The batch
  summary additionally rolls up a `prediction_counts` field — the count of
  predictions per class.

## Symptoms observed

The run completed cleanly (exit 0, no exceptions, no structured-output
failures). The summary read:

| Metric | Value |
| --- | --- |
| Accuracy | 0.835 |
| Scored | 200 / 200 |
| Structured-output failures | 0 |
| Retries | 0 |
| Predictions: GOF / DN / LOF | 18 / 12 / **170** |
| Latency p50 / p95 | 1.37 s / 2.35 s |

The accuracy alone reads as a success. The prediction distribution does not: 170
of 200 calls returned LOF, far more than any plausible balanced classifier would
produce. That single field was the first sign something was wrong.

## Diagnosis and the tools used

1. **The summary's `prediction_counts` field flagged the skew immediately.**
   Without it, 0.835 would have looked like a working judge and shipped. The
   per-class count is what made the collapse visible at a glance.

2. **Compared against the ground-truth distribution.** The 200-variant sample is
   161 / 200 = **80.5% LOF**. An always-predict-LOF constant classifier therefore
   scores 0.805. The judge's 0.835 is a three-point margin over predicting the
   most common class — almost no real signal.

3. **Computed per-class recall from the prediction JSONL.** Aggregate accuracy
   hides per-class behaviour, so recall was recomputed per class:

   | True class | n | Recall |
   | --- | --- | --- |
   | LOF | 161 | 0.93 |
   | GOF | 21 | 0.52 |
   | DN | 18 | 0.33 |

   The model only does well on the dominant class. On GOF it is barely better
   than a coin flip; on DN it is worse than chance for three classes. The
   confusion is directional: of 18 true-DN variants, 10 were called LOF; of 21
   true-GOF variants, 10 were called LOF. Errors flow *into* LOF.

4. **Read individual Langfuse traces for the misclassified minority cases.** The
   per-call rationales showed the model defaulting to "destabilizing → loss of
   function" whenever ΔΔG was positive, ignoring mechanism-specific cues. The
   traces also confirmed `n_structured_failures = 0` — this was a reasoning
   failure, not a parsing or tool-call bug.

## Root cause

Two compounding causes, both hidden by the headline metric:

1. **Class imbalance made global accuracy uninformative.** When one class is 80%
   of the data, accuracy rewards predicting that class and is nearly
   indistinguishable from the always-majority baseline. The metric measured the
   prior, not the model.

2. **The prompt fed the model a single feature (FoldX ΔΔG) that correlates with
   LOF.** A positive ΔΔG means the variant destabilizes the protein, which the
   model reads as "loss of function." It anchored on that one feature and
   collapsed the minority classes (GOF, DN) into LOF — destabilization is common
   to many mechanisms, so it is a weak basis for telling them apart.

Neither cause was visible from accuracy alone. The collapse only surfaced because
the prediction distribution and per-call traces were recorded.

## Permanent fix

The durable fix is to the **metric and observability**, not the prompt:

- **Demote global accuracy; report per-class recall and a macro-averaged score as
  the primary verdict.** Macro-recall weights each class equally, so a
  majority-class collapse can no longer hide behind a high aggregate number.
- **Always report the always-majority baseline next to the metric**, so any score
  is read as a margin over predicting the prior rather than as an absolute.
- **Turn `prediction_counts` into a guardrail.** Any run where a predicted class
  exceeds its true frequency beyond a set margin is flagged as a likely collapse,
  so a degenerate judge fails loudly instead of reporting a flattering accuracy.

A follow-up on the prompt side (remove the ΔΔG anchor, add explicit class
definitions and balanced few-shot exemplars, then re-measure macro-recall) is
worth doing, but the lasting lesson is the metric/observability change — the
prompt will keep drifting; the guardrail is what catches the next collapse.

## Generalized lesson

On an imbalanced eval, a high aggregate score can be indistinguishable from
predicting the prior. Always log the per-class prediction distribution next to
the headline metric and compare against the majority-class baseline. The
distribution — not the accuracy — is what reveals a degenerate judge.

## Provenance

- Code: `src/esm2_mech/experiments/llm_judge/judge.py` (traced judge),
  `src/esm2_mech/experiments/llm_judge/run.py` (batched runner).
- Results: `results/run6/llm_judge/summary.json`,
  `results/run6/llm_judge/predictions.jsonl`.
- Run parameters: 200 variants, sampling seed 0, model `claude-haiku-4-5`,
  concurrency 8. Ground-truth labels are the `label_3class` field of
  `data/variants.json`.
- Traces: Langfuse US Cloud (`https://us.cloud.langfuse.com`), one trace per
  `judge_variant` call.
- Always-LOF baseline: 161 / 200 = 0.805. Judge accuracy: 167 / 200 = 0.835.
  Per-class recall: LOF 0.93, GOF 0.52, DN 0.33.

## Summary (recap)

The system here is an evaluation harness that uses an LLM as a "judge": for each
genetic variant it is given the gene, the amino-acid substitution, and one
structural feature (a computed protein-stability score), and it must predict
which of three disease mechanisms the variant acts through — loss of function
(LOF), gain of function (GOF), or dominant-negative (DN). The harness runs the
LLM over a batch of variants whose true mechanism is already known and scores the
predictions, so it is a benchmark of the LLM, not of any model in the wider
project.

On a batch of 200 labelled variants the judge reported **83.5% accuracy**, which
looked like a working system. It was not. The model had quietly collapsed onto
the single most common answer — **170 of its 200 predictions were LOF** — and the
high accuracy was an artifact of the dataset itself being 80.5% LOF, so that a
trivial classifier which *always* answers LOF already scores 80.5%. The judge was
beating that baseline by only three points and was near-random on the two
minority classes (GOF, DN), which are the scientifically interesting ones. The
collapse was invisible in the headline metric and only surfaced because of the
specific observability the harness was built with: the batch runner does not stop
at accuracy, it also emits a `prediction_counts` field — a histogram of how many
times each class was predicted — and every individual model call is wrapped in
Langfuse's `@observe` decorator, which captures the predicted class, the true
label, latency, retry count and token usage as a structured trace. The
`prediction_counts` histogram is what flagged the skew at a glance; the per-call
traces are what let the skew be diagnosed rather than merely noticed. From that
one field the diagnosis was a short chain of checks against data the tooling had
already captured — compare the prediction histogram to the label distribution
(revealing the 80.5% majority baseline), recompute accuracy *per class* from the
stored prediction/truth pairs (LOF 0.93 vs GOF 0.52, DN 0.33), and read the
model's own rationales in the Langfuse traces for the misclassified GOF/DN cases
(which showed it anchoring on the one structural feature in the prompt, ΔΔG,
instead of reasoning about mechanism). The fix follows directly: stop trusting a
single aggregate metric on imbalanced data, make a class-balanced macro score and
the prediction histogram the primary verdict, report the always-majority baseline
beside any accuracy, and promote the histogram skew to an automatic guardrail
that fails a run loudly instead of letting a flattering number through.
