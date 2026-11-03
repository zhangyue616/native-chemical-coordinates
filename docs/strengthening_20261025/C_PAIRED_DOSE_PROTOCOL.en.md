# C: Finite-segment scalar dose intervention with fixed artificial pairing

> English reader translation of [`C_PAIRED_DOSE_PROTOCOL.md`](C_PAIRED_DOSE_PROTOCOL.md), added for this public release. It is not a study record: the Chinese file is the record of this protocol and governs if the two differ.

2026-10-25. Post hoc supplementary evidence, cleared under `DESIGN_DECISION.md`. The historical results, research choices and unequal-dose phenomenon are all known; this file is locked before the new intervention outputs of this round.

## Fixed inputs and selection

Use the three objects 24/25/26 of the original `MECHANISM_D_PLAN.json` and their respective selected+5 controls, the anonymous q124/q125/q115 and the D means. All 200 positive examples/200 negative examples of each object's original `MECHANISM_<id>.npz` are retained. The two strata are each sorted by SHA256(`eighth-dose-pair-v1-20261025:`+decimal CID), CID and original T index, and paired one-to-one; pairing must not be optimized by activations or outputs.

Compute full-dose using the existing T float32 post-GELU atom_a, the original atoms_T offsets and the D outgoing-column norm. The D mean uses its float32-representable value in the actual float32 model; displacement/RMS and the weight norm are computed in float64, and these two precision conventions are recorded accurately. The minimum of the 12 full-dose values of each pair is tau, the three s are 0.25/0.5/1, and alpha=s*tau/d; if d is 0, tau must be 0, alpha is set to 0 and the whole pair is retained. Epsilon denominators are prohibited. The pairing, tau, all alpha, source bindings and parameter SHAs must be written to disk first, before any model output of this round begins.

## Real forward pass and numerical verification

Batches are fixed at 4 molecules, in the interleaved positive/negative order of the pairs; all baselines and conditions use the same batches/right padding. The original official MTR wrapper is loaded; the actual RobertaIntermediate output of the original penultimate block is the intervention site, and only one coordinate of the atom anchors within the mask changes. Alpha is broadcast independently per molecule; the interpolation is computed in float64 and then cast back to the original float32 model; the original outputs and the non-intervened coordinates/non-atom positions are unchanged. The real output returns normalized anonymous q through the original head.

A single small-batch engineering prerequisite covers baseline/alpha0 recovery, the real hook position and atom-only scope, the alpha range, numerical differences from the old cache q/activations and the actual dose. All baseline and engineering samples count toward this round's forward count and CPU, and must not be written to the old ledgers. The actual projected dose is computed as the float64 RMS of the model-consumed float32 change relative to the original activations, multiplied by the norm; the error against the target must satisfy max absolute<=1e-6*(1+max target), and the relative error for target>1e-8 is additionally reported. On failure, the failure is retained and only the implementation/numerics may be fixed; pair/tau/s or objects must not be changed to rescue results.

Planned: 21600 intervention evaluations, plus 1200 (400 per object) baseline and small-batch engineering checks, total hard cap 24000; the cumulative CPU of the entire actual execution process of C must be <=3600 seconds (including statistics, using the more conservative accounting), with at most 4 scientific threads. Each completed condition is persisted; budget/numerical/source-conflict failures retain partial output; the budget must not be expanded without authorization and old runs must not be reopened.

## Estimation and disclosure

For each s, compute the selected positive examples' mean absolute output change minus the negative examples' mean, then subtract the median of the same difference across the five controls. The primary metric H_pair(s=1) uses the 200 artificial pairs as clusters, PCG64 seed20261025, 2000 resamplings with replacement, common weights across all positives/negatives and six coordinates, and a fixed 3-object 98.3333333333% percentile interval. s=.25/.5 are descriptive trajectories only, and no new significance may be added. The old H and its intervals are a historical control only, and joint six-interval 95% coverage must not be claimed.

Retain all signed/absolute responses, the 200pair tau and alpha, quantiles, true-zero/tau<1e-4 proportions, the positive/negative molecules and coordinates that limit tau (all tied minima are retained), and the actual target errors. Zero-dose pairs must not be removed. Missing values/numerical failures are reported completely; results conditional on these artificial pairs and the minimum dose must not be written up as a general mechanism, an equal-vector perturbation or a dose-explained proportion. An interval containing 0 does not prove equivalence.
