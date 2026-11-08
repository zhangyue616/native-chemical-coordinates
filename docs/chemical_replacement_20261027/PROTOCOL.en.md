# Two-coordinate chemical replacement protocol

> English reader translation of [`PROTOCOL.md`](PROTOCOL.md), added for this public release. It is not a study record: the Chinese file is the record of this protocol and governs if the two differ.

2026-10-27. The scientific definitions are locked at the first save of this file; the run budget is written to `RESOURCE_LOCK.json` after a baseline pre-check on the first 16 molecules of the independent D, and new T conditions are permitted only when both are in place. This is an added post hoc analysis on a T that has already been analyzed; it is not preregistration or independent confirmation.


## Fixed objects and inputs

- DeepChem/ChemBERTa-77M-MTR revision `66b895cab8adebea0cb59a8effa66b2020f204ca`, the original official checkpoint and head; block index 1, actual FFN post-GELU, the original normalized anonymous 199-dimensional output.
- Primary object j293, concept 26, q115; predefined comparison j454, concept 24, q124. q125 of j454 is a secondary description and must not be treated as a third coordinate.
- The original D5000 molecules/124530 heavy atoms are used only for fitting; the T10000 molecules/246722 heavy atoms are all evaluated. The original molecule order, graph atom order, accepted anchors and the 91-concept dictionary are retained. Layers, coordinates, endpoints, features, samples or concepts must not be reselected.
- Cache metadata and shape/atom/molecule identity are checked once; the hashing of all old files must not be repeated. The new protocol, the D fit, the programs and the final products each retain one necessary binding. The strict load check of the original checkpoint reuses the model wrapper.

## D fitting and prediction

Each coordinate is fitted independently; all means use D atoms as the denominator and accumulate in float64.

1. D-mean: the all-atom D mean of this coordinate.
2. M1: by the binary target-atom label of this object, take the D positive and negative activation means separately; it must not be changed to an occurrence label. Both groups must be non-empty.
3. M2 = L_a: reuse the exact r0/r1 graph keys of `chem_input_reference.input_keys` and alpha=20. The global prior is the D activation mean; r0 is `(sum+20*global)/(n+20)`; r1 is `(sum+20*parent_r0)/(n+20)`; the T fallback is r1→r0→global. It is a new independent continuous regression function; it must not call the old binary fit_one and must not take as input the real a_j, its binning or the output q. token is not a fallback of L_a; no token replacement condition is added in this round. The number of categories, unseen r1/r0 and the actual fallback proportions are recorded.

The 20/40/60/80 percentiles of the D activations are computed once by the linear method, and all four cut points are saved as they are (equal cut points are allowed to produce an empty bin); T `searchsorted(side=right)` fixes five bins. Binning is used only for post hoc description of error and chemical composition and must not enter the model input. D_FIT.json saves the model, the two classes of means, the cut points and the D denominator; after saving it is bound and used for all of T, and must not be modified on the basis of T.

## Real forward and displacement permutation

Each coordinate has five fixed conditions, in the order `D_mean, M1, M2, M1_delta_perm, M2_delta_perm`; a total of 100000 condition-molecule evaluations. Counted separately are 10000 fresh baseline and the predefined 48-molecule sham. New conditions must not be inferred from old mean/alpha outputs.

T always runs in consecutive 4-molecule batches along the original order, with right padding, CPU float32, eval, torch threads=4/inter-op=1, and relevant BLAS threads=4. Each batch first computes the fresh baseline, then runs the five conditions for each of the two coordinates. The hook changes only the real atom anchors of this coordinate and keeps the other coordinates and special/PAD tokens. Anchors are explicitly indexed in graph atom order; it must not be assumed that the boolean mask order equals the graph order.

M1/M2 first generate per-atom target predictions; delta=prediction−actual is computed on the real activations of this forward. The perm condition applies a fixed permutation to delta across the atom positions of the same molecule and then adds it back to actual. The generator is NumPy PCG64, with SeedSequence integer list `[20261027, coordinate, global_T_molecule_index]`; M1 and M2 share the permutation of the molecule. Degenerate cases such as atom count 1 and constant displacement are retained and must not be redrawn. The number of identity permutations, the actual number of nonzero displacements and the consistency of the L2/empirical displacement set are recorded. The diagnostic preserves the displacement distribution and the L2 dose but changes the direction and the marginal of the replacement activation; it must not be called a strictly distribution-matched null.

The output saves the fresh baseline [10000,199] and the per-condition outputs [2,5,10000,199] (float32), and separately saves the per-molecule realized displacement RMS, the nontrivial-permutation flag and the necessary hook checks. Differences from the original network are always relative to the same-batch fresh baseline. progress is written each time a consecutive batch is completed; automatically overwriting completed products or automatically rerunning everything is prohibited.

## Pre-result fixed numerical gates

- fresh baseline against the original cache: q115/q124/q125 maximum absolute difference ≤2e-6; all 199 outputs ≤2e-5; selected j293/j454 atom activations ≤2e-5. The cumulative maximum difference over full T and its location are recorded; numerical gates must not be interpreted as scientific fidelity tolerances.
- no-op exact replay: at T batch start points 0, 4000, 8380, 9996, run one original-value write-back for each of the two coordinates; in addition, run a full-activation scope check for the two coordinates at start points 0/8380, a total of 48-molecule sham (32 replay +16 scope). no-op outputs must be equal value by value, and in the scope check other coordinates/non-anchors must be unchanged value by value. The scope check uses specified targets from the existing real baseline activation and must not introduce new scientific conditions.
- All predictions, activations and outputs must be finite; offsets, graph atoms and the anchor bijection must be consistent; the condition count and completion index of each batch must be complete. The sorted values/L2 of the actual displacement and the permuted displacement are equal by float64 arithmetic definition; the error after float32 write-back is listed separately, atol=2e-5, rtol=2e-6, and gates must not be relaxed to rescue results.
- On failure, record the current batch/condition/endpoint and the raw difference, and stop; fix only the actual defect and run the unfinished part, and keep the failure count; scientific definitions must not be changed and the budget must not be automatically increased.

## Summaries and intervals

Output endpoints are summarized by molecule: MAE of the primary/comparison q, absolute-error 95th percentile (linear, point estimate), signed mean shift, mean normalized199-RMS. Denominators are reported for full T, for molecules positive at any target atom of this object, and for molecules negative at all target atoms. q125 is listed separately on the same basis as a secondary description.

The paired difference is defined as the MAE of this condition minus the D-mean MAE; M1/M2 additionally subtract their own delta-perm MAE; a negative value indicates smaller output error. The five conditions are fully reported, including results that do not support a semantic position effect or that show low sensitivity; recovery rates with near-zero denominators must not be computed.

Activation MSE and R²: for all of T and the above molecule strata; additionally reported by target-atom label 0/1 and the five D-fixed bins. The point estimate is the pooled MSE of the included atoms, R²=1−SSE/SST of the corresponding T atoms relative to their own mean; SST=0 is recorded as undefined. The atom count and independent molecule count of each group are reported, and atoms must not be treated as independent replicates. Five bins×all 91 concepts save the hit atom counts/proportions and the proportion not hitting the dictionary; labels may overlap, must not be treated as mutually exclusive components, and per-concept significance must not be assessed.

One common molecule bootstrap: 2000 replicates, PCG64 seed=20261028; each replicate draws 10000 with replacement from the full 10000 T molecules, fully carrying all their atoms and conditions; all objects/conditions/strata share the same set of draw counts. 95% percentile linear intervals; a stratum uses the molecules of that stratum drawn in that replicate; if undefined, the number must be reported and not hidden. Intervals are given for the MAE, signed shift and mean 199-RMS of the primary/comparison/secondary q, for the above paired MAE differences, and for the overall/molecule-stratum MSE/R² of the three base replacement activations; atom-label/five-bin errors and tail95 are descriptive point estimates. The intervals are marginal estimates, not joint 95% or formal multiple hypothesis tests; the 199 individual outputs must not be searched, and equivalence/complete fidelity must not be claimed.

## Single-stage acceptance and delivery

Save the D model, full-T predictions and forward outputs, molecule summaries, intervals, all-bin chemical composition and the run ledger. The fixed analysis introduces no new training or H4 rerun.
