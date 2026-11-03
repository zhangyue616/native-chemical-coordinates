# A3: Post hoc sensitivity protocol for the fixed D-unseen non-empty scaffold T slice

> English reader translation of [`A_SUBSET_PROTOCOL.md`](A_SUBSET_PROTOCOL.md), added for this public release. It is not a study record: the Chinese file is the record of this protocol and governs if the two differ.

2026-10-25. This file is written before the scores/estimates of the new slice are read. New results are saved separately in results/strengthening_20261025/scaffold_subset/; the original scientific files, old locks and old CPU ledgers remain read-only.

## Objects and statistical definitions

- Use `results/strengthening_20261025/lineage/T_D_UNSEEN_SCAFFOLD_MASK.npy`, frozen independently by structure in the lineage lane, and the corresponding `FREEZE.json`. All molecules in T that have a non-empty Bemis–Murcko scaffold not appearing in D are retained; the mask must not be changed according to scores, label support or effects. The T row order of the frozen mask must be consistent with the original `sealed/T.tsv`.
- All 71 concepts, 4 networks and 10 bases (native + the original 9 Q) of the original `D_FINAL_WINNERS.npz` are retained. There must be no reselection, no reading of the T oracle and no new PLM forward. The original labels/atom mapping, T `atom_a.npy` and Q are the only primary statistical inputs.
- Scores remain float32; Q is cast to float32 from the originally saved float64 files. Rotation executes the original `native_block @ Q32` on 2048-row blocks of the original complete atom rows, and only then are the required rows extracted, to avoid changing the BLAS block shape by sparsely sampling rows first. Exact float32 equality defines ties. AP uses the original `chem_statistics.fixed_winner_ap`; prevalence, `ell=log(AP/pi)` and `estimands` all reuse the original pure functions and the original aggregation order: within each concept, first the rotation/initialization medians, and finally the equal-weight mean over concepts.
- For the subset, the 20 positive molecules / 100 all-negative molecules / 50 positive atoms are only a historical support-diagnostic reference from the original complete T; they must never be used to screen molecules/concepts in the new slice, and they must not block mathematically well-defined estimates. Positive/negative molecules, positive/negative atoms, prevalence and the reasons for falling short of the historical reference are saved per concept. An all-positive concept's original AP may be 1, with pi being 1 and ell being 0, while the lack of negative-label support is flagged.
- If any fixed concept has pi=0 or an undefined AP/ell, the complete 71-concept summary is recorded as `NOT_ESTIMABLE`; the set must not be changed, and the surviving concepts must not be renormalized or given a separate summary; the descriptive points of the other concepts may still be retained. Support diagnostics must not be used as a new confirmation PASS.

## New shared-molecule bootstrap

- The slice has m molecules; each replicate independently draws m times from them with replacement, and all atoms of each drawn molecule are carried together. 2000 replicates, PCG64 (NumPy `default_rng`), fixed new seed `20261025`.
- Call the original `cluster_weights(m,2000,20261025)` directly in the slice's m-molecule space and save the new `SLICE_BOOTSTRAP_WEIGHTS.npy`. The old 10000-molecule weights must not be truncated. The weights in each row must sum to exactly m.
- All networks/bases/concepts and the 12-panel use the same weight matrix. Weighted AP, pi, and the original medians and means are recomputed each time. The point estimate additionally uses all-1 weights.
- If the complete-set point is defined, execute the complete 2000 bootstrap; report the number/proportion undefined per concept and jointly. The 95% percentile CI uses NumPy linear quantile, computed over the jointly finite replicates; the original <=1% undefined stability diagnostic is retained. This diagnostic is not a new scientific gate or confirmatory test; the CI is a post hoc sensitivity interval that depends on the fixed training checkpoint, 3 initializations, 9 Q and the D selection. No new p value/confirmation PASS may be added.
- If the fixed-set point is undefined, only the complete-set point/support records and the 12-panel Brier are completed; new bootstrap values must not be manufactured for an undefined summary.

## Original 12-panel Brier

- Read the original `predictions_T.npy[atoms,12,4]` (token/r0/L/Lj) in the order of the 12 concepts in the original `D_SECONDARY_FREEZE.json`, on strictly the same slice atoms. There is no new fit, and bins/fallback must not be changed.
- Atom-pooled Brier is computed per concept; DeltaBrier = Brier(L)-Brier(Lj), and the panel is the equal-weight mean of the 12 concepts. The original `panel_brier` is reused, and the panel 95% CI is computed with the same 2000 slice molecule weights described above.
- Report the Brier L, Lj, Delta, prevalence and positive/negative label support of the 12 concepts. Zero-label concepts still report Brier, distinguishing AP being undefined from Brier being defined.

## Execution budget and retention

- This lane's cumulative `process_time` hard cap is 9000 CPU seconds, with 4 threads and a recommended RSS <=2 GiB; the total A/B budget is recorded jointly. This lane has its own independent `CPU_LEDGER.json`, containing run, failure and resume records, and must never call the old Budget. Each network/basis is an independent subprocess that reuses deterministic checkpoints; the first unit reports the actual CPU and an estimate of the remaining cost.
- Run after one small exact-tie/cluster/Brier implementation check; at the end of the stage, check the final integrated results, without repeating the old full goldens. Each completed unit retains AP/ell and identity; a budget or error stop retains partial/failure; concepts/samples/replicates must not be reduced, the seed must not be changed and definitions must not be altered.
- Outputs include the mask binding, support CSV/JSON, new weights, 40-unit checkpoints, complete AP/ell/estimands/undefined, 12-panel CSV/JSON/bootstrap, an independent CPU ledger and failure. Identity binding is performed only for necessary inputs/implementations and completed checkpoints, and must not rehash all large caches.
