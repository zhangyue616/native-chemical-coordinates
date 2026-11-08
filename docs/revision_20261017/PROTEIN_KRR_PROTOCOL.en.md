# Fixed j1246: contract for unified KRR development and conditional long-protein confirmation

> English reader translation of [`PROTEIN_KRR_PROTOCOL.md`](PROTEIN_KRR_PROTOCOL.md), added for this public release. It is not a study record: the Chinese file is the record of this protocol and governs if the two differ.

2026-10-17. Implementation contract, overriding the inconsistent readout and normalization recommendations in the two design proposals. This file releases only the offline preparation and synthetic checks; real D fitting requires the research operator to issue a `D_GO` that binds `FINAL_READY.json`. This script has no network, model loading, forward, or new T entry point. The original P1–P5 files and conclusions are frozen.

## 1. Task, inputs, and scientific boundaries

The existing ESM2 8M, the third block (0-based 2) post-GELU, the original j1246, and the original H are fixed. H = (`−1` in `AILMFPVWY`) AND (`+2` in `AILMFPVWY`); j/H, the model, the initialization, or the layer must not be reselected, and candidate hypotheses must not be redesigned. H is a deterministic second-order AND of S; its expressiveness does not represent added sequence information.

D is explicitly composed of all 176 pairs of P3, the old T 128 pairs of P4, and the old T 128 pairs of P5, a total of 432 pairs and 864 sites; the former T has entered a new development role and must not again be called fresh T. Records come from `data/pilot{3,4,5}a/proteins.jsonl`. For each `case_id`, an `id/component/complete sequence/q/label/y` consistency check is done against the historical P5 `D_CASES` (which contains the complete P3+P4, including P3's original C16) and P5 `T_CASES`; concatenation by file position must not be used. j is joined by `case_ids` from P3 `D_NATIVE/C_NATIVE/T_NATIVE.npz` and P4 and P5 `T_NATIVE.npz`; after the complete 1280-dimensional shape and finite values are checked, only 1246 is extracted.

The sole default input for the current global group mapping: `selected_pair_group_mapping` of `data/revision_20261017/protein_metadata/OLD_SELECTED_UNION_GROUPS.json`, joined by `(origin,id)`, field `current_known_union_group`. Currently there are 432 pairs and 421 known global groups; new metadata produced bridges, so folds must not be split independently along the old 432 connected components. Future exact-sequence may merge groups further: the script accepts `--groups <finalgroup file> --group-field <final field>`; each old pair must have exactly one mapping. After an update, prepare again/reissue D_GO; if fitting has already been done, the research operator must separately set update authority targeting the affected folds, and automatic rerun must not occur. Known groups do not guarantee remote-homology independence or the absence of pretraining overlap.

The target confirmation population is long proteins that are reviewed+PDB with length 257–512; this is a **short-D→long-T transfer**. At present, the 2074 metadata connected components after exhausting the 13658 raw records of the new source are only an upper bound before sequences/labels, not a sufficient final new cohort. D contains no new long-protein output, and D variance does not guarantee long-T precision or power. Insufficient source and infeasible precision must both be reported as failure; N must not be lowered to pass it off as completion.

## 2. Features, pairing, and strict nesting

Each pair is a/b in ascending q order; a is not guaranteed to be HELIX; `y_a=1{a is HELIX}`, `t=2y_a−1`. No reordering by label, no mirror rows, no intercept. X follows the old 64 dimensions: the 20AA composition of global (excluding the center), ±7, and ±15, plus `q/(L−1)`, its square, `L/256`, and the available-position fraction within ±15. Even if L>256 in the future, `/256` is retained. P follows the old S local 630 dimensions: 30 ordered offsets outside the center (−15…−1,+1…+15), each one-hot in 21 states (20AA+PAD); therefore S still has the 694-dimensional feature semantics of X+P.

All models use the same KRR, D, four folds, loss, and fixed lambda candidates. Feature blocks have no per-coordinate standardization, RMS, or centering; the per-block trace normalization below is the only scale transformation. This contract replaces the overall S/694 and the old logistic per-coordinate RMS in the design proposals, to avoid changing the weight of X in S/L.

For any site kernel k, the paired kernel is:

`B_k(c,d)=k(a,a′)−k(a,b′)−k(b,a′)+k(b,b′)`.

The four raw PSD blocks:

- `B_X(c,d)=ΔX_c·ΔX_d`.
- `B_P`: the four terms above applied to the local one-hot inner product; the site inner product equals m, the number of category matches at the same position.
- `B_Q`: the four terms above applied to the complete second-order site kernel `k2(v,w)=m(m−1)/(30×29)`. It corresponds to the interaction inner product over all distinct position pairs and all category combinations, divided by 435, and contains the expression space of H. There is no position/interaction selection.
- `B_j(c,d)=Δz1246_c Δz1246_d`, the scalar being the existing native activation.

For each training fold I, let `r_b(I)=trace(B_b[I,I])/|I|`. If r>0 then `K_b=B_b/r`; a zero-trace block is defined as zero (no added jitter, substitute unit, or search), including constant j. r is obtained only from training pairs; validation/T must not update it. Each model adds blocks with equal weight:

`K_X=K_Xblock`; `K_Xj=K_Xblock+K_j`;

`K_S=K_Xblock+K_P`; `K_Sj=K_Xblock+K_P+K_j`;

`K_L=K_Xblock+K_P+K_Q`; `K_Lj=K_Xblock+K_P+K_Q+K_j`.

Therefore the X block has exactly the same scale in X/S/L, and the P block has the same scale in S/L. The kernels must not be further divided by total dimension, number of blocks, or overall model trace. What is nested is the kernel/function family; the performance difference after independent lambda selection is not a fixed-penalty ablation or mechanistic mediation.

## 3. Fitting, folds, selection, and budget

The objective is `mean_pair(t−f)^2 + λ||f||²_RKHS`, and the solution is `α=(K+n_pair λI)⁻¹t`, float64 Cholesky, with no optimization restarts. The numerical residual must be ≤1e−8. In training each pair has equal weight, i.e., a group that contains multiple old pairs after group merging contributes multiple training rows; this is an explicit training objective and does not treat them statistically as independent samples.

The same group must not span folds. The sole seed is 2026101708: the unique groups are sorted by `SHA256("protein-krr-cv:{group}:2026101708")` and group name, rank mod4; changes in row order do not change the fold assignment. The current 421 groups are split into 106/105/105/105 groups; the number of pairs in each fold is recorded separately in ready. If group identities merge in the future, replacing the mapping updates the corresponding folds and training scales; the same group must not be kept across folds.

All six models use `{0.0001,0.001,0.01,0.1}`, with four folds per lambda. This grid was corrected by the research operator before any real D activation read or fit: currently each block has `trace/n=1` and the solution is `K+nλI`, while the old `{0.1,1,10,100}` values belong to the old sum-SSE/feature-scale contract; carrying them over directly would systematically and strongly shrink the high-dimensional P/Q (at n about 400, the minimum diagonal penalty is about 40). The new four values cover weak to strong regularization matched to the current trace, keep four candidates, four folds, and 102 fits, and are not adjusted based on any new D/T results.

`f=K_valid,train α`; `p_a=clip((1+f)/2,0,1)`, `p_b=1−p_a`. The paired Brier is `(p_a−y_a)^2`, exactly equal to the average of the two complementary site losses. When a/b are swapped, f changes sign, the probabilities are complementary, and the loss is unchanged. OOF selection first averages the Brier of all pairs in each group, then averages with equal weight across groups; an exact tie takes the larger lambda. Each model selects lambda independently and is then fit once on all 432 pairs.

Each model has `4λ×4fold+1final=17` fits, for a total of **102 real KRR fits**; there is no separate logistic bridge, no full-Z or common-lambda sensitivity, and no new H→j regression. H→j beyond S may be a secondary estimate under a separate pre-lock (two ridge regressions on the same folds, at most 34 fits); this implementation does not currently run it, it is not counted in the 102, and the old H regression must not be used as a substitute for new evidence beyond S. New HTTP/model-load/forward for D are all 0. The readout-fitting CPU limit is 3600 seconds; failures are entered in the ledger, with no automatic restart/grid expansion.

## 4. D precision gate and the fixed four-term family for future T

Per-pair loss is computed from the OOF probabilities at each model's own D-selected lambda, and then the fixed column order `(gX,gS,gL,psi)` is formed:

`gX=ell(X)−ell(Xj)`; `gS=ell(S)−ell(Sj)`;

`gL=ell(L)−ell(Lj)`; `psi=gX−gS`.

The four columns are first averaged within known global groups, and groups are then weighted equally. The means on D serve only as development description; **D confirmation p-values must not be computed or reported**. This scheme is not nested CV: the same OOF is used for lambda selection and variance planning, and the development optimism must be disclosed. The operational precision bound is Brier .01, not a proven biological minimum effect.

For the sample SD (ddof=1) of the four-term group means, use the conservative inflation 1.25 and the fixed Bonferroni four-term simultaneous 95% approximate constant z=2.4977:

`N_req=ceil[(2.4977×1.25×max(SD_gX,SD_gS,SD_gL,SD_psi)/.01)^2]`.

`target=max(512,N_req)`, hard cap 1536; no additional rounding to a multiple of 32. If N_req>1536, `PRECISION_INFEASIBLE_WITHIN_CAP`, the compliant target is null, and the real requested_target is retained; the target must not be truncated to 1536 and claimed as passing. If ≤1536, it is only `PRECISION_TARGET_FEASIBLE_PENDING_FRESH_SOURCE`, and the sufficient-source gate after sequence/label/old-pool exclusion is still required. 2074 is only an upper bound and must not replace the actual gate. The planning CI half-width is not an 80% power commitment, and the short-D and long-T distributions may also change the variance.

On the same future fresh T, the four-term family is retained: gX>0, gS>0, gL>0, psi two-sided ≠0, Holm fixed at four terms, alpha=.05. Untestable terms (missing, non-finite, zero SD, etc.) keep p=1 and are not removed from the denominator. Groups are the unit of inference, and multiple pairs within a group are averaged first. Report means, raw per-pair/per-group contributions, SD/SE, uncorrected p and Holm p, 95% estimate CI, and the Bonferroni per-term 98.75% two-sided t reference interval (four terms simultaneously ≥95% approximate). The t reference and the residual-homology limitation must be explicitly labeled; no bootstrap may be added, and sides must not be switched according to results. The code's future-inference helper does not run on D.

psi>0 indicates only that the increment between the predetermined independently tuned pipelines is smaller; only if gX also supports a positive increment may a shrinking of positive gain be discussed. Non-support of gS/gL does not equal equivalence to no information. L covers only the complete second-order local family and does not represent all nonlinearity or all sequence information.

## 5. Offline preparation, gating, and limited acceptance

Use the existing `.venv/Scripts/python.exe`; the environment must not be changed. Write only the new scripts, tests, this protocol, and `results/revision_20261017/protein_D/`. First run `protein_krr.py selftest`, which uses only synthetic data to verify the complete second-order explicit features/PSD/nesting, pair swap, constant j, no leakage in training-fold scales, the mean-loss solution, center exclusion/PAD, group fold assignment, precision over cap, and the fixed family. Then `protein_krr.py prepare` performs the old metadata join and the current group fold assignment, without reading activation values or doing real fitting.

`FINAL_READY.json` binds only this round's code, protocol, group file, new input manifest, and synthetic acceptance receipt; the old data/old acceptance are reused, and the old models or the hashes of all old files must not be recomputed. The pre-D grid correction reuses the original 9 items of kernel/solver/grouping evidence, uses `GRID_REVISION_CHECK.json` to separately verify the strict ordering of the new grid, the four candidates, and that the actual production selection function takes the larger lambda on an exact tie, and binds the new script/old SELFTEST, without redoing the metadata join or all selftests. The research operator's D_GO JSON must contain `status="D_GO"`, `D_only_authorized=true`, `T_authorized=false`, `maximum_real_fits=102`, `ready_sha256=<SHA256 of FINAL_READY>`. The real execution is `protein_krr.py develop --d-go <research operator receipt>`. After D fitting starts, automatic rerun is refused; future group changes must have separately set authority and targeted scope.

Main artifacts: `D_INPUT_MANIFEST.json` (complete old cases and final fold assignment), `FIT_LEDGER.json` (ledger of the 102 fits), `D_OOF.npz` (all lambda/model OOF and four-term contributions), `FINAL_MODELS.npz` (final alpha, training-pair features, per-block D trace, lambda), `PRECISION_PLAN.json`, `D_DEVELOPMENT.json`. `predict_saved` only provides array predictions for reuse by a separately authorized T evaluator; it does not load PLM and does not accept CLI T. The specific cohort, source gate, extraction, and confirmation lock for new T are not authorized by this contract.
