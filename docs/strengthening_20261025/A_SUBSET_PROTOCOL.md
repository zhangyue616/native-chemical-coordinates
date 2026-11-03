# A3：固定 D-unseen 非空 scaffold T 切片的事后敏感性协议

2026-10-25。本文件在读取新切片分数/估计前写入。新结果单独保存于results/strengthening_20261025/scaffold_subset/，原科学文件、旧锁和旧CPU账本保持只读。

## 对象与统计定义

- 使用谱系 lane 独立按结构冻结的 `results/strengthening_20261025/lineage/T_D_UNSEEN_SCAFFOLD_MASK.npy` 及对应 `FREEZE.json`。保留 T 中非空 Bemis–Murcko scaffold 且该 scaffold 未在 D 出现的全部分子，不按分数、标签支持或效果更改 mask。冻结 mask 的 T 行序必须与原 `sealed/T.tsv` 一致。
- 保留原 `D_FINAL_WINNERS.npz` 的全部 71 概念、4 网络与 10 基底（native + 原 9 个 Q）。不重新选择、不读取 T oracle，不新增 PLM forward。原 labels/atom mapping、T `atom_a.npy` 及 Q 为唯一主统计输入。
- 分数保持 float32；Q 从原保存 float64 文件 cast 为 float32。旋转在原完整 atom 行的 2048-row block 上执行原 `native_block @ Q32` 后才提取所需行，避免因先稀疏抽行而更改 BLAS block 形状。exact float32 equality 定义 ties。AP 用原 `chem_statistics.fixed_winner_ap`；prevalence、`ell=log(AP/pi)` 及 `estimands` 均复用原纯函数和原聚合顺序：概念内先求旋转/初始化中位数，最后等权概念均值。
- 子集的 20 positive molecules / 100 all-negative molecules / 50 positive atoms 仅是原完整 T 的历史支持诊断参考，绝不用于新切片分子/概念筛选，也不阻止数学定义良好的估计。逐概念保存正/负分子、正/负原子、prevalence、历史参考不足原因。全正概念的原 AP 可为 1、pi 为 1、ell 为 0，同时标明缺少负标签支持。
- 任一固定概念 pi=0 或 AP/ell 不可定义时，完整 71 概念汇总记为 `NOT_ESTIMABLE`，不改集合、不对幸存概念重新归一或另立汇总；其它概念描述点仍可保留。不以支持诊断作新确认 PASS。

## 新共享分子 bootstrap

- slice 有 m 个分子，每 replicate 独立从其中有放回抽 m 次，每个被抽分子的全部原子一起携带。2000 replicates，PCG64（NumPy `default_rng`），固定新 seed `20261025`。
- 直接在 slice 的 m 分子空间调用原 `cluster_weights(m,2000,20261025)` 并保存新 `SLICE_BOOTSTRAP_WEIGHTS.npy`。不截取旧 10000 分子权重。每行权重和必须恰为 m。
- 全部网络/基底/概念与 12-panel 使用同一权重矩阵。每次重算 weighted AP、pi、原中位数与均值。点估计另用全 1 权重。
- 若完整集合点可定义，执行完整 2000 bootstrap；逐概念与 joint 报 undefined 数量/比例。95% percentile CI 用 NumPy linear quantile，对 joint 有限 replicates 计算；保留原 <=1% undefined 稳定性诊断。该诊断不是新的科学门或确认性检验；CI 是依赖固定训练 checkpoint、3 初始化、9 Q 和 D 选择的事后敏感性区间。不新增 p 值/确认 PASS。
- 若固定集合点不可定义，只完成完整集合点/支持记录与 12-panel Brier；不为不可定义汇总制造新的 bootstrap 数值。

## 原 12-panel Brier

- 按原 `D_SECONDARY_FREEZE.json` 的 12 个概念顺序读取原 `predictions_T.npy[atoms,12,4]`（token/r0/L/Lj），严格同一 slice 原子。无新 fit，不改 bins/fallback。
- 原子 pooled Brier 每概念计算；DeltaBrier = Brier(L)-Brier(Lj)，panel 为 12 概念等权均值。复用原 `panel_brier`，以上述同一 2000 slice 分子权重求 panel 95% CI。
- 报 12 概念 Brier L、Lj、Delta、prevalence、正/负标签支持。零标签概念仍报告 Brier，区分 AP 不可定义与 Brier 可定义。

## 执行预算与保留

- 此 lane 累计 `process_time` 硬上限 9000 CPU 秒，4 线程，建议 RSS <=2 GiB；A/B总预算合并记录。本 lane 独立 `CPU_LEDGER.json`，含运行、失败和恢复记录，绝不调用旧 Budget。各 network/basis 独立子进程，复用确定性 checkpoint；首单元报告实际 CPU 和剩余成本估算。
- 一次小型 exact-tie/cluster/Brier 实现检查后运行；阶段末检查最终整合结果，不重复旧全量goldens。每个已完成单元保留 AP/ell 及 identity；预算或错误停止保留 partial/failure，不缩概念/样本/replicates、换 seed 或更改定义。
- 输出含 mask 绑定、支持 CSV/JSON、新权重、40 单元 checkpoint、完整 AP/ell/estimands/undefined、12-panel CSV/JSON/bootstrap、独立 CPU ledger 与 failure。只对必要输入/实现和完成 checkpoint 做 identity 绑定，不重哈希全部大型缓存。
