# B：D/T post-GELU 尺度标准化旋转事后补证

日期：2026-10-25。状态：**POST_HOC_PROTOCOL_RECORDED_BEFORE_NEW_OUTPUTS**。本方案新增独立事后结果，不覆盖旧协议、数据、旋转、科学结果或账本。

## 固定对象与参数

- 网络顺序固定 `trained,init21000,init21001,init21002`；原 D=5000、T=10000 分子，原 atom_a 分别为 float32 `[124530,464]`、`[246722,464]`。
- 概念严格为原 `results/revision_20261017/chemistry_D/D_FINAL_WINNERS.npz` 的全部71个 concept_ids；不因结果/支持度减少。
- 原9个共同 Q 为 `results/revision_20261017/chemistry_D/rotations/Q30000.npy` 至 `Q30008.npy`，从原 float64 文件转 float32，禁止生成新 Q 或替换种子。
- 每网络复用自身 `data/revision_20261017/chemistry/activations_D/{network}/D_moments.npz` 的 `atom_mean` 和 `atom_sd`。已亲读 `chem_run.py:extract_D`，这些 moments 来自原保存 float32 atom_a 的分块 float64 sum/squares，`sd=sqrt(max(0,E[a²]-E[a]²))`，population ddof0；不再全量拟合/重算。
- `sd<=1e-12` 的坐标完整标记，缩放分母置1，仍保留全部464坐标；报告每网络具体索引/均值/原SD。参数只依赖D，固定应用T。

数值路径固定为：每块原 float32 激活转 float64，`(a-mu_D)/safe_sd_D` 后 cast float32，再以原 Q cast float32 做矩阵乘法，最终 scores 为 float32。中心化/缩放与float32量化可能改变exact ties；AP仍按最终float32的精确相等分组，绝不按epsilon合并分数。AP累加为float64。线程上限4，单进程，仅一个旋转 scratch mmap 在新目录内；不调用会写旧目录的 load_D/load_T/Budget/basis_matrix。

## D 重选与 T 门

native 四网络的原 D winner/AP、T point AP/ell/bootstrap ell 全部只读复用，不重算native AP或其bootstrap。每网络×每Q（36个单元）必须在完整D上调用原纯函数 `chem_statistics.select_winners`，搜索全部928带符号候选；winner tolerance固定 `8*np.finfo(float64).eps`，按unit升序、先正后负选择，候选编码 `+a_j=j;-a_j=464+j`。

每个完整D单元保存新checkpoint。全部36个完成后，保存 `[4,10,71]` 的 `D_STD_WINNERS.npz`（native来自旧winner，9Q来自新D选择）及 `D_STD_OUTPUT_LOCK.json`，绑定参数、输入、脚本和协议身份。**这个新D锁落盘后才读取T标签/权重/native bootstrap并计算新T旋转结果**。不使用旧raw旋转winner作为标准化winner，不在T重选。

## T AP、bootstrap 与汇总

复用只读原 `chemistry_T/T_BOOTSTRAP_WEIGHTS.npy`，恰为 int32 `[2000,10000]`、seed42017 的整分子shared count weights，不生成新的随机权重。点估计附加一行全1权重；对原T全部原子和71概念重新评价新D锁定旋转winner的AP，每个replicate重算prevalence和ell=`log(AP/pi)`。调用原 `fixed_winner_ap`，分数排序/ties由其原 `CachedScalar` 实现复用。

原 `T_FIXED_REPLICATES.npz` 仅取所需成员：concept_ids、point_prevalence、point_AP/native、point_ell/native；replicated_ell只解包一次并复制 `[:, :, 0, :]` 后释放原大成员。构造完整 `[2001,4,10,71]` ell，native来自旧证据，36旋转来自本次输出。主要Theta_std严格调用原 `estimands` / `bootstrap_summary`：先各网络9Q的逐概念median，再初始化参考median，最后71概念mean。保留全部NA及原undefined-replicate规则，2000次整分子bootstrap、双侧95%线性分位数区间。

G_init原点值/区间从旧 `T_FIXED_SUMMARY.json` **原样复用**；不为其重跑native bootstrap。Theta_raw及原95%区间从同一旧summary并列作历史对照，不重算旧bootstrap、不计算raw−std差值CI、不新增显著性主张。保存所有网络/基/概念的AP/ell与逐概念Theta/B/G；保留每个新D/T单元checkpoint。

标准化不是白化，不据此声称识别了“纯方向”，raw/std效应差不作成分贡献比例。仅中心化在精确实数下给每个标量加常数，正负pole的排序和ties不变；因此不单独运行中心化重复科学分析。这个数学结论不保证浮点量化下绝对不变。新补证不复验原H4 pre-GELU J2，不复跑199孪生，不是新的独立确认。

## 输入身份、预算和验收

新文件仅限本协议、`scripts/strengthening_20261025/b_standardized_rotation.py` 及 `results/strengthening_20261025/standardized_rotation/`。原小文件实际SHA核对一次；大atom_a复用原CACHE_RECEIPT已保存的SHA并现场核对size/mtime/shape/dtype，避免逐基重哈希。标签/atoms、原Q、moments、旧winner与weights各在首次使用时核对身份并记录。

本B初始累计硬限 **10790 CPU秒**，`time.process_time()` 包括导入、数据、编译、计算和写出，跨同一入口resume的ledger累计。实际CPU上限在独立 `CPU_BUDGET.json` 的 `cpu_cap_seconds` 中记录；不改变C/Q/reps/科学参数。保留10秒持久化余量，单元中检查预算/RSS，RSS停止线2GiB；新scratch与所有partial保留，不自动缩减科学范围。A+B总预算为6 CPU小时，统一累计。0 PLM forward、0新训练、0新随机权重生成。

先跑实际首个已锁D标准化旋转单元并记录CPU/墙钟/完整性，据其实测及已有原T实测作36D+36T成本估计；首个新T完成后再更新实测预计。所有完整单元保存后持续执行固定范围，失败/预算不足如实partial。

仅新增数值路径做一个最小合成golden：分块float64标准化→float32→Q32结果与显式逐坐标参考一致，近常数分母1且坐标数不变。原 `STATISTICS_ENGINEERING.json` 的6项golden证据只读复用，不重做统计测试。新checkpoint维度/有限性/原概念/候选范围与最终汇总一致性随运行记录。
