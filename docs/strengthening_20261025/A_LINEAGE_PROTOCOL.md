# A1：H4/D/T 谱系与固定 D-unseen scaffold 切片事后协议

记录日期：2026-10-25。状态：**POST_HOC_PROTOCOL_RECORDED_BEFORE_NEW_OUTPUTS**。本次为独立事后补证，不回溯为预注册，不覆盖旧协议、数据、结果或账本。执行入口为 `scripts/strengthening_20261025/a_lineage.py`，全部输出另存于 `results/strengthening_20261025/lineage/`。

## 固定对象与只读输入

| 集合 | 精确路径 | 预期行数 | 原行标识 |
|---|---|---:|---|
| H4 | `<legacy-source-root>/data/derived/lineB/corpus/H4_used.tsv` | 10000 | 原 rank，另保留零起始文件行索引 |
| D | `./data/revision_20261017/chemistry/development/D.tsv` | 5000 | 同上 |
| T | `./data/revision_20261017/chemistry/sealed/T.tsv` | 10000 | 原 rank 为 5000–14999，T 内索引为 0–9999 |

仅读取以上 TSV 的分子身份/来源字段及本协议和实现；不读取任何概念标签、分数、AP、模型激活或模型输出。实际输入字节的 SHA256 随本次读取记录。相邻 `<legacy-source-root>` 全程只读。不读取 PubChem 大文件、不联网、不安装、不训练、不做 Git 操作。

既有标准化来源只读核对：相邻根 `scripts/lineB/build_corpus.py:standardize`、`build_corpus_v5.py`，以及本根 `scripts/revision_20261017/chem_data.py:standardize` 和 `docs/revision_20261017/CHEMISTRY_DATA_PROTOCOL.md`。H4 使用原 v1 标准化，D/T 保存规则也明确复用该 v1 算法：原构建阶段均默认 parse/sanitize、按 heavy-atom 数选择最大片段（平局保留首个）、RemoveHs、原元素/大小/自由基/同位素/手性/键/生物结构过滤及规范立体 SMILES 回读。H4 和 D/T 的抽样盐、排序、排旧对象与 mapper 门不同。本审计**不重做**这些构建步骤，也不把它们相同解释为总体独立性。

## 本次共同身份定义

锁定现场可用的 RDKit **2026.03.6**。每个已保存 SMILES 仅执行 `Chem.MolFromSmiles(saved_smiles)`（默认标准 sanitize），然后 `Chem.RemoveHs(mol)`。对该相同分子图：

1. CID 身份：输入十进制 CID 的整数规范文本；不据 CID 查询或修复结构。
2. 规范立体身份：`Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)`。
3. 去立体身份：复制上述 mol，`Chem.RemoveStereochemistry(copy)`，再 `MolToSmiles(canonical=True, isomericSmiles=False)`。
4. 非空 Bemis–Murcko scaffold 身份：对相同 mol 调用 `MurckoScaffold.GetScaffoldForMol`，去立体后 canonical 输出。保留 RDKit Murcko 的元素/键与环连接定义，不转 generic scaffold。

本次不去盐、不 neutralize、不做 tautomer 标准化、不再选最大片段、不应用新的分子过滤。输入结构在共同身份规范化前后均保留。解析、RemoveHs、身份或 scaffold 错误分别留行并记录阶段/异常原因；不静默删除失败行。没有环的空 scaffold 单列，**绝不把空字符串当作共享 scaffold**。缺失/失败身份不进入对应身份集合。

对四级身份分别报告：各集合的有效行数、唯一数、重复超额行数、重复组及参与重复的行数、身份失败数；H4–D、H4–T、D–T 和三交的唯一身份数。行级表保留各身份及该身份是否存在于 H4/D/T、是否跨集合、是否三交。交集表保留实际共享身份及每集合行数，避免把唯一身份数当成分子对数。

## 纯结构冻结切片

唯一规则：取 **T 中 scaffold 成功且非空，并且该 scaffold 不在 D 的非空成功 scaffold 集合中的全部行**。保留原 T 内零起始索引、原 rank、CID 和原保存 SMILES。H4 任何身份/骨架重叠只报告，**不额外剔除**这些行。

保存长度恰为 10000、原 T 行序的布尔 `T_D_UNSEEN_SCAFFOLD_MASK.npy`，相应全部 T 行的 mask TSV，以及选中行 TSV。`FREEZE.json` 固定上述规则、精确输入路径/字节 SHA256、mask SHA256、总 T 行数/选中数/空 scaffold/失败数、UTC 和未读取分数/标签的声明。该切片属于既有 T 的事后结构切片，**不是新独立 scaffold split 确认**。后续统计不得根据结果改选样本、去掉 H4 重叠或改变空 scaffold 处理。

## 预算、故障与本阶段工程检查

科学调用：**0 次模型前向，0 次拟合，0 次统计重采样**。单进程，设置数值线程上限 1（不超过批准的 4）。独立 CPU 账本按 `time.process_time()` 累计，包含导入、输入、计算与输出；硬上限 900 CPU 秒，895 秒触发停止并预留 5 秒写 partial/账本。逐行预算检查；每 500 行保存一次进度。失败/预算停止保留已完成行和原因，不换规则、不自动重试覆盖。

唯一一次合适的工程检查合并在该次运行末尾：核对输入预期行数/固定 rank 范围、所有输出行与原输入 CID/SMILES/rank 顺序一致；有效/失败/空 scaffold 分拆完整；唯一数及交集集合边界合理；mask 类型/长度、每位与固定结构规则一致；落盘 NPY 回读与内存 mask 一致。仅一次计算输入 SHA 和 mask SHA，不全量重复哈希、不重新解析全部分子或回扫 PubChem。工程检查成功只证明本次构造完整性，不构成作者采用/投稿放行。

实际运行命令、Python/RDKit/NumPy 版本、线程设置、开始/结束 UTC、CPU/墙钟、状态及输出清单写入 `CPU_LEDGER.json`；完整计数写入 `SUMMARY.json`。
