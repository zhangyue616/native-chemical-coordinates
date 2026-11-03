# 新化学确认 B：数据规则

状态：2026-10-17 在新母体首次解压/抽样前固定；研究操作者负责写入 `CHEM_DATA_RULE_LOCK.json` 后才执行。研究操作者已授权 D=5,000、T=10,000 的单 MTR 新确认。分子输入的预定资格检查可以在构建阶段应用；T 的概念标签、激活、输出、单位选择及摘要仍不允许生成或读取，直至独立的 D_OUTPUT_LOCK。

## 来源、抽样和排旧

唯一公开源及其版本身份按 `CHEMISTRY_SOURCE_RULE.md`：PubChem Compound/Extras/CID-SMILES.gz，1,486,837,335 bytes，SHA256 `36b683a59272b196ed7ece140903a249f7df475a06c312c963a5c761f1037f1a`。本次来源接收方式由真实 receipt 记录，不把缓存复制写成网络下载。旧研究排除来自 `intake/old50k_exclusion/old_50k_exclusion.tsv` 的50,000个CID和50,000个保存canonical SMILES身份哈希；其receipt已由研究操作者核验，不重新读取旧50k或旧研究标签/激活。

新盐为ASCII `native-axis-transfer-20261017-v1:`。对原CID十进制文本计算完整SHA256(salt+CID)。取前12 hex字符的整数模1,000,000 <2,000者为候选；读取整个已锁源，按**完整SHA256 digest字节、numeric CID**升序。阈值和顺序不会由概念覆盖或任何模型表现调整。若固定候选不足，报告失败，不更换盐/阈值或追加来源。

按上述顺序逐个检查，直到15,000个合格且不重复分子；仅对该实际处理前缀报告筛除计数。先排旧CID；再标准化、排旧canonical identity、排本次canonical重复、检查mapper。通过者第0–4,999为D；第5,000–14,999为T。D/T及旧结构均按标准化后的canonical SMILES UTF-8无换行SHA256去重。未通过mapper的分子不占合格数；该规则在所有新结果前固定，报告对应人群限制。全库候选数与处理前缀数分开，不把前缀筛除率外推到全库。

这是一份按CID哈希的分子划分，不是scaffold划分，也不能证明未见于MTR预训练。两个CID标准化到同一分子时只保留排序最先者。

## 标准化

锁 RDKit 2026.03.6，遵循已接收且可复现的原始v1 `build_corpus.py` 标准化规则：默认parse/sanitize；分片sanitize；保留heavy atoms最多片段，平局取原片段顺序的第一个；RemoveHs；5–50 heavy atoms且不保留explicit H。元素限B/C/N/O/F/Si/P/S/Cl/Br/I；排radical、isotope、非四面体扩展chiral tag、非单/双/三/芳香bond或不支持bond direction；排原冻结tripeptide与nucleoside SMARTS；默认MolToSmiles生成canonical isomeric SMILES并reparse核原子数与允许的手性/键。不做neutralization、tautomer标准化或额外药物相似性筛选。保存精确过滤定义及来源，不重建旧v3身份。

## 原子位置观测门

使用官方固定revision `66b895cab8adebea0cb59a8effa66b2020f204ca` tokenizer，不修词表、merges或offset。输入模型的字符串必须就是已保存的canonical字符串。调用 `SemanticMapper.audit_molecule(..., require_canonical=True, max_tokens=512)`，全分子通过后每个RDKit atom才有一个可用的元素anchor token；不局部丢原子、不截断、不以字母数/顺序相等替代语义映射。

采用 `unique_element_anchor`：token原offset的源字符、完整原子span、词法图和RDKit atom/邻接必须一致；缺失/重复/倒序/跨原子/UNK/原文不匹配/图不符/不支持语法/超512总token均整分子排除。身份信息有损（例如Cl源位置上的C token）不单独排除，但记录source/token spelling、lost属性和输入碰撞。已识别的非字母语法token错位只有在不影响任一元素anchor且没有其他错误时可通过；原offset及完整offset失败原样保留，不修补或搜索位移。

合成工程证据 `results/revision_20261017/mapper/ENGINEERING_CHECKS.json` 47/47复用，不在每次构建时重跑。该检查只验证实现边界；每个真实分子仍执行一次上述gate。位置准确不证明encoder保留了化学身份、完整语法、原始预训练语义或旧v3正确性。

## 概念与数据接触

从已保存 `concepts_v1.json` 中按 `partition=readable` 且source∈{topology,fg_hierarchy,fragment_descriptors}取固定91概念，保留原ID/名字/SMARTS或function定义；不含82个Ertl经验类型、26输入属性及1个input-equivalent概念。只复制定义，不把旧prevalence/n_pos当新数据。SMARTS在精确canonical图上匹配，`uniquify=True,maxMatches=100000`，标签为所有match atoms的并集；function沿冻结topology定义，包含原“至少两个RDKit rings”的命名限制。

D标签允许生成；D确定C的门为每概念≥100阳性分子、≥100全阴性分子、≥200阳性atoms。全部合格概念进入C，按原ID序，不按网络效应挑选；少于12则数据覆盖门失败。最多12输入panel按topology→fg_hierarchy→fragment_descriptors逐类轮转、类内ID递增取得，不按j效果筛。机制候选取C中fragment_descriptors按ID最先3个且D阳/阴分子各≥200者；随后关联或剂量门失败者不补挑后面的概念。

构建输出包括D.tsv、sealed/T.tsv、逐分子anchor/身份损失计数、source/标准化/mapper/排旧/输入表身份及实际筛除计数。T文件仅保存精确分子身份与预定输入门证据；不得调用概念label或encoder/head函数。D生成labels、atoms、候选概念覆盖与面板，并保存各SMARTS每分子的unique match occurrence count，用于后续匿名输出的D关联；该count不同于原子标签并集大小。topology function仅保存positive-atom count并明确类型。构建可记录实际token input_ids fingerprint及本次重复/跨D-T碰撞计数，但不因此改变已锁canonical纳排，不把token碰撞伪装成化学身份重复或新发现。T标签/概念计数/激活/输出只有后续D_OUTPUT_LOCK绑定概念、赢家、读出、机制对象、精度与CPU预算后生成一次。

## 执行与故障

先准备源接收、冻结实现、有限合成sample/exclusion/split/schema检查，再写本地DATA_RULE_LOCK。源SHA、版本或实现漂移停止；缺少来源或mapper门不能静默降级。一次构建可保留中间候选和进度以应对纯工程失败，但不得重抽盐或以已见科学结果改变筛选。所有D/T原始科学统计、forward和拟合的调用数由各后续阶段单独记录；建池阶段为0。
