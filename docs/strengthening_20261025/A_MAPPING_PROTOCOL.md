# A2：H4 原子位置事后严格核验协议

2026-10-25。本文件在本次H4逐分子审计前固定。本次是新增的事后审计，不追溯为旧H4的预注册门；原结果、旧p、层、模型、样本资格和旧账本均不改变。

## 输入与本地来源绑定

- H4 唯一输入：`<legacy-source-root>/data/derived/lineB/corpus/H4_used.tsv`，预期 10,000 行。按原始行序、原始 `smiles` 字节审计；保留 rank、cid、hash 及重复/异常，不去重、不规范化、不替换字符串。
- 旧规则源：相邻只读根的 `scripts/lineB/models.py` 中 `chemberta_atom_positions`。旧位置为 tokenizer 输出中非 special 且 `str.isalpha()` 的 token 序号；只有 RDKit 默认 parse 成功、字母 token 数等于默认 RDKit atom 数且总 token 数不超过 512 才返回数组。runner 从原源码 AST 提取并执行该函数体，不导入原模型模块，也不重写其规则。
- 历史实际编码路径：`chemberta_tokenizer()` 固定加载 `models/chemberta_77m_mlm`，且 `_extract_chemberta`/多层入口对 MLM 与 MTR 均共用这一 tokenizer。不能把 MTR 自身路径的存在表述为旧代码分别调用了两套 tokenizer。
- 本次分别载入原根的 `models/chemberta_77m_mlm` 和 `models/chemberta_77m_mtr`，不用本根 D/T tokenizer。核对各自的 tokenizer.json、tokenizer_config.json、vocab.json、merges.txt、special_tokens_map.json、added_tokens.json，以及 AutoTokenizer 判定类型所需 config.json，对照原 `models/download_log.csv` 的来源、字节数和 SHA256。历史记录指向各 HF 仓库的 `resolve/main`；本审计不补造固定 commit 身份。
- 两套六项 tokenizer 文件必须与历史下载摘要相符且彼此字节相同，才把本次双路径审计解释为与历史共用 MLM 编码等价。逐分子仍独立运行两套本地路径；每一行检验本地 AutoTokenizer 编码与严格 mapper 原始编码的 token IDs、token spelling、offset、attention/special mask 一致。MTR 每行另外与 MLM 同行的实际编码指纹核对。任一编码不一致只报告受影响项，不能用另一 tokenizer 静默代替。
- 严格实现复用 `scripts/revision_20261017/semantic_mapper.py` 的词法 atom span、辅助 source atom-map 图、原图签名/完整邻接核验和 token offset 归属；新的 subclass 仅注入各自本地 tokenizer 与来源身份，不改变旧文件。已有 `results/revision_20261017/mapper/ENGINEERING_CHECKS.json` 的 47/47 合成验证直接复用；只新增本地 tokenizer adapter 的小型验证。
- `prepare` 先生成 `results/strengthening_20261025/mapping/INPUT_BINDING.json`，保存上述输入/源码/复用证据/协议/runner 的实际 SHA256、大小、mtime 和已核 tokenizer 下载记录，随后生成 `AUDIT_LOCK.json`。未完成 lock 与必要 adapter 检查不运行逐分子审计。此绑定是协议的一部分；run 检查绑定文件、协议与 runner，及输入 stat 无漂移。

## 独立核验对象与分母

主分母为每模型完整 10,000 行，合计 20,000 molecule-model rows。重复行仍为不同审计行。运行中断时保留完整原始分母、已处理数和未处理数，不能把已处理数冒充全体。每模型的原子分母使用输入原字符串的 RDKit 图实际原子数，不用旧 first-letter 检查的 245,749 作为先验答案；两者数值可在完成后比较。

位置对象为原始精确 SMILES 在 RDKit `removeHs=False` parse 下的零基 graph atom index。严格 mapper 使用完整 lexical atom span（Cl/Br 与完整 bracket atom 各为一个 source atom），辅助带 source map 的 parse 将其绑定到原图，并核验原子性质、键、立体信息与独立词法邻接。不能把旧字母 token 顺序或“第 k 个 lexical atom”单独当成图索引的证明。另核旧规则使用的 RDKit 默认 parse 与严格图节点/边索引是否相同；不相同时阻断历史 graph-index 对照。

每个严格 atom 只有在原 offset 与解码源文一致、token 未跨原子、该 graph index 有且只有一个验证过的元素 anchor，且 mapper 的 `element_anchor_position_status=position-aligned` 时，才具有可比较位置。比较的是旧 `pos[graph_atom_id]` 与独立严格 anchor token index 的整数相等，不是 first letter、元素字符频数或数组长度相同。

## 互斥主分类与细分

按以下顺序给每一行一个主分类，并保留全部具体原因：

1. `hardfail`：输入缺失/无效、原 tokenizer 或来源绑定不可用、编码异常或与精确旧编码不相容、旧位置规则返回 None（包括 atom-count 与窗口失败）、旧/严格图索引契约不一致、或意外执行错误。记录具体失败，不丢该行。
2. `ambiguous/unverifiable`：严格词法/辅助图归属契约不支持该有效输入，或 strict 元素 anchor 不存在、非唯一、offset 不可核、跨原子等，因而没有完整独立的 graph-index→position 对照。此类不能作为已证实错位；候选差异可另列。
3. `actual_position_disagreement`：旧规则与独立 strict 元素 anchor 的完整可比较关系均成立，至少一个 graph index 的 token 序号不同。保存全部不同 graph index、旧/新位置与 token/offset；发现差异时记录影响，继续仅审计，不修科学源。
4. `verified_unique_anchor_match`：上述完整可比较关系成立，所有 graph index 的位置均相同。该结论仅关于元素 anchor 位置，不证明模型保留完整原子化学身份或输入语法。

`hardfail` 与 `ambiguous/unverifiable` 的 reason subtype、strict anchor-error prefix、full-offset-error prefix均完整计数。失败不触发 H4 过滤、重新编码模型、重跑旧 p、改层或补算激活。

## 与位置分开的侧向记录

- 原子身份损失：完整 source atom span 的未覆盖位置、Cl/Br 后字母、bracket 边界、同位素、形式电荷、显式 H、手性等各 lost attribute；统计 molecule/atom 数，保留原 token identity。
- offset 缺陷：全部原始 offset、源文/解码不符、非法/重叠/倒序、跨原子等；报告 full offset status 与元素 anchor status 两个字段。既有 mapper 允许已核非字母、且不涉及任何元素 anchor 的纯标点源文 mismatch 不单独否决位置匹配；缺陷仍原样记录，不修 offset、不搜索位移。
- token encoding 一致性、完整编码指纹、原子数与各图索引关系、重复 CID/SMILES/整行计数、非原子语法损失均保存。canonical 差异不作为新排除规则，也不改变输入。

## 交付、预算与验证

每模型一个全 10,000 行 TSV（含原输入、主分类、细分原因、旧/严格位置数组与差异 graph indices），一个逐行完整 mapping detail 的 `.jsonl.gz`；另有 `SUMMARY.json`、`ADAPTER_CHECKS.json`、输入绑定/lock 与独立 `CPU_LEDGER.json`/events。异常详情保留原输入，不静默跳过。最终阶段集中验收负责检查总分母/分类加和/图索引与差异抽样，并复用 47 项旧工程证据，执行方不重复全量旧测试链。

0网络、0模型load/forward、0新训练/重编码/拟合/bootstrap；只用本地Python和tokenizer，禁止访问权重内容。CPU用time.process_time()逐次调用累积到独立ledger，硬上限1,800 CPU秒；runner在1,760秒预留收尾并保存partial。预算不足不缩分母或降标准。
