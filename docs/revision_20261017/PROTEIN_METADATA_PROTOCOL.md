# 蛋白 metadata 可行性前置门

2026-10-17。范围为已获授权的一次纯 metadata 门；**不授权新的科学路线、T、模型运行或确认检验**。脚本为 `scripts/revision_20261017/protein_metadata_gate.py`，所有新增数据与回执仅写入 `data/revision_20261017/protein_metadata/`。旧 data/scripts/results 只读，不读取旧研究数据。

## 固定来源与预算

唯一 endpoint：`https://rest.uniprot.org/uniprotkb/search`。query 固定为 `(reviewed:true) AND (length:[257 TO 512]) AND (database:pdb)`；`sort=accession asc`；`format=json`；`fields=accession,length,xref_pfam`。不请求 `sequence.value`、HELIX/STRAND/其他结构标签、`xref_pdb`/PDB 映射或激活。来源 release 固定为 `2026_03` / `02-September-2026`；不同或缺失即 `SOURCE_DRIFT`，保留实际响应并停止，不接受新 release。

API 首响应 `X-Total-Results` 是官方 query raw 总数，**不是可用 N**。本次从官方起点、accession 升序取前至多 **16,384 raw / 33 成功 HTTP 页**或官方耗尽，先到即停。初页 size=500；之后只沿官方 `next` cursor，更改 size 为 `min(500, 16384-已接收raw)`，其他参数含 query、fields、sort、cursor 不变。满额为 32×500+384。重复、旧连接、无效与无 Pfam 记录全部计入 raw 预算，不补池、不换 query/排序/来源、不扩至 >512/unreviewed。

只接受固定官方 HTTPS endpoint，不跟随重定向；next URL 参数不符、重复 cursor/URL、降序 accession、分页/总数异常或超额即停。重复 accession 允许计账并用于图桥接。每个 URL 最多 3 次临时网络错误/HTTP 429、500、502、503、504尝试；非临时错误即停。HTTP 200 后 JSON 解析失败保留 raw 并停止，不重试成功响应。既有代理环境只按运行时默认使用，不读取、打印或更改配置。异常仅记录安全类型和 HTTP 状态，不记录潜在含凭据的异常正文。

## 旧全 raw 联合图

旧池固定为 `data/proteins.jsonl` 全512，以及 `data/pilot2/http_provenance.json`、`data/pilot3a/http_provenance.json`、`data/pilot4a/http_provenance.json`、`data/pilot5a/http_provenance.json` 各自绑定的全4096 raw，总 **16,896**，不局限旧 selected；生物学无效/无标签记录同样保留桥接作用。仅投影旧 accession、完整 sequence、全部已知 Pfam，不使用旧结构标签或激活。复用 P5旧锁/asset manifest及每个 receipt 声明的身份，核对条数、schema、bytes与现场 mtime；**不全量重算旧 raw SHA**，不声称重新完成旧资产验收。

先在全部旧 raw 节点上按相同 ID、完整序列、任意共享 Pfam 做传递合并，再把旧 connected components 无损压缩为 supernodes，保存 `OLD_POOL_GRAPH.json`，含各组件全部 ID/Pfam、raw成员数及旧身份入口；不复制旧序列。后续新 metadata 仅以 ID/Pfam并入这些 supernodes，与直接在全部16,896旧节点上联合构图等价。新节点的任何传递连接触及旧 supernode，整个组件都排除；重复 ID 携带的 Pfam 桥不能丢弃。

另仅投影旧 P3/P4/P5 `proteins.jsonl` 中已选的176/128/128个 ID和旧 component key，共432 pair；不读其结构标签、位点或模型输出。报告各轮旧组件key在全旧raw联合图中是否合并，为以后统一CV提供当前已知group映射；新metadata若进一步桥接旧组件，则再更新映射与合并数。预HTTP映射保存在 `OLD_SELECTED_UNION_GROUPS_PRE_HTTP.json`，遍历后保存在 `OLD_SELECTED_UNION_GROUPS.json`；这些都是旧D身份分组，未运行CV或拟合。

新记录只投影 `accession/sequence.length/Pfam IDs`，逐行保存在 `metadata_projection.jsonl`，它是全部已见 metadata的投影，**不是候选 T 或 T selection**。无效记录仍桥接并消费预算；新无 Pfam raw 单独计数。返回中若意外含新 sequence.value、features、PDB或其他未请求内容，raw原文仅作来源回执保留，记录字段路径/数量，不将其值用于排除、标签判断、映射或任何统计。默认 `entryType` 仅视为 API 隐含元数据，不用于额外筛选。

## 上界及判定

完成冻结 traversal 后，旧排除后剩余的新 connected components 数（包括仅有无效 metadata 的组件）是保守上界。欠缺的新 exact-sequence 边可能合并组件或接入旧图；后续结构标签、PDB映射、中心A/有效配对与其他预定质量门也只会继续减少数量。因此不得把该数写作合格 fresh T、独立 N或科学路线已可行。已知 Pfam/ID隔离也不证明远同源独立或无预训练重叠。

输出官方总数、实际 raw、唯一 accession、重复/invalid、新Pfam与无Pfam、旧连接的直接/传递排除、剩余组件上界；分别回答512和1536是否连上界都不可能。若完成后上界<512，则这条固定预算的数据路线拒绝；若上界≥阈值，仅表示 metadata 上界尚未排除相应阈值，不通过新T门。若抓取失败/漂移，只报告实际已见与失败，部分上界不得用于拒绝尚未完成的路线；本次不另开 query。

## 锁与执行

1. 执行 `.venv/Scripts/python.exe scripts/revision_20261017/protein_metadata_gate.py prepare`：HTTP=0。运行有限合成桥接/预算/字段忽略检查，读取一次旧 ID/sequence/Pfam建立旧图，写 `SYNTHETIC_CHECKS.json`、`OLD_POOL_GRAPH.json`、`DATA_RULE_INPUTS.json`。后者绑定本协议、脚本、旧图SHA和旧source身份/stat；不会运行模型或新科学。
2. 向研究操作者交 `FINAL_READY`。由研究操作者完成锁；metadata执行者收到研究操作者明确 `DATA_RULE_LOCK` 回复以前保持所有 HTTP=0。
3. 研究操作者在同目录写 `DATA_RULE_LOCK.json`，最小结构为 `{"status":"DATA_RULE_LOCK","frozen_utc":"带时区的当前ISO时间","inputs_sha256":"DATA_RULE_INPUTS.json的SHA256","http_requests_before_lock":0}`。研究操作者可加授权/边界文字，但不得修改已准备 inputs。脚本核对锁时间、inputs、脚本/协议/旧图身份、旧source stat均未漂移。
4. 收到锁后执行一次 `.venv/Scripts/python.exe scripts/revision_20261017/protein_metadata_gate.py fetch`。运行时复用已压缩旧图，不再次全读旧raw；只做一次冻结的GET分页遍历。已有 HTTP receipt/raw即拒绝再次执行，不自动 resume/requery。
5. 保存 raw与安全HTTP状态/时间/必要响应头/官方next、`http_provenance.json`、`METADATA_GATE_RESULT.json`。每次尝试与每个成功响应均及时落盘；失败留实际 receipt。结果包含 readonly/scope计数：新序列/标签/映射使用0、activation读0、model/forward/fit/T selection/旧统计重跑0。

本子任务只做该前置门。大阶段最终验收由研究操作者负责，复用这次有限合成检查与实际回执，不重复旧科学或整套资产检验。
