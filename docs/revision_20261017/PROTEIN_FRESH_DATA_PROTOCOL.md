# 长蛋白 fresh source gate：固定来源、全 raw 排旧与1304组门

2026-10-17。本合同仅授权离线实现/有限合成检查/旧exact-sequence键补充。真实HTTP必须先有研究操作者签发的 `SOURCE_DATA_GO` 数据锁。无PLM加载/forward、activation、预测或确认p；旧P1–P5与既有D产物冻结。判断：工程合同可核验（高），足额合格1304 fresh组未知，长T实际精度未知。

## 1. 固定来源与请求预算

唯一来源 `https://rest.uniprot.org/uniprotkb/search`；唯一query `(reviewed:true) AND (length:[257 TO 512]) AND (database:pdb)`；`sort=accession asc`、JSON、size500。从官方起点沿每次官方next，仅更换metadata阶段fields为 `accession,sequence,length,xref_pfam,xref_pdb,ft_helix,ft_strand`，取得旧P5资格所需完整序列、Pfam/PDB及精确Helix/Beta-strand特征。

完整来源必须恰为已穷尽metadata快照的13658 records、28成功页（27×500+158），release/date必须为 `2026_03` / `02-September-2026`，每页 `x-total-results=13658`。每页accession顺序与 `metadata_projection.jsonl` 的对应切片完全相同，最终集合与顺序完全相同；28页末无next，之前每页必须有next。重复URL、字段/排序/query/size漂移、release/total/IDs/count变化均立即停止，不能把partial当来源不足、不能使用partial资格结论。

每个URL最多3次总尝试（首次+最多2次同URL transient重试）；仅429/500/502/503/504或已分类超时/连接/DNS暂态错误可重试，间隔1/2秒。45秒请求timeout。URL必须HTTPS、唯一官方search路径，不允许重定向或外域。无批量并发、无另query、无换来源/长度/unreviewed、无自动全流程resume；失败保留已收raw及安全回执后交研究操作者。raw成功响应即保存并记入来源账，再做完整性检查。

继承既有网络路径，不枚举、修改或输出代理/VPN设置；不改系统网络、TUN、DNS或路由。回执只存公开官方请求URL、允许的来源headers、状态/异常类别，不保存任意HTTP错误正文或异常文本，避免意外保留代理/认证秘密；不输出Cookie/token/本机代理信息。

## 2. 全 raw 联合图与D分组门

复用metadata `OLD_POOL_GRAPH.json` 的3967 old supernodes，已代表全部旧16896 raw（P1 512+P2/P3/P4/P5各4096）以及ID/完整sequence/任意已知Pfam的旧内部传递闭包。该文件未存sequence键，所以只为这个新必要键，离线再读这16896旧raw的ID/sequence/Pfam各一次，按旧source size/mtime/count与supernode投影核验，保存 `OLD_EXACT_SEQUENCE_INDEX.json` 的 `SHA256(完整sequence UTF-8)→old_component_index`。不重新建旧图、不重算旧统计或旧raw文件全SHA；既有来源身份/验收复用。SHA此处是必要exact-sequence连接键，不是全目录复验。

下载必须完整且来源核对通过后，将全部13658新raw节点加入旧supernodes。按同accession、完整sequence的SHA、任意已知Pfam做联合图的完整传递闭包。无效序列/无标签/无PDB/无Pfam/将被筛除记录都在图里；无Pfam节点仍可经ID或sequence连接。缺失sequence无该键但仍保留其它键，不能把无效raw提前删除。先构全图并标所有触旧supernode的组件，再做资格筛选。已知图隔离不保证远同源独立或无预训练重叠。

旧432 selected pair按 `(origin,id)` 与当前 `OLD_SELECTED_UNION_GROUPS.json` 恰好联接。新根以所连旧supernode最小index命名 `old-union-{index+1:06d}`；映射和组身份须与既有D421组完全一致（哈希fold依赖组身份）。若出现进一步合并或影响组身份，输出 `OLD_SELECTED_FINAL_GROUPS.json` 与 `D_GROUPS_CHANGED_QUALIFICATION_HELD`，**资格筛选立即暂缓**，先交研究操作者对受影响D组/折与精度定点更新。不能继续声称原N1304精度合同适用。当前实现无自动D重拟合或修改precision权限。

图输出 `components.jsonl`、13658行 `raw_group_join.jsonl`、`GRAPH_RESULT.json` 与432项最终旧selected映射。每个完整raw恰有一个group；所有新组件、成员、已知Pfam、触旧标记均保存。pure-fresh组件key为其全部成员按accession排序后的 `acc:<ids>`。

## 3. 旧P5资格与pair规则原样复用

直接只读导入 `scripts/prepare_pilot5a_data.py` 的 `labels_and_pair`、`exact_interval`、`pdb_crossrefs`、`parse_chains` 与其原生辅助函数；不复制/重写结构标签选择器。仅在独立载入的module实例将SEED置为固定新salt `protein-long-transfer-20261017-v1`，原文件不写。外层长度唯一从64–256改为257–512；规范20AA、source length与实际长度一致、reviewed Swiss-Prot要求不变。

先排整个触旧组件，再要求上述有效长序列与PDB crossref，随后调用原生selector。保留原P5全部规则：仅Helix/Beta strand；位置为canonical accession的EXACT整数区间且有效范围；HELIX/STRAND重叠位置冲突排除；中心必须A并具有完整±7；feature evidence必须 `source=PDB`、合法 `ECO:\d{7}`、匹配entry PDB crossref；实验方法仅原集合 X-ray/NMR/EM/Neutron/Electron diffraction；必须显式可解析非`-` chain及UniProt坐标覆盖，与feature交集覆盖中心。±15可不足并由将来PAD编码；同PDB或同chain并非强制，不额外加两位点距离/窗口不重叠条件。这里是UniProt覆盖证据，不宣称新PDB原子映射。

每个合法位点多证据按旧tuple与新salt的hash取最小。遍历所有合法HELIX-A×STRAND-A不同q组合，按旧pair hash/tuple取唯一pair。精确hash envelope保持P5：

- evidence：`SHA256("pilot5a-evidence:{accession}:{q}:{label}:{pdb}:{chain}:{start}:{end}:{eco}:{salt}")`，tie旧 `evidence_tuple`。
- pair：`SHA256("pilot5a-pair:{accession}:{hq}:{sq}:{hpdb}:{hchain}:{spdb}:{schain}:{salt}")`，tie `(accession,hq,sq,hpdb,hchain,spdb,schain)`。
- 同组件成员：`SHA256("pilot5a-protein:{accession}:{salt}")` 最小，tie accession。每组件最多1 protein/pair。
- 组件：`SHA256("pilot5a-component:{component_key}:{salt}")` 升序，tie component_key。

不按标签效应、activation、预测或p排序。pair输出按q升序a/b，`y_a=1{a为HELIX}`；a不保证是HELIX。任何选择salt/hash/tie先于新source标签与模型输出固定。

## 4. 足额门、封存输入与失败

现有D精度计划唯一目标1304最终known fresh组件，cap1536；本gate不得降成512、扩池或根据T效果补人。仅当D分组完全不变且合法组件≥1304时，按预定组件顺序取前1304，余下标为unused，不作按效果补样的reserve。合法组件<1304即 `FRESH_DATA_INSUFFICIENT`，不写partial正式T cohort，不运行模型。2074 metadata组件上界从不视为真实足额。

通过时写 `proteins.jsonl`（1304）与 `SEALED_FRESH_CASES.json`（2608 cases、1304 canonical pairs，含完整sequence/q/token q+1/label/group与配对），状态 `INPUTS_SEALED_NO_MODEL_OUTPUT`。结构标签读取仅用于预锁eligibility与固定pair选择；不含激活、预测、检验或PLM输出。该输入集供研究操作者后续单独 `T_OUTPUT_LOCK`，本数据锁不放行T输出。没通过足额门则保留全部漏斗/证据与unused候选清单，不能将其称确认完成。

统一报告raw→触旧排除→规范reviewed长序列→PDB crossref→双类A位点/合法pair记录→组件代表→selected；保存各步骤保留数及非互斥筛除原因、合法pair组合数、组件数、raw/selected长度分布、feature/site/evidence排除与完整groupID联接数。不同筛除原因可重叠，不将原因相加冒充raw计数。

## 5. 准备、锁与一次集中检查

只写新脚本/测试、本协议和 `data/revision_20261017/protein_fresh/`。现有Python标准库；无新依赖或模型库。有限合成检查覆盖全raw无效桥接、sequence及ID桥接、D组漂移停止、source release/total/同IDs顺序、长度、直接调用旧P5 selector及证据拒绝规则、新salt、固定1304/不足不降N、候选顺序稳定和HTTP预算。

使用项目既有 `.venv/Scripts/python.exe`（Python3.11），不使用系统默认Anaconda3.7、不改环境。执行 `.venv/Scripts/python.exe scripts/revision_20261017/protein_fresh_data.py selftest` 后 `prepare`。有限检查含mock HTTP的同URL三次总尝试/非暂态即停，以及真实联合图合并D组时不调用qualify；mock不发网络请求或读取真实新标签。prepare核对3967旧supernodes、16896旧raw、15341旧exact序列、432 selected pairs/421当前D组与N_req/requested/authorized全部1304、cap1536。READY绑定代码、测试、协议、旧P5 selector、metadata projection/result/graph/currentDgroups、现有 `PRECISION_PLAN.json`、新old-exactseq index和synthetic回执；只对这些本次合同入口求SHA，不对旧raw重hash。selftest同时绑定当前测试文件与旧P5 selector；prepare拒绝已有HTTP receipt/raw或RUN_STARTED。准备阶段新HTTP/结构标签下载/PLM/预测为0。

研究操作者签发 `data/revision_20261017/protein_fresh/FRESH_DATA_RULE_LOCK.json`，需 `status="SOURCE_DATA_GO"`、`source_fetch_authorized=true`、`model_forward_authorized=false`、`T_output_authorized=false`、`ready_sha256=<FRESH_SOURCE_READY的SHA256>`、`created_utc=<晚于READY且含时区的实际锁定时间>`。执行 `.venv/Scripts/python.exe scripts/revision_20261017/protein_fresh_data.py run --lock <root锁>` 进行一次 fetch→完整source核对→全raw graph→Dgroup门→仅在unchanged时qualify。已有RUN_STARTED则不自动重开。实际失败保留安全receipt，只做受影响定点修复并由研究操作者决定接续；不重复旧P1–P5或D验收。
