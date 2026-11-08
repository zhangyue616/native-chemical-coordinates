# 固定 j1246：统一 KRR 开发与条件性长蛋白确认合同

2026-10-17。实施合同，覆盖两份设计建议中不一致的读出与归一建议。本文件只释放离线准备及合成检查；真实 D 拟合须研究操作者签发绑定 `FINAL_READY.json` 的 `D_GO`。本脚本无网络、模型加载、forward 或新 T 入口。原 P1–P5 文件与结论冻结。

## 1. 任务、输入与科学边界

固定既有 ESM2 8M、第三块（0-based 2）post-GELU、原 j1246、原 H。H = (`−1` 属 `AILMFPVWY`) AND (`+2` 属 `AILMFPVWY`)；不重新选 j/H、模型、初始化或层，不重新设计候选假说。H 是 S 的确定性二阶 AND，其表达性不代表新增序列信息。

D 明确由 P3 全176对、P4旧T128对、P5旧T128对组成，合计432对、864位点；原先 T 已进入新开发角色，不能再次称 fresh T。记录来自 `data/pilot{3,4,5}a/proteins.jsonl`。逐 `case_id` 与历史 P5 `D_CASES`（含完整P3+P4，包含P3原C16）及 P5 `T_CASES` 做 `id/component/完整sequence/q/label/y` 一致性校验；不按文件位置拼接。j 从 P3 `D_NATIVE/C_NATIVE/T_NATIVE.npz`、P4和P5 `T_NATIVE.npz` 按 `case_ids` 联接，检查完整1280维形状和有限值后仅提取1246。

当前全局组映射唯一默认输入：`data/revision_20261017/protein_metadata/OLD_SELECTED_UNION_GROUPS.json` 的 `selected_pair_group_mapping`，按 `(origin,id)` 联接，字段 `current_known_union_group`。当前为432对、421个已知全局组；新metadata产生桥接，不能沿旧432组件独立分折。未来 exact-sequence 可能继续合组：脚本接受 `--groups <finalgroup文件> --group-field <最终字段>`；每个旧对必须恰有一条映射。更新后重新准备/签发D_GO，若已拟合则须研究操作者另定针对受影响折的更新权限，不能自动重跑。已知组不保证远同源独立或无预训练重叠。

目标确认总体为 reviewed+PDB、长度257–512的长蛋白；这是 **short-D→long-T 转移**。目前新来源13658 raw耗尽后2074 metadata组件仅为序列/标签前上界，不是足额最终新队列。D不含新长蛋白输出，D方差不保证长T精度或功效。源不足或精度不可行都须报告失败，不能降N冒充完成。

## 2. 特征、配对与严格嵌套

每对按 q 升序为 a/b，a不保证是HELIX；`y_a=1{a为HELIX}`，`t=2y_a−1`。不按标签重排、不加入镜像行、无截距。X沿旧64维：全局（排中心）、±7、±15的20AA组成，加 `q/(L−1)`、其平方、`L/256`、±15可用位置比例。即使未来L>256也保留 `/256`。P沿旧S局部630维：中心外30个有序offset（−15…−1,+1…+15），各21态（20AA+PAD）one-hot；因此S仍是X+P的694维特征语义。

所有模型采用同一KRR、D、四折、loss与固定lambda候选。特征块没有逐坐标标准化、RMS或中心化；以下每块trace归一是唯一尺度变换。该合同取代设计建议中整体S/694与旧logistic逐坐标RMS，避免改变X在S/L中的权重。

对任意位点核k，配对核为：

`B_k(c,d)=k(a,a′)−k(a,b′)−k(b,a′)+k(b,b′)`。

四个原始PSD块：

- `B_X(c,d)=ΔX_c·ΔX_d`。
- `B_P`：上面四项作用于局部one-hot内积；位点内积等于相同位置类别匹配数m。
- `B_Q`：上面四项作用于完整二阶位点核 `k2(v,w)=m(m−1)/(30×29)`。它对应全部不同位置对、全部类别组合的交互内积除以435，包含H的表达空间。没有位置/交互挑选。
- `B_j(c,d)=Δz1246_c Δz1246_d`，标量为既有原生激活。

每个训练折I，令 `r_b(I)=trace(B_b[I,I])/|I|`。若r>0则 `K_b=B_b/r`；零trace块定义为零（无补抖动、替代单位或搜索），包括常数j。r只从训练对求取，验证/T不更新。各模型逐块等权相加：

`K_X=K_Xblock`；`K_Xj=K_Xblock+K_j`；

`K_S=K_Xblock+K_P`；`K_Sj=K_Xblock+K_P+K_j`；

`K_L=K_Xblock+K_P+K_Q`；`K_Lj=K_Xblock+K_P+K_Q+K_j`。

因此X块在X/S/L的尺度完全相同，P块在S/L相同。禁止再除总维数、块数量或整体模型trace。嵌套的是核/函数族；独立选择lambda后的性能差不是固定惩罚消融或机制中介。

## 3. 拟合、折、选择与预算

目标为 `mean_pair(t−f)^2 + λ||f||²_RKHS`，解为 `α=(K+n_pair λI)⁻¹t`，float64 Cholesky，无优化重启。数值残差必须≤1e−8。训练每对等权，即合组后含多个旧对的组会贡献多个训练行；这是明确的训练目标，并不将其统计上当作独立样本。

同组不跨折。唯一seed为2026101708：唯一组按 `SHA256("protein-krr-cv:{group}:2026101708")` 与组名排序，rank mod4；行序变化不改变分折。当前421组分成106/105/105/105组，各折对数另在ready记录。若未来组身份合并，更换映射使相应折和训练尺度更新；不能保留同组跨折。

六模型均使用 `{0.0001,0.001,0.01,0.1}`，每lambda四折。这一网格由研究操作者在任何真实D激活读取或拟合前修正：当前每块 `trace/n=1` 且解为 `K+nλI`，旧 `{0.1,1,10,100}` 数值属于旧sum-SSE/特征尺度合同；直接沿用会使高维P/Q系统性强收缩（n约400时最小对角惩罚约40）。新四值覆盖与当前trace匹配的弱至强正则，保持四候选、四折与102 fits，不依据任何新D/T结果调整。

`f=K_valid,train α`；`p_a=clip((1+f)/2,0,1)`，`p_b=1−p_a`。paired Brier为 `(p_a−y_a)^2`，恰等于两个互补位点损失的平均。交换a/b时f变号、概率互补、损失不变。OOF选择先每组平均该组全部对Brier，再对组等权平均；精确平局取较大lambda。各模型独立选lambda，再全432对拟合一次。

每模型 `4λ×4fold+1final=17` fits，合计 **102真实KRR fits**；无另行logistic桥接，无全Z或共同lambda敏感性，无新的H→j回归。H→j beyond S可作为另行预锁的次要估计（两个同折ridge最多34 fits），本实现当前不运行、不计入102，也不拿旧H回归替代新S外证据。D新增HTTP/model-load/forward全部0。读出拟合CPU上限3600秒，失败计账、无自动重启/扩网格。

## 4. D精度门与未来T固定四项家族

从每模型自身D选定lambda的OOF概率求逐对loss，再形成固定列序 `(gX,gS,gL,psi)`：

`gX=ell(X)−ell(Xj)`；`gS=ell(S)−ell(Sj)`；

`gL=ell(L)−ell(Lj)`；`psi=gX−gS`。

先将四列按已知全局组内平均，再对组等权。D的均值只作开发描述，**不计算或报告D确认p值**。本方案不是nested CV：同一OOF用于lambda选择与方差规划，开发乐观性必须披露。精度操作性界为Brier .01，不是已证实的生物学最小效应。

对四项组均值的样本SD（ddof=1），采用保守膨胀1.25与固定Bonferroni四项同时95%近似常数z=2.4977：

`N_req=ceil[(2.4977×1.25×max(SD_gX,SD_gS,SD_gL,SD_psi)/.01)^2]`。

`target=max(512,N_req)`，硬上限1536；不另外凑32倍数。若N_req>1536，`PRECISION_INFEASIBLE_WITHIN_CAP`，合规target为null，保留真实requested_target；不能把目标截成1536并声称通过。若≤1536仅为 `PRECISION_TARGET_FEASIBLE_PENDING_FRESH_SOURCE`，还须序列/标签/排旧后足额来源门。2074只是上界，不能代替实际门。规划CI半宽不是80%功效承诺，短D和长T分布也可能改变方差。

未来同一fresh T上保留四项家族：gX>0、gS>0、gL>0、psi双侧≠0，Holm固定四项、alpha=.05。不可检验（缺失、非有限、零SD等）保留p=1，不删分母。组作为推断单位，组内多对先均值。报告均值、原始逐对/逐组贡献、SD/SE、未校正p与Holm p、95%估计CI，以及Bonferroni每项98.75%的双侧t参考区间（四项同时≥95%近似）。t参考及残余同源限制显式标注；不追加bootstrap或按结果换侧。代码的未来推断helper不在D运行。

psi>0仅说明预定独立调参管线间的增量较小；只有gX也支持正增量才可讨论正收益缩小。gS/gL非支持不等于等效无信息。L仅覆盖完整二阶局部族，不代表所有非线性或全部序列信息。

## 5. 离线准备、门控和有限验收

使用现有 `.venv/Scripts/python.exe`，不改环境。只写新脚本、测试、本协议与 `results/revision_20261017/protein_D/`。先执行 `protein_krr.py selftest`，仅合成数据核对完整二阶显式特征/PSD/嵌套、pair swap、常数j、训练fold尺度无泄漏、mean-loss解、中心排除/PAD、组分折、精度超cap与固定家族。随后 `protein_krr.py prepare` 做旧metadata联接和当前组分折，不读取activation值、不做真实拟合。

`FINAL_READY.json`只绑定本次代码、协议、group文件、新输入manifest和合成验收回执；复用旧数据/旧验收，不重算旧模型或全旧文件哈希。D前网格修正复用原9项核/求解器/分组证据，以 `GRID_REVISION_CHECK.json` 单独验证新grid严格排序、四候选和实际生产选择函数的精确平局取较大lambda，并绑定新script/旧SELFTEST，不重做metadata联接或全部selftests。研究操作者D_GO JSON需包含 `status="D_GO"`、`D_only_authorized=true`、`T_authorized=false`、`maximum_real_fits=102`、`ready_sha256=<FINAL_READY的SHA256>`。真实执行为 `protein_krr.py develop --d-go <研究操作者回执>`。D拟合开始后拒绝自动重跑；未来group变化须另定权限与针对范围。

主要产物：`D_INPUT_MANIFEST.json`（完整旧病例与最终分折）、`FIT_LEDGER.json`（102 fits计账）、`D_OOF.npz`（全部lambda/模型OOF和四项贡献）、`FINAL_MODELS.npz`（最终alpha、训练对features、每块D trace、lambda）、`PRECISION_PLAN.json`、`D_DEVELOPMENT.json`。`predict_saved`仅提供数组预测供另行授权T评估器复用，不加载PLM、不接收CLI T。新T具体队列、source门、提取与确认锁不由本合同授权。
