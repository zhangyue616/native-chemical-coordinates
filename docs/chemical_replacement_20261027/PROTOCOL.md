# 双坐标化学替代协议

2026-10-27。科学定义锁定于本文件首次保存；运行预算在独立 D 前16分子的 baseline 预检后写入 `RESOURCE_LOCK.json`，两者齐备后才允许新 T 条件。此为在已分析过的 T 上新增的事后分析，不是预注册或独立确认。


## 固定对象与输入

- DeepChem/ChemBERTa-77M-MTR revision `66b895cab8adebea0cb59a8effa66b2020f204ca`，原官方 checkpoint 与 head；block index 1，实际 FFN post-GELU，原归一化匿名199维输出。
- 主对象 j293、概念26、q115；预定比较 j454、概念24、q124。j454 的 q125 为次要描述，不能当第三坐标。
- 原 D5000分子/124530重原子只用于拟合；T10000分子/246722重原子全部评估。沿用原分子顺序、图原子顺序、已接受 anchors 与91概念字典。不重选层、坐标、端点、特征、样本或概念。
- 缓存元数据及 shape/原子/分子身份检查一次；不重复全体旧文件哈希。新协议、D 拟合、程序和最终产物各保留一次必要绑定。原 checkpoint 严格加载检查复用模型包装器。

## D 拟合与预测

每坐标独立拟合，所有均值以 D 原子为分母、float64 累计。

1. D-mean：该坐标 D 全原子均值。
2. M1：按该对象的二元目标原子标签，分别取 D 阳性与阴性激活均值；不得改为 occurrence 标签。两组需非空。
3. M2 = L_a：重用 `chem_input_reference.input_keys` 的精确 r0/r1 图键与 alpha=20。全局先验为 D 激活均值；r0 的 `(sum+20*global)/(n+20)`；r1 的 `(sum+20*parent_r0)/(n+20)`；T fallback 为 r1→r0→global。新独立连续回归函数，不调用旧二元 fit_one，不输入真实 a_j、其分箱或输出 q。token 不是 L_a 的 fallback；本轮不添加 token 替代条件。记录类别数、未见 r1/r0 及实际 fallback 比例。

D 激活20/40/60/80百分位以 linear 方法一次求取，原样保存全部四切点（相等切点允许空 bin）；T `searchsorted(side=right)` 固定五 bin。分箱仅用于事后误差与化学构成描述，不进入模型输入。D_FIT.json 保存模型、两类均值、切点及 D 分母；保存后绑定并用于全部 T，不根据 T 修改。

## 真实 forward 与位移置乱

每坐标固定五条件，顺序为 `D_mean, M1, M2, M1_delta_perm, M2_delta_perm`；合计100000条件分子评估。另计10000 fresh baseline 和预定48分子 sham。不得由旧 mean/alpha 输出推算新条件。

T 始终沿原顺序连续4分子批次，right padding、CPU float32、eval、torch threads=4/inter-op=1、相关 BLAS threads=4。每批先算 fresh baseline，再对两个坐标分别做五条件。hook 只改该坐标真实 atom anchors，保持其他坐标及特殊/PAD token。以图原子顺序显式索引 anchor，不能假设布尔mask顺序等于图顺序。

M1/M2 先生成逐原子目标预测；在真实本次 forward 激活上算 delta=prediction−actual。perm 条件以同一分子原子位置固定置乱 delta 后加回 actual。生成器为 NumPy PCG64，SeedSequence 整数列表 `[20261027, coordinate, global_T_molecule_index]`；M1、M2共享该分子的排列。保留原子数1和恒定位移等退化情况，不重抽。记录 identity 排列数、实际非零位移数和 L2/经验位移集合一致性。诊断保留位移分布及 L2剂量，但改变方向和替代激活边际；不称严格分布匹配 null。

输出保存 fresh baseline [10000,199] 与每条件输出 [2,5,10000,199]（float32），另存每分子实现位移 RMS、非平凡置乱标记及必要 hook 检查。与原网络之差一律相对同批 fresh baseline。每完成连续批次写 progress，禁止自动覆盖完成产物或自动全量重跑。

## 结果前固定数值门

- fresh baseline 对原缓存：q115/q124/q125 最大绝对差≤2e-6；全部199输出≤2e-5；选定 j293/j454 原子激活≤2e-5。全T累计最大差与位置记录，不将数值门解释为科学保真容差。
- no-op exact replay：T 批次起点0、4000、8380、9996，分别对两个坐标运行一次原值替回；另对两个坐标在起点0/8380做全activation范围检查，共48分子 sham（32 replay +16 scope）。no-op 输出逐值相等，范围检查其他坐标/非anchor逐值不变。范围检查使用既有真实 baseline activation 的指定目标，不引入新科学条件。
- 所有预测、激活、输出有限；offset、图原子与anchor bijection一致；每批条件数和完成索引完整。实际位移与置乱位移的排序/L2在float64算术定义上相等；float32写回后的误差单列，atol=2e-5、rtol=2e-6，不放宽门救结果。
- 失败记录当前批次/条件/端点与原始差值，停止；仅修实际缺陷并补跑未完成部分，保留失败计数，不改变科学定义或自动增预算。

## 汇总与区间

输出端点按分子汇总：主/比较 q 的 MAE、absolute-error 95th percentile（linear，点估计）、signed mean shift、mean normalized199-RMS。全T、该对象任一目标原子阳性分子、全部目标原子阴性分子均报告分母。q125同口径单列为次要描述。

配对差定义为该条件 MAE 减 D-mean MAE；M1/M2 另外减自身 delta-perm MAE，负值表示较小输出误差。完整报告五条件，包括不支持语义位置作用或低敏感性的结果；不计算近零分母恢复率。

激活 MSE 与 R²：全部T以及上述分子层；另按目标原子标签0/1和D固定五bin报告。点估计为所含原子 pooled MSE，R²=1−SSE/对应T原子相对自身均值的SST；SST=0记undefined。报告各组原子数与独立分子数，不将原子当独立重复。五bin×全91概念保存命中原子数/比例及未命中字典的比例；各标签可重叠，不作为互斥成分，不作逐概念显著性。

一个共同 molecule bootstrap：2000次、PCG64 seed=20261028，每次从全10000 T分子有放回抽10000个，完整携带其全部原子与条件；各对象/条件/分层共用同一组 draw counts。95% percentile linear 区间，分层使用当次抽到该层的分子；若未定义须报告数量而不隐藏。给主/比较/次要q的MAE、signed shift、平均199-RMS、上述配对MAE差，以及三种基础替代激活总体/分子层MSE/R²区间；原子标签/五bin误差与tail95为描述点估计。区间是边际估计，不是联合95%或正式多重假设检验；不搜索199个单独输出，不宣称等价/完全保真。

## 单阶段接受与交付

保存D模型、全T预测与forward输出、分子汇总、区间、全bin化学构成和运行台账。固定分析不引入新训练或H4重跑。
