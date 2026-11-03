# C：固定人工配对的有限段标量剂量干预

2026-10-25。事后补证，依`DESIGN_DECISION.md`放行。历史结果、研究选择和不等剂量现象均已知；本文件在本轮新干预输出前锁定。

## 固定输入与选择

使用原`MECHANISM_D_PLAN.json`三个对象24/25/26及各自selected+5 controls、匿名q124/q125/q115和D均值。每对象原`MECHANISM_<id>.npz`的200正例/200负例全部保留。两层分别按SHA256(`eighth-dose-pair-v1-20261025:`+十进制CID)、CID、原T索引排序并一一配对；不按激活或输出优化配对。

用既有T float32 post-GELU atom_a、原atoms_T offsets和D outgoing-column norm计算full-dose。D均值在实际float32模型中使用其float32可表示值；位移/RMS和权重norm使用float64计算，准确记录这两个精度口径。每对12个full-dose的最小值为tau，三个s为0.25/0.5/1，alpha=s*tau/d；若d为0，tau必为0，alpha置0且保留整对。禁止epsilon分母。配对、tau、全部alpha、源绑定及参数SHA先落盘，再开始任何本轮模型输出。

## 真实前向与数值核验

固定4分子批次，按pair的positive/negative交错顺序，所有baseline与条件使用相同批次/右padding。加载原官方MTR wrapper，原倒数第二block的实际RobertaIntermediate输出是干预位置，仅mask内原子anchor的一个坐标改变。每分子alpha独立广播；插值在float64中计算后cast回原float32模型，原始输出和未干预坐标/非原子位置不变。真实输出经原head返回normalized anonymous q。

一次小批工程前置包括baseline/alpha0复原、真实hook位置及atom-only范围、alpha范围、与旧cache q/activations的数值差异和实际剂量。所有baseline与工程样本均计入本轮forward次数及CPU，不能写旧账本。实际投影剂量以被模型消费的float32改变相对原激活的float64 RMS乘norm计算；与目标误差必须满足max absolute<=1e-6*(1+max target)，另报告target>1e-8的relative误差。若失败，保留失败且只修实现/数值，不改pair/tau/s或对象来救结果。

计划21600 intervention evaluations，加1200（每对象400）baseline及小批工程检查，总硬上限24000；C整个实际执行进程累计CPU<=3600秒（包含统计，采取更保守计法），最多4科学线程。每完成条件持久化，预算/数值/源冲突失败保留partial，不擅自扩大预算或重开旧run。

## 估计与披露

每个s计算selected正例平均绝对输出变化减负例平均，再减五个control相同差的中位数。主指标H_pair(s=1)以200个人工pair为cluster、PCG64 seed20261025、2000次有放回抽样，全部正负/六坐标共同权重，固定3对象98.3333333333% percentile区间。s=.25/.5只描述性轨迹，不加新显著性。旧H及其区间仅历史对照，不宣称联合六区间95%覆盖。

保留所有signed/absolute response、200pair tau与alpha、分位数、true-zero/tau<1e-4比例、限制tau的正负分子及坐标（并列最小值全部保留）、实际target误差。零剂量pair不剔除。缺值/数值失败完整报告，不能把条件于该人工pair和最小剂量的结果写成普遍机制、等向量扰动或剂量解释比例。区间含0不证明等价。
