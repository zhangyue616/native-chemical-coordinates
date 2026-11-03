# 化学确认 B：固定单元迁移、有限输入条件增量与匿名输出作用

2026-10-17。研究操作者已正式选择B，此文在新D模型输出和新T科学数据产生前定义合同。数据规则见 `CHEMISTRY_DATA_PROTOCOL.md`；当前仅工程和建池，正式D执行需工程实现就绪与D_RUN_LOCK；T执行需D_OUTPUT_LOCK。所有锁是本地前瞻性冻结，不称公共预注册。

## 研究对象与主张

唯一训练checkpoint为DeepChem/ChemBERTa-77M-MTR revision `66b895cab8adebea0cb59a8effa66b2020f204ca`，权重SHA256 `7882205587d3985c07e425632d0e57ba0fc2fc778e3b47aab02aef9b7961407f`。选择理由是可审计的三层小encoder和可恢复的匿名回归head，不是旧结果p最小。倒数第二层（0-based1）具有后续attention到CLS，且该层旧v4结果已经看过，明确作为开发先验。

在该层实际FFN post-GELU向量a（464维）观测、选择和干预。native候选为全部(+a_j,-a_j)，正pole先于负pole、j从0升序，共928个；每个rotation候选为±(aQ)_j，同样928个。Q由NumPy PCG64 default_rng以seed30000–30008生成464×464标准Gaussian矩阵，float64 QR，按R对角符号统一Q列，保存矩阵及生成代码；评分使用已锁dtype，所有网络共享同9矩阵。它是同一实际激活空间的可逆基变换；只有同时补偿输出权重时才有相同线性映射，不宣称未补偿的旋转保持完整模型函数。

三个配置/架构匹配初始化参考的Torch seeds固定21000、21001、21002。逐个报告，不把三个初始化当训练随机化重复。初始化构造算法和Torch/Transformers版本随实现锁；不加载旧activation或用旧缓存补充这些新分子。唯一训练checkpoint、三个seed和九个Q均固定；区间条件于它们和已选D，不含checkpoint总体/训练因果不确定性。

历史v3/v4采用preactivation上的非对称(GELU(z),-z)，新lane的±a与其不同。旧八模型atlas、39参照和posthoc保持原身份，不用新lane追认缺失历史源或逆向修复历史FWER。已知released-tokenizer身份损失属于复现限制，不作新贡献；本项目逐原子位置审核与身份保留分开。

## 固定赢家、主估计与统计家族

以数据规则D=5,000生成的全部合格概念C为主集合。对每个network m、每个basis b和每个c，在D最大化标准tie-aware average precision(AP)选unit/pole。相同AP按unit升序、positive pole先平局决胜；所有网络/basis同候选空间、同D、同标签。AP用每个distinct score阈值处的precision按正例recall增量加权；不使用trapezoidal PR-AUC。常数score的AP等于prevalence，标签无阳性时AP为NA，不补epsilon。实现只通过确定性的极小浮点量识别实际数值平局，不人为把相近分数合为一组。

T=10,000上只评估D固定winner，设

`ell[m,b,c] = log(AP_T[m,b,c] / prevalence_T[c])`；
`B[m,c] = ell[m,native,c] - median_r ell[m,Q_r,c]`；
`Theta = mean_c (B[trained,c] - median_k B[init_k,c])`；
`G_init = mean_c (ell[trained,native,c] - median_k ell[init_k,native,c])`。

这里是先对每概念取固定参照中位数再对概念等权平均，不与“中位数的模型总体均值”混用。负值不截断。Theta为主要估计，G_init是联合主张必要条件；两者各自双侧95%区间下界均>0，才允许限定为“这个训练checkpoint在D固定坐标的T迁移中，具有超出这三个初始化参考的native优势”。两者是intersection–union必要条件，不允许任选一个阳性。不得称一般训练因果效应、全部模型规律或所有单元都有语义。

T每个冻结概念必须≥20阳性分子、≥100全阴性分子、≥50阳性atoms。任何一项不足均使联合主确认事件门未满足；不删除失败概念后重算一个通过集合，不追加T。逐概念完整报告含事件数和最大单分子正原子贡献。没有positive AP的概念为NA，未通过门的统计只作描述。

区间用2,000次分子簇bootstrap，seed42017；每次有放回抽10,000分子，携带全部原子、所有重叠标签及所有固定scalar，同一次重采样用于全部比较。weighted tie-aware AP重算prevalence、中位数和汇总；D赢家不重选，概念不当独立重复。若任一概念无positive导致replicate未定义，报告比例；>1%则区间不可靠、联合主张不通过；≤1%明确报告剔除数量及条件区间。固定scalar排序/tie groups可以缓存，不能用atom-IID bootstrap或仅对概念均值重采样替代。

Oracle另在相同T、相同全部928候选上对每个network/basis/concept重选，仅报告点估计及oracle-minus-fixed差。对所有network/basis对称处理。oracle≥fixed是构造性事实，其差值不单独作为创新；oracle Theta与fixed Theta之差未必为正。oracle不启动另一套确认或重选主张。

## 有限局部输入参考

面板按数据规则最多12概念，不按网络效应选择。所有读出只用D，不以T调平滑、类别或分箱。无MLP/目标SMARTS输入/额外搜索。每概念都给token、radius0及radius1参考的AP与覆盖限制；条件信息主次要量为radius1与radius1+固定native j的**同T配对Brier增量**。

radius0类别为(element atomic number, formal charge, aromatic)。radius1类别为中心(element,charge,aromatic,total H,hybridization)及排序后的邻居多重集；每个邻居含bond type和邻居(element,charge,aromatic,total H,hybridization)。不使用目标匹配/SMARTS输出、目标官能团名字、全分子descriptor或学习的标签生成特征。radius1是有限局部非线性类别参考，可直接表达部分概念，这正是检验的输入解释；它不代表所有输入信息。它从原SMILES图取值，可能比有损tokenizer保留更多身份信息，必须披露。

固定平滑强度alpha=20。token类别和radius0频率均平滑到D全局原子prevalence。radius1类别g的基线`p_L(g)=(s_g+20*p_0(g))/(n_g+20)`，p0为其中心radius0参考；D未见radius1回退radius0，D未见radius0回退全局pi。

联合读出对每个概念采用其D固定trained/native scalar的全D原子quintile阈值（0.2/0.4/0.6/0.8，重复阈值合并；固定searchsorted side）。在radius1类别g中，对观察到的j-bin b设 `alpha_gb=20*n_gb/n_g`，`p_Lj(g,b)=(s_gb+alpha_gb*p_0(g))/(n_gb+alpha_gb)`；未见组合回退p_L，未见g亦回退相同p0。此选择使j恒定时联合读出恰等于基线，避免仅因重复平滑改变正则化而产生伪增量。报告每类/组合计数、实际bins、未见category及回退比例。分箱阈值/类别/频率表在D锁后不变。

`Delta_L_panel = mean_c mean_atoms_T [(y_c-p_L)^2-(y_c-p_Lj)^2]`。原子pool加权与主AP一致；分子cluster bootstrap仍携全部atoms，报告双侧95%次要区间。它独立于Theta/G_init联合家族，不用它挽救未支持的主要主张。不对12个概念逐个宣布显著性；逐概念结果描述性，完整报告。token/radius0及AP比较也描述性。胜过这个有限L不等于创造输入之外的信息；L无增量或CI宽时如实保留。

## D精度检查

从D固定hash次序以前4,000为D_fit、后1,000为D_precision。C仍由全D预定覆盖规则决定；仅用D_fit为全部network/basis选择临时winners，在D_precision作500次分子bootstrap(seed42016)，不接触T。以Theta和G_init两者的最大双侧半宽，乘1.25及sqrt(1000/10000)，估计10k T的操作性预期半宽。预定目标为log(1.10)≈0.09531（约10% multiplicative contrast），只是设计精度而非生物意义。

若D精度无定义比例>1%或估计半宽超过目标，报告实际值并停止T解封，交研究操作者判断这条固定预算是否不足；不换C/层/seed或增加T到显著。即使通过，也不保证真实T区间宽度。通过后在全部D5k重新选择唯一最终赢家；D_precision属于开发数据，不再叫独立确认。有限输入和作用对象在全部D固定，完整写入D_OUTPUT_LOCK。

## 匿名head与最小作用

按固定官方源恢复position0→dropout→Linear384→ReLU→dropout→Linear199；eval关闭dropout。使用checkpoint norm buffers；默认记录normalized匿名index q，反标准化另列。官方199 descriptor名字/顺序未恢复，不能把经验rho命名为官方任务。

机制候选为数据规则预先选择最多3个frag概念；使用同一主层的trained/native固定winner。以D分子该SMARTS unique occurrence count与199匿名normalized输出的Spearman相关选择最大|rho|的q（ties最低q），锁符号；|rho|<0.5或常数count/output时该候选机制关联门失败，不补选后续概念。报告199次选择和重复q/重复unit，不能把重复对象当独立机制。

干预在真实post-GELU a点：仅改全部atom anchor positions的指定坐标`a'_j=(1-alpha)*a_j+alpha*mu_D_atom,j`，不按真阳性原子标签选择位置。主alpha=1；selected-only alpha=.5为描述。CLS-only独立用mu_D_CLS，不与atom-only合并。special/PAD均不进atom mask。

每个selected unit以D原子mean|a|、SD(a)、输出dense W列范数进行匹配；前3量ratio在[0.5,2]，真实残差dose=SD(a)*||W[:,j]|| ratio在[0.8,1.25]，数值近零用固定1e-12容差，排全部selected units。对合格池以seed44000+原conceptID无放回取5个control；不足5则该候选机制匹配门失败，不放宽caliper。干预使用每个control自身D均值。T实际dose与D匹配程度完整报告，不据T重新匹配。

在T按CID的新固定mechanism salt `native-axis-mechanism-20261017-v1:` SHA顺序，取每候选最多200概念阳性分子和200全阴性分子，至少各50；不足则机制事件门失败，不补池。baseline输出复用完整T pass。每概念≤400×(selected+5 controls的全剂量atom-only + selected半剂量atom-only + selected全剂量CLS-only)=3,200 molecule evaluations，最多3概念9,600；另最多400个最终层atom-only结构零工程对照，总新增上限10,000。前置合成sham和末层结构零不作为化学发现。主层atom→下一层attention→CLS路径必须实际验证。

每分子q输出改变delta=y_intervened-y_baseline。报告mean|delta|与signed mean，不能混用|mean delta|。预定作用量为 `H_c=(mean_pos|delta_selected|-mean_neg|delta_selected|)-median_control(mean_pos|delta_control|-mean_neg|delta_control|)`。分子按正负层各自bootstrap，seed45017、2,000次；最多3候选，对H使用固定Bonferroni双侧98.3333%区间（即使少于3个成功也不缩为95%）；q的T同向Spearman关联也报告同口径区间，绝不命名官方descriptor。无正下界时只能描述效应与不确定性。即便支持，它也是冻结模型的计算敏感性和化学条件关联，不是移除官能团的化学反事实或一般生化机制。

## 实施预算与完整性

基础encoder量为4×15,000=60,000 molecule evaluations；rotation复用同一激活但AP排序和bootstrap另有CPU成本，不把它们算免费。MTR已完成synthetic小型CPU测速及head/mask/批次/PAD/sham/位置工程门；84token组实测约190 molecules/second，另对允许的512总token窗口作定点计时，不重复整套检查。研究操作者已批准模型提取与干预累计不超过4 CPU小时，AP/rotation/cluster bootstrap统计累计不超过12 CPU小时；使用进程CPU时间累加并同时报告wall time。各进程限4计算线程，避免并行过度占用，下载/建池单独记录。以上是有余量的硬停止政策，不能当预计耗时；不得凭旧GPU日志推估CPU，性能优化不能改变统计或少算replicates。

数值合同：a缓存/native score为float32；Q以float64保存，评分时转float32作a@Q；AP累计float64。score ties为实际float32值精确相等，winner AP绝对平局容差为8×float64 epsilon，仍按unit低/positive先。bootstrap分位数固定NumPy quantile method=linear；D两项各自半宽为(q.975−q.025)/2后取最大。

D与T独立cache，记录输入表、checkpoint、实现、版本、seed、层、dtype、atom anchors的身份绑定。允许对纯工程中断恢复相同缓存，不改模型或统计合约。T第一次解封后不得调整主C/winner/panel/output/control/threshold/样本数或删除不利分子；实际错误需保留失败证据、定点修复并说明影响范围。root是最终整合大阶段唯一验收责任方，复用本轮工程证据，局部工程PASS不升级科学PASS/作者批准/投稿。
