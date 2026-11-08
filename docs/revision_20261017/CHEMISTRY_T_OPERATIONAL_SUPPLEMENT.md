# 化学T前操作性补充：匿名输出关联与干预

2026-10-17，研究操作者在新T激活/输出读取前明确批准。此补充引用现有 `D_RUN_LOCK.json` 与 `CHEMISTRY_CONFIRMATION_PROTOCOL.md`，不改写它们。除以下澄清外，C、样本数、赢家、读出表、作用对象、控制单元、统计与CPU预算沿原合同。

匿名q的T关联在**全部10,000分子**上评估：D选定概念的原SMARTS unique occurrence count与D锁定q的normalized输出计算Spearman，使用精确ties平均rank。2000次全分子配对bootstrap，seed45018；每次重采样count/output对并重排rank。报告原rho、原区间，以及乘以D相关符号后的rho/区间。不得以T重新选择q、符号、概念或强关联子集。全T不能换成机制平衡子样本。

H仍按原hash选择最多200阳性+200全阴性分子，在两层内分别重采样、seed45017、2000次。q关联与H各自保留固定3概念家族的双侧98.333333%百分位区间（分位点0.05/(2×3)、1−0.05/(2×3)），即使失败对象减少也不改变分母。两类区间各有自己的家族口径，不声称六区间联合95%覆盖。一个概念的限定机制主张必须同时满足H正下界与D符号定向rho正下界，是3概念的intersection–union决策；只有一项支持不能升级。任何undefined rho bootstrap须报告；存在未定义重复时不宣布该机制对象联合通过。

全T rho排序与重采样计入已批准43,200秒统计CPU硬限，不增加forward预算。finite-input面板仍复用seed42017的同一2000×10000分子权重，概率/Brier用float64；描述性AP使用与主管线相同float32 scalar ties合同，明确记录转换。

全剂量与半剂量在原真实post-GELU位置仅替换一个native坐标：`a_j + alpha*(mu_D_j-a_j)`，alpha为1或0.5，所有其他坐标逐值保持。CLS-only使用原D的CLS均值、atom-only使用原D的atom均值，mask含全部atom anchors且排special/PAD。负pole只决定预测scalar的符号，物理干预仍对同一native unit和native均值。实现可在同一模块点注册独立hook，以支持半剂量，不能替换成对整464维的近似缓存回放。

干预T实际dose逐分子计算 `RMS_mask(a'_j-a_j)*||W[:,j]||2`，报告正/负层、selected/control与D匹配dose；不据此重新挑control。输出仍为delta、mean(abs(delta))及signed mean分别报告。

最多400个最终层atom-only结构零对照采用T前400分子及与全T提取相同的连续batch4/相同padding几何。最终层所有atom FFN激活置零，仅CLS读出理论不变；与已缓存同batch baseline比较，预定最大绝对误差容差1e−5，作为工程检查而非化学效应。这最多400次包含在原10,000新增molecule evaluation上限中。实际前置合成检查需证明alpha1与已验收mean-replacement路径一致、alpha0为sham、alpha0.5的坐标变化与dose为全剂量一半、非目标坐标和mask外值不变。

所有T执行程序及此补充的SHA在首次T科学读取前进入root `D_OUTPUT_LOCK`。T科学进程串行使用同一CPU账本；任何实际中断保留已完成条件与失败证据，恢复须保持同一对象与参数，绝不扩预算追求阳性。
