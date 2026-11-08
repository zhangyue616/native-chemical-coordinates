# 1304组长蛋白T：冻结KRR、单中心MASK与固定四项确认

2026-10-17。唯一科学合同为同目录 `PROTEIN_KRR_PROTOCOL.md`；本文件只细化其已预定的长T执行、模型观察点、预算、锁与产物，不修改科学定义。当前仅授权新wrapper/有限合成/READY。真实模型加载、forward、T预测与确认统计须在研究操作者审阅代码后签发独立 `T_OUTPUT_LOCK.json`。

## 冻结入口与观察点

来源/资格/全raw排旧已集中验收，复用 `data/revision_20261017/protein_fresh/FRESH_DATA_ACCEPTANCE.json`、`SEALED_FRESH_CASES.json`、最终旧Dgroup映射及原脚本来源证据，不重新下载、重做selector、重建图或重跑D。真实源13658条，合法pair蛋白5779、fresh合法组件1498；固定hash前1304组/2608 cases进入T，194 unused不作按效果补样reserve。D432对/421组身份不变、102拟合已完成，唯一 `FINAL_MODELS.npz` 的六模型lambda/alpha/训练features/块trace全部冻结。

新 `scripts/revision_20261017/protein_T_confirm.py` 直接导入旧 `scripts/pilot3a.py` 的 `NativeRunner` 与新科学合同实现 `protein_krr.py`。不改旧文件，不调用旧短蛋白build_cases/其它旧科学命令。只在新wrapper放行257–512 AA。旧模型config为6块、hidden320/intermediate1280、GELU、rotary、max_position_embeddings1026；新最长512AA+2special=514 tokens，须显式检查支持并保持 `truncation=False`。tokenizer必须验证逐残基对齐和BOS偏移q+1。

每case独立运行一次原 `NativeRunner.run`：batch1、仅该中心A替换为唯一MASK，读取第三块（0-based2）的intermediate模块输出，即post-GELU中心token的1280维z，固定j1246。同一protein两个位点为**两次不同单MASK forward**；不得复用一次未mask输出、两个同时MASK或整序列一次forward读两个位点。无patch、干预、额外calibration/重复/初始化/候选单位。继承原trained本地ESM2 8M及已验收模型身份；不下载、不替换模型。wrapper内部仅在已校验根T锁之后将原NativeRunner要求的FROZEN状态作为构造adapter传入，不把旧锁当作本次T授权。

## 不拟合的预测与固定四项推断

严格复用 `krr.pair_data`，保留X64原定义、长度L/256、局部30位置21态与PAD；a/b顺序来自sealed canonical q升序。输入j为两个各自masked中心的z1246差。只调用 `predict_saved` 六模型（X/Xj/S/Sj/L/Lj），冻结D alpha、lambda及每块D trace，绝不更新T尺度或拟合。允许保存fullZ作缓存，但无全Z读出；无额外H回归、全Zoracle、共同lambda或bootstrap。

复用 `paired_contributions`、`group_average`、`future_primary_summary`：gX/gS/gL单侧greater、psi双侧，Holm固定四项alpha=.05；不可检验项p=1且仍在四项家族。组内先均值再组等权；报告逐对/逐组原始贡献、四项均值/SD/SE、原p/Holm p、95%估计CI与每项98.75% Bonferroni双侧t参考CI。逐项另报真实同时CI半宽和是否≤.01，不能用短D规划宣称长T必然达标。known图隔离不保证远同源或预训练无重叠，t推断为参考近似；不按结果换侧或追加分析。gS/gL非支持不解释为等效无信息。

## 计账与失败边界

固定1次model load、2608次单sequence forward；失败也先计入预算，0 retry、0 fit、0 source请求。CPU线程4，模型阶段CPU上限3600秒（含模型load、tokenization、forward与缓存处理），读出阶段CPU上限3600秒，独立账。每forward前检查已用CPU与两倍既往最大单call预留；阶段watchdog每50ms检查本进程CPU，用尽即写terminal STOP并退出该科学进程，无自动重启。50ms检测有相应调度粒度，不把预算描述为硬实时截止。所有模型阶段计账在 `FORWARD_LEDGER.json`，读出在 `READOUT_LEDGER.json`。

全2608成功且fullZ形状2608×1280/有限值后才进入读出。任何加载、token、shape、预算或数值失败：保留已计账事件、实际部分native缓存和STOP，不把partial称确认、不重新跑forward/fit或按效果补194 unused。已有T_STARTED、ledger或产物即拒绝自动重开。真实科学进程须由root与化学重CPU阶段串行协调；本wrapper不启动其它科学任务。

## 准备、前瞻T锁与交付

使用既有 `.venv/Scripts/python.exe`，只写本wrapper、对应 `test_protein_T_confirm.py`、本协议与 `results/revision_20261017/protein_T/`。有限合成只用临时人工数组/假tokenizer和假model（真实NativeRunner.run的mask/hook控制流），不加载pretrained模型或读取真实T activation/predictions。覆盖long长度/514 token支持、同protein两次不同单MASK、原post-GELU hook固定1246、forward次序/计数/CPU停止、原L/256/PAD、saved predictor使用冻结alpha/trace且不fit、固定四项/不可检验保留/实得精度。

先 `protein_T_confirm.py selftest`，后 `prepare`。READY绑定sealed输入与source acceptance、旧D最终模型/开发/precision/acceptance、冻结科学合同与两个不可变实现依赖、当前wrapper/测试/本协议/selftest、已验收模型identity与原接受回执、模型文件stat、Python/依赖版本。不重新全SHA旧raw或模型weights；复用已验收声明identity，锁后复核准备时模型文件size/mtime。prepare仅读取新sealed inputs、已接受开发元数据和合同，不读取真实T output；模型加载/forward/预测均0。

研究操作者锁 `results/revision_20261017/protein_T/T_OUTPUT_LOCK.json` 须 `status="T_OUTPUT_GO"`、`T_model_forward_authorized=true`、`T_readout_authorized=true`、`maximum_forwards=2608`、`new_fits_authorized=false`、`ready_sha256=<T_RUNNER_READY.json的SHA256>`、`created_utc=<晚于READY且带时区的实际时间>`。唯一执行命令：`.venv/Scripts/python.exe scripts/revision_20261017/protein_T_confirm.py run --lock results/revision_20261017/protein_T/T_OUTPUT_LOCK.json`。未签锁不运行。

通过时交付 `T_NATIVE.npz`（fullZ与case_ids）、`T_PREDICTIONS.npz`（固定六模型score/p/loss、j1246、原始逐对/逐组四项贡献、case/group联接）、两本预算账与 `T_SUMMARY.json`。随后由单一责任方集中验收新T结果；本执行PASS不替代作者放行、论文稿件改写或投稿授权。
