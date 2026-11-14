# Camera-ready 与重建代码一致性审查

审查日期：2026-11-13。基线提交：`867ac04f15f57025b1a396529492f3369b54a438`。
论文依据为 `paper/main.tex`，即已核对与 camera-ready 源码包一致的正文。
本次只增加审查材料与测试，不修改正式论文、历史数据、运行时或已发布性能结论。

## 结论

**核心选择规则在同一执行兼容组内与 Algorithm 1 一致；完整运行时、基线定义和实验协议不能确认与原实现一致，现有证据不能验证论文数字。**

新增独立伪代码参考实现：2,000 个随机、同兼容组 ready pool 的有序选批结果全部一致。
构造的混合 step 反例则不同。共 16 项测试通过，包含原有 3 项真实上游 CPU/toy-model 集成测试。
“通过”包括确认已知差异的 characterization 测试，不表示所有论文要求已满足。

## 一、算法逐项对照

| 项目 | 论文位置 | 重建代码位置 | 判断 |
| --- | --- | --- | --- |
| 长度 L=P+G、padding ratio=B*max(L)/sum(L) | Background；Design | `shapeshift/scheduler.py:28,136` | 相同；相等 G 的兼容组内也对应上游实际矩形宽度 |
| 超过 tau 的最久未服务请求作 anchor，否则最短请求 | Algorithm 1；Service-Gap Anchoring | `scheduler.py:119–125` | 相同；代码用 req_id 补足完全相同键的稳定排序 |
| 按长度距离、last_service 排序 | Algorithm 1 | `scheduler.py:131–133` | 相同 |
| rho<=beta 才加 companion；拒绝后继续扫描；允许不满批 | Algorithm 1 | `scheduler.py:136–145` | 相同 |
| 未完成状态保留，下步重选；last_service 在派发时更新 | Algorithm 1 最后几行 | `scheduler.py:146–147`；上游 engine/denoise_loop | 派发时间与伪代码一致；不等同于 GPU 完成时间 |
| 从整个 R 选择最近 companion | Algorithm 1 | `scheduler.py:33–40,126–127` | **额外约束**：要求 block/step/G/采样配置相同 |
| admission 后的 active pool | Runtime Scope | `scheduler.py:113–115` | 思路一致；4B 容量、256 pending、未服务请求以 admission 时间初始化是具体实现选择，原正式协议未完整给出 |
| beta / tau | 配置表写“dlmserve defaults” | `scheduler.py:50–52` | beta=1.25 是重建选择，无原值证据；历史摘要有 tau=5000 和 pool factor=4，但只能证明这些已保存运行的设置 |
| ratio budget 与固定 token budget C | Compactness Filtering | `ALIASES['token_budget']='shapeshift'` | 名称兼容不证明旧 token_budget 语义已恢复；固定 C 与固定 beta 不是一般意义上的相同约束 |

反例：三个 active 请求的 `(L,step)` 分别为 `(100,0),(101,1),(110,0)`，B=2、beta=1.25、均未 overdue。
Algorithm 1 选请求 `[0,1]`，当前代码选 `[0,2]`。因此只能声称“受兼容约束的重建”。
不能直接删除兼容检查：固定上游 step_batch 的文档要求同 block/step，且使用第一行的部分采样参数。
若要扩大 R，必须先验证混合进度、不同 G、LocalLeap、位置编码和 attention mask 的语义。
异步到达自然产生混合进度，额外约束会影响 batching 机会，但收益方向和幅度尚未测量。

此外，论文“最多 B 次 compactness 检查”的文字不准确：若连续拒绝候选，伪代码和重建代码最多检查 |R|-1 次；总复杂度仍为 O(|R|log|R|)。这属于文字与伪代码差异，不建议为贴合这句话改变算法。

## 二、阻止直接复现的实现与协议差异

1. **基线身份不明确。** 论文称 FCFS 是 dlmserve 默认调度器。重建 `fifo` 是最老 req_id anchor 加 arrival-order compatible fill；固定上游 `dd3efe7...` 的默认调度器却按 `(step,bucket,LocalLeap)` 分组，并按组大小、等待时间和 flush bonus 选组，active pool 默认为 2B。测试反例：三个请求 step=[0,1,1]、同到达时间、B=2，上游选[1,2]，重建fifo选[0]。两者不能互换命名。应同时保留 paper-style FCFS 与未修改 upstream-default，并显式记录各自 admission 设置。StaticBucket-FIFO、SJF-Step 尚未实现；length_sorted_admission 是重排入队顺序，不能冒充 SJF-Step。
2. **trace replay 不符合正文。** `run_dlmserve_online_trace_benchmark.py:81–105` 读取长度后 shuffle，再取 n_requests；不读取到达时间。`main:396–411` 默认 24 请求、B=4、8 steps；正文配置为 full split、保序、B=8、默认128 steps。现有 `asyncio.sleep(gap)` 是相对 sleep，且同步 GPU 工作会阻塞事件循环，不能代表固定外生到达时间；跨策略的真实入队时间也可能漂移。
3. **长度实验与真实语义输入分离。** `make_prompt` 用重复的 “the” 合成输入，再用各模型 tokenizer 重新测长。可用于长度机制实验，但不能说明真实任务质量，更不能保证跨模型共享同一实际长度分布。trace 的 gen_len 被忽略，实际 G 来自 CLI。
4. **跨模型路径未恢复。** 当前 adapter 继承上游的 LLaDA loader；该上游其他名称也回退到 load_llada。没有 Dream/CoDA/LLaDA2-mini 专用接入与验证，不能通过替换模型路径宣称支持四模型。
5. **dInfer 不是完整 Algorithm 1。** `run_dinfer_online_trace_benchmark.py:107–134` 只有 fifo、length_bucket、aging_bucket 和 sorted admission，无 rho/beta 检查；整批生成到完成的执行方式符合论文对同步 wrapper 的限定，但其 aging 是入队等待而非反复 step-service。必须准确标记为 wrapper 分组策略。
6. **计量不足。** `monitor_start` (`runner:257–269`) 观察 active pool membership，不是首次 GPU service；不能用于排队分解或 inter-service wait。shape trace 的 request IDs 与派发时间可以重建相邻派发间隔，但尚无正式聚合器、首末间隔规则或完成时间定义。需要记录每次 dispatch/finish、scheduled/actual arrival、失败/拒绝和完整 request identity。
7. **开销与质量均未证明。** 当前每步同步打开/写关闭 CSV；无排除冷启动的正式 warmup 或 scheduler/日志开销测量。toy model 输出不依赖上下文注意力，无法验证真实模型在 padding、batch composition 和 BF16 数值差异下的 token 等价性。

## 三、历史证据审计：不能把图形重画当作结果验证

可重新运行：`python tools/audit_paper_artifact.py`，输出保存在 `PAPER_AUDIT_EVIDENCE.json`。

- 53 个 CSV 中有 51 个 `*_summary.csv`、72 行运行摘要。另两个是汇总/工作负载统计，不是新增逐 seed 运行。
- 这 72 行 arrival_gap_ms 全部为 0；62 行 dlmserve 结果只有 seed13，分别为8 steps（42行）和32 steps（20行）。10行 dInfer 为 seed41。没有保存的128-step逐运行摘要。
- 18行跨模型摘要全是 `shapeshift`，缺少对应模型的 FCFS 逐运行结果。
- 汇总 CSV 中的 n_seeds/std 字段不能替代丢失的 seed17/23 原始结果，也不能据此断言当时没有跑过这些实验。

### 长步数图存在策略与工作负载标签冲突（优先处理）

`paper/plot_fullstep_llada8b.py:6–40` 的数值与已保存 CSV 逐项比较：

| 图中标签 | 图中 tok/s、p95秒 | 现有 CSV 的对应行 |
| --- | --- | --- |
| ShareGPT / FCFS | 20.70、296.8 | `dlmserve_32step_hotpotqa...` / fifo |
| HotpotQA / FCFS | 7.89、778.8 | `dlmserve_32step_sharegpt...` / fifo |
| ShareGPT / ShapeShift | 29.31、209.6 | `dlmserve_32step_hotpotqa...` / **length_sorted_admission** |
| Multitask / ShapeShift | 15.14、405.8 | `dlmserve_32step_multitask...` / **length_sorted_admission** |
| Alpaca / ShapeShift | 116.61、52.7 | `dlmserve_32step_alpaca...` / **length_sorted_admission** |

这些是按图中精度舍入后的数值匹配；不是证明图一定来源于这些文件，但已不足以支持其当前标签。OpenOrca 的长步数图值在已保存32-step CSV 中没有找到相同对应项；不能强行拼接。必须找回来源或重测，不能将 sorted-admission 的收益归给完整 ShapeShift。论文 longer-horizon 正文主要描述 Shape-only/Aging，图却还画了完整 ShapeShift；两者需分别处理。

### dInfer 的 8.41× 归属可追溯到 aging-only

ShareGPT 的 5.1166→43.0492 tok/s、8.4136× 对应已保存 CSV 中 fifo→**aging_bucket**。其余三个 workload 的图中 throughput 也对应 aging_bucket。wrapper 没有 compactness，因此这些数字不能验证完整 ShapeShift 的 compactness 贡献。dInfer 摘要还未记录 mean latency，图中的 mean 值需要 per-request rows 或其他来源。

不要原地修改 camera-ready PDF 或历史 CSV 来消除差异。保留已发布文件的哈希，另建可追溯的修订图/说明与新测量结果。

## 四、按依赖顺序补实验

先完成协议和代码缺口，再投入大规模 GPU 预算。以下是验证计划，均非已完成结果。

| 优先级 | 实验/工程前置 | 对应论文主张 | 必须输出/通过条件 |
| --- | --- | --- | --- |
| P0 | 固定 algorithm scope、beta/tau、baseline身份、active pool、精度；补 SJF-Step/StaticBucket；实现保序和绝对时间 replay | 所有比较的可比性 | 配置文件+commit+trace hash；不得按测试集调参后回报最优点；旧默认无法恢复时标记为新协议 |
| P0 | 真实 LLaDA token/state 正确性：单请求、同长批、异长批、异步插入、取消、长 outlier；对照单请求与FIFO | “只改调度，不改输出” | 相同输入/采样设置，记录逐步 mask 数、最终 token、错误/完成数；出现差异先定位位置编码/attention/数值/early completion，不能只看吞吐 |
| P0 | 修订 replay 与指标：scheduled arrival、actual arrival、first dispatch、每次 dispatch/finish；失败处理和 warmup | 真实在线吞吐、p95、公平性 | 原始 request/step 行；全部策略同一外生 trace；明确排除冷启动；失败率非零不得算成成功提速 |
| P1 | 动机 microbench：3模型 × all-short/all-long/2短2长；B=4，L=256/2048，G=128，128 steps | Motivation 表中 mixed≈all-long | 实际模型路径、GPU计时同步、warmup、重复延迟分布、实际rectangular shape；先LLaDA，再补其他模型 |
| P1 | LLaDA 主矩阵：5 workloads × 6独立策略 × seeds13/17/23 = **90 runs** | 主表、ablation、strong baselines | FIFO、StaticBucket、SJF、LengthBucket(即Shape-only)、Aging、Full；B8/G128；同一输入/到达/精度；per-seed throughput/mean/p95、paired ratio与区间 |
| P1 | 其他3模型 × 5 workloads × FIFO/Full × 3 seeds = **90 runs** | 1.11–2.15×跨模型结论 | 先实现并验证各模型适配；保持同一路径内基线公平，记录实际token长度；不能用LLaDA FCFS归一化其他模型 |
| P1 | 长步数核查：5 workloads × FIFO/Shape/Aging/Full × 3 seeds，分别明确8/32/128 steps | longer-horizon 1.14–1.88× | 128-step部分可复用主矩阵；额外32-step为60 runs；8-step诊断若跑也需60 runs。分开展示，禁止跨steps拼表或以sorted-admission替代Full |
| P1 | dInfer：4 workloads × FIFO/Shape/Aging/Full × 3 seeds = **48 runs** | 同路径组合收益1.36–8.41× | 如要评Full须新增compactness；若保持原wrapper仅能标注Aging结果。固定后端、模型与到达，记录batch wait/执行时间；不与dlmserve绝对延迟横比 |
| P2 | Low/Medium/High CV traces，固定均值/总有效token/到达；4策略 × 3 seeds | padding 1.06→2.09与收益1.08→2.31 | 每步B,Lmax,sumL,rho与GPU时间；同时报告step平均和work-weighted padding，避免混淆聚合口径 |
| P2 | bimodal+长outlier持续到达，固定严格tau；4策略 × 3 seeds | p95 inter-service wait 142.7→57.9秒 | 逐请求每次服务间隔，定义dispatch/finish口径、首末截尾；按长度分组报告，另列最长等待/未完成。tau不是硬SLO保证 |
| P2 | 2 traces × spacing1.5/1/.75 × 4策略 × 3 seeds = **72 runs**；另补100–500ms spacing sweep | burst replay与薄队列边界 | 原始session timestamp或可复现session生成器、绝对时间缩放、队列深度分布、paired speedup/p95 |
| P2 | scheduler/trace开关、pool size、B、beta/tau敏感性 | “CPU开销远低于GPU step”及边界 | schedule/sort/CSV写入分时、端到端日志开销；同长/薄队列反例必须保留，不承诺所有场景加速 |

建议第一阶段只做 LLaDA 的 Alpaca（窄分布）、ShareGPT（长尾）、OpenOrca（最大收益主张），先4策略×3 workload×3 seed=36 runs；补齐强基线后为54 runs。只有完成数、质量、时间协议通过后再扩展到全部矩阵。
配置表的默认128 steps与已保存8/32-step摘要冲突尚未解决：复核时至少明确报告这三个协议，不可自行挑一个称为“原配置”。

## 五、结果归档与最终判定标准

每个运行保存模型/tokenizer revision、CUDA/PyTorch/后端版本、RTX A6000 48GB硬件信息（其他GPU须注明迁移验证）、种子、dtype、完整配置、trace hash、raw request/step记录、GPU/host计时、失败与拒绝、warmup规则及日志开关。
每个图表单元格必须映射到 run ID 与 raw→summary→plot 的脚本，图表不再手填数字。三 seed 为论文协议最低要求，尾延迟置信区间和多次重复按观测方差决定。

可以确认的层次应分开：

1. 同兼容域内算法符合伪代码：本次已获得CPU证据。
2. 真实模型状态/输出语义正确：未完成。
3. 匹配协议下机制和加速趋势重现：未完成。
4. 原论文具体数值复现：需要恢复原配置与trace，并完成来源核查，不能由前三层自动推出。
