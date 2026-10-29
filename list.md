# 多模态文档问答系统任务清单

> 范围：根据课程图片中的 doc2md、DocBase、AGENTS.md、后台 Service、Web 应用和度量要求整理。完成一项后勾选，并附实际运行结果或测试证据。最终问答继续使用 DeepSeek；Qwen3 相关能力通过模力方舟提供，不在本机加载 8B 模型。

## 0. 当前基础与待解决阻塞

- [x] 已有 PDF、DOCX、TXT、图片解析，输出 Markdown、图片资产、manifest 和错误日志。
- [x] 已有固定窗口与结构化切块、BM25／Dense／bge-m3 三路索引、RRF、精排、Top 5 引用问答。
- [x] 已有 `parse`、`build`、`query`、`all`、`evaluate` 等 CLI 命令及 Recall@K／MRR 计算程序。
- [x] 已建成可查询的 Qwen3-Embedding-8B snapshot；视觉增强快照 `20261026T093858Z-8329980e` 含 415 个共享 Chunk，三路索引完整，并已通过真实 Query、Reranker、DeepSeek 回答与引用验证。旧服务 snapshot 不复用。
- [x] 已从本地 MMLongBench-Doc 导入 3 份 PDF。2026-10-29 发现原导入器错误地把本来 1-based 的 `evidence_pages` 再加 1；修正后共有 18 条带证据页的问题和 8 条无证据页的问题。旧 17 题评测仅保留审计，不再作为正式结论。

## 1. 模力方舟模型接入（先完成）

- [x] 已实际确认 Qwen3-Embedding-8B、Qwen3-Reranker-8B 与 Qwen3-VL-30B-A3B-Instruct 的模型 ID、接口及短请求可用；持续批量调用仍受账号额度限制。
- [x] 2026-10-29 再次完成三模型最小真实调用与完整 snapshot 端到端 Query：Embedding 为有效归一化 4096 维，Reranker 排序正确，视觉描述非空，Query 输出 Top 5；脱敏记录见 `docs/MOARK_INTEGRATION.md`。
- [x] 在 `.env.example` 增加模力方舟令牌变量；实际令牌只放环境变量或忽略的 `.env`，不写入代码、日志或 snapshot。
- [x] 将 Dense 的 Build 与 Query 改为同一个模力方舟 Qwen3-Embedding-8B 服务；显式请求 4096 维，并完成真实 Build 验证。
- [x] 在 snapshot 中记录服务商、模型 ID、维度、构建配置和 chunk 校验值；Query 严格使用相同模型与 snapshot。换模型时重建索引。
- [x] 保留单 chunk 失败记录和跳过机制；补齐超时、有限重试与调用失败提示，避免生成全零向量。
- [x] 已用真实视觉增强产物完成三路 Build，并在 2026-10-27 实际运行完整 Query、Reranker、DeepSeek 回答和真实引用输出。
- [x] 旧 17 题两阶段锦标赛已完成，但因 Gold 页码整体偏移而废止；不得继续引用 Hybrid 0.0882、Dense 0.3235 等旧指标。
- [x] 已使用修正 Gold 对旧报告保存的 17 题 Top 5 离线重算：BM25 Recall@5 0.6176、Dense 0.7353、bge-m3 0.6882、Hybrid 0.7059；报告：`data/benchmarks/mmlongbench_page_corrected/rescored_saved_17/report.md`。
- [x] 已使用修正后的 18 题 Gold 正式重跑三路检索、RRF 与 Reranker，含此前缺失的 benchmark_row=217；报告：`data/benchmarks/mmlongbench_page_corrected/evaluation_vision/20261029T105551Z-8cc7e5a6/report.md`。BM25 Recall@5 0.6111、Dense 0.7222、bge-m3 0.6778、Hybrid 0.6944。
- [x] 第 12 题 Gold 页码已由用户确认应仅为 PDF 物理第 10 页；已保留原 Gold，生成 `data/benchmarks/mmlongbench_page_corrected/reviewed_q12/gold_reviewed.jsonl` 与离线复核报告。派生标签下 BM25/Dense/bge-m3/Hybrid Recall@5 分别为 0.6389/0.7222/0.7056/0.7222。原始标签的正式评测及 6 PASS / 1 FAIL 验收仍保留，不混用指标。单题精排诊断见 `diagnostics/q12/rerank_diagnostics.md`。
- [x] 第 12 题已在独立完整快照 `20261029T155325Z-5c7d8346` 中修复证据缺口：结构化切块为长表格追加同页图表开头原文，三路索引均为 444/444；原题最终 Top 5 的 `43ed3b8ed03fe220cd1602ae` 同时包含图表标题、46% 和 39%。修正 Gold 的 18 题新快照融合 Recall@5 为 0.7500、MRR 为 0.6852，报告见 `data/benchmarks/mmlongbench_page_corrected/evaluation_table_context/20261029T155804Z-179cd89d/report.md`。原始正式报告与第 12 题人工复核补充报告继续保留，不与新快照指标混为同一次评测；数值仍未经过视觉人工完整核验。
- [x] 已完成修正标签下的零额度保底组合实验：Dense 前 3 + Hybrid 补齐与 Dense 持平；三条单路 Top 5 再 RRF 为 0.7647，但绕过最终 Reranker，不符合规定流程，因此不修改生产排序。

2026-10-29 已用修正后的 18 题正式报告重新运行 Acceptance：项目骨架、解析产物、视觉溯源、完整快照、三路 Chunk 对齐和正式评测共 6 项 PASS；融合质量不退化 1 项 FAIL（Hybrid 0.6944 < Dense 0.7222）。报告：`data/benchmarks/mmlongbench_page_corrected/acceptance/20261029T105944Z/report.md`。

**验收：** 新 snapshot 三路索引均完整；Build 与 Query 的 Qwen3 模型 ID 一致；旧 snapshot 不被误用；不依赖本机加载 Qwen3 8B 权重。

## 2. doc2md 多模态解析

- [x] 已增加 PPTX 读取：按幻灯片顺序输出文本、标题、备注（若有）及图片资产，保留 slide 编号且不伪造 PDF 页码；合成样例通过。
- [x] 已使用一份真实 7 页 PPTX 验证标题、正文、图片资产和 slide 编号；发现并修复组合形状未递归解析的问题。修复后 43 个图片资产完整、错误为 0、无伪造页码；源文件没有 speaker notes，备注功能仍由合成测试覆盖。报告：`data/validation/pptx/VALIDATION_REPORT.md`。旧 `.ppt` 不在本次支持范围。
- [x] PDF、DOCX、PPTX 图片资产已记录 `asset_id`、来源文档、可得的页码／幻灯片、附近文字；无法确认的图题留空，不猜测。
- [x] 已实现独立视觉增强流程：渲染人工标注证据页并调用模力方舟 Qwen3-VL-30B-A3B-Instruct；Markdown 明确标记模型、`unreviewed` 状态并保留原图引用，不保存 raw response。
- [x] 视觉描述按真实 `PAGE` 标记放回副本 `parsed_vision`，不修改原解析产物；3 文档试点 22/22 页成功且资产、页码、描述一一对应。
- [x] 已对 3 份真实 PDF 做离线资产审计：48 页、327 个图片资产均可定位，22 个视觉描述保留 `unreviewed` 标记，`errors.jsonl` 均为空；人工查看页 14/15 后确认页面与资产对应。报告：`data/benchmarks/mmlongbench_page_corrected/PARSE_AUDIT.md`。
- [x] 已逐项核验真实 PDF 表格页、图表页、图片型扫描页和真实 PPTX 第 2 张的 Markdown、资产文件与页／slide 映射。发现扫描信息图第 4 页只有 16 个图片资产而无 OCR 文本、复杂表格列关系不稳定、PPT 图片 OCR 噪声及图表视觉描述仍待人工核验；详见 `data/validation/parse_cases/CASE_AUDIT.md`。
- [x] 为无原生文本的 PDF 页增加可选本地整页 OCR：`python app.py parse --ocr-scanned-pages`。真实扫描信息图的独立副本实测得到 152 字符、0 解析错误，来源渲染图和 `unreviewed` 标记齐全；小字仍有误识，图表数值仍须人工逐项比对，见同一审计报告。

**验收：** 能从检索结果定位到原文页／幻灯片与原图；纯 OCR 文本和视觉模型描述可区分。

## 3. 检索、问答与质量评测

- [x] 图片语义文本已进入独立的 415-chunk 完整 snapshot `20261026T093858Z-8329980e`，并保留文档与页码映射。
- [x] 代码核验保持 BM25、Dense、bge-m3 各取 Top 20 → RRF(k=60) Top 50 → 精排 Top 5；不同路分数不直接相加。
- [x] 旧 17 题报告继续只作审计。修正 Gold 的 18 题四路评测已实际完成；同一 18 题 Gold 的候选覆盖分析已重跑（最终命中 14 题、RRF Top 50 未召回 3 题、候选已召回但精排淘汰 1 题），见 `data/benchmarks/mmlongbench_page_corrected/retrieval_analysis/report.md`。旧三条复核涉及的原 PDF 物理页码已重新核对，见 `OLD_REVIEW_SPOTCHECK.md`；旧复核数值未沿用。
- [x] 已核对原始 benchmark：`evidence_pages` 为 1-based，与解析器 `PAGE:n` 直接对应；导入器、测试和派生 Gold 已修正，旧 Gold 未覆盖。
- [x] 已用同一 MMLongBench-Doc 小样本离线对比固定窗口／结构化切块的 BM25：两者均为 Recall@5 0.0529、Recall@20 0.4471、MRR@5 0.0735；该样本解析结果无 Markdown 标题，因此两种模式实际等价。
- [x] 已用真实 PPTX 第 2–4 张的同一已解析语料（一级 slide 标题与从实际标题提升的二级 Markdown 标题）分别 Build 固定窗口与结构化快照：8/8 和 11/11 Chunk 均完成三路索引。三题同 Gold 的 BM25、bge-m3、Qwen3 Dense、融合精排 Recall@5/MRR 均为 1.0000；结构化 Chunk 跨 slide 数为 0，固定窗口为 3。小样本不证明通用质量提升，详见 `data/validation/chunk_modes/COMPARISON.md`。
- [x] 已实际运行 DeepSeek 引用问答，输出引用编号只指向本次 Top 5；自动化测试覆盖资料不足、伪造引用回退和 API 失败时保留检索结果。

**验收：** 测试报告可复现到标注集与 snapshot；引用能打开对应来源，不出现编造的页码或图片结论。

当前结论：修正 Gold 的 18 题正式评测已完成，结果可复现；融合质量仍低于 Dense，因此第 3 部分不能整体标记完成。

## 4. AGENTS.md 调度约定

- [x] 编写项目级 `AGENTS.md`：约定何时检索、如何组合文本与图片证据、何时请求用户补充、如何处理资料不足。该文件是开发约定，运行时仍由 QA prompt 控制。
- [x] 已在代码与测试中区分 Context prompt、模型响应、引用解析和最终输出；服务事件日志只保存类型、成功状态和耗时，不保存密钥、问题或原文。
- [x] 已验证有引用回答、资料不足、伪造引用回退和来源冲突提示；视觉描述已接入检索，图像证据仍明确标记为未经人工核验。

**验收：** 对同一测试问题能够追踪检索输入、选中证据、最终引用及失败原因。

## 5. 后台 Service 与 Web 界面

- [x] 使用范围已确定为可信局域网多人共享、无需登录；服务默认仅绑定本机，局域网绑定需显式指定。所有可访问者都能查看和上传，已在界面与 README 提示。
- [x] 增加仅限服务所在电脑访问的模型与密钥管理页面；局域网其他设备不能读取或修改设置。Windows 密钥使用当前用户 DPAPI 加密并保存在仓库外，接口只返回密钥状态。CLI 与 Web 共用该运行配置，且强制 DeepSeek/Qwen3 的既定角色，防止静默换模型。
- [x] 已提供文档列表、上传／已有来源解析、构建状态、提问、答案和来源 Markdown；文档可设置共享分类，分类仅保存 doc_id 与标签，不改解析产物。
- [x] Web 解析可显式选择 Qwen3-VL 图片描述；描述绑定原始图片资产和页码／幻灯片，标记为未经人工核验，单图失败只记录错误且不丢弃本地解析结果。
- [x] 已实现共享多轮会话的恢复、切换和删除；最多携带最近 20 轮／约 48,000 字符帮助理解追问，每轮仍从所选完整 snapshot 重新检索 Top 5，历史回答不作为证据。当前无登录，所有局域网用户可查看和删除共享会话。
- [x] Build 已放入串行后台任务，保存状态、错误数和报告路径，重启时标记中断；失败／部分成功／中断任务支持按已保存参数重试，页面可查看最多 100 条结构化错误摘要。
- [x] Build 增加切块、BM25、Qwen3 Dense、bge-m3 的真实完成数／总数回调；页面按阶段显示，不估算模型加载进度。
- [x] 引用可打开来源 Markdown；最终 Top 5 的每条来源按页／幻灯片关联 manifest 中实际存在的图片资产，并只读打开。若解析产物与 snapshot 来源哈希不符，不显示图片关联。
- [x] 匿名统计已有请求次数、失败次数、耗时和 DeepSeek API 实际返回的 Token 用量；逐问题好评／差评只按 turn_id 保存评分。未返回 usage 的调用显示为未知，不估算；事件日志不保存问题正文、原文或密钥。
- [x] 2026-10-30 使用隔离的无敏感 PDF 实际走通上传解析、共享分类、完整三路 Build、DeepSeek 带引用回答、同页图片打开、Token usage 与好评保存；记录在 `data/validation/web_flow_smoke/smoke_summary.json`。发现并修复任务列表按随机 ID 排序的问题；浏览器手工点击与跨设备验收仍待做。

**验收：** 从添加文档到提问与回看来源可通过界面完成；服务重启后任务与 snapshot 状态仍可辨认。

## 6. 最终提交物（源码、可执行文件、介绍材料）

### 6.1 源码包

- [x] 已整理并推送 GitHub 仓库，制作源码压缩包；包含 `app.py`、`src/`、`tests/`、`requirements.txt`、`config.yaml`、`.env.example`、`README.md`、`AGENTS.md`、`list.md` 和 `SUBMISSION_PLAN.md`。
- [x] Git 索引及源码包均已检查：没有 `.env`、真实 API Key、DPAPI 密钥文件、私人文档、模型缓存、FAISS 大索引、会话数据库或请求日志；`data/validation/` 已加入忽略规则。
- [ ] 确认克隆源码后可以按照 README 创建 Python 3.10+ 环境并运行 CLI 与 Web。
- [x] 已保存源码交付 Git commit ID；源码包内 `SOURCE_COMMIT.txt` 记录对应版本。
- [ ] 如课程要求附带评测产物，单独提供脱敏压缩包；不要把被 `.gitignore` 忽略的大型 `data/` 直接提交 GitHub。

### 6.2 Windows 可执行文件

- [x] 已制作 Windows `onedir` 试运行发行包 `dist/TraceDocRAG/`，不使用难以排错的单文件模式；入口支持启动 Web，CLI 继续通过源码运行。
- [x] 可执行包包含 Web 静态页面、默认配置和 `.env.example`，不包含真实密钥、原始文档、历史会话或个人配置。
- [ ] 明确本地 bge-m3、FAISS、PyTorch／SentenceTransformers 会显著增大体积；打包前先做最小 PyInstaller 试验，不在截止前盲目打包全部模型权重。
- [ ] 模型权重和现有索引默认作为外部资源，由 README 说明放置位置；首次运行缺资源时应显示明确错误，而不是静默失败。
- [x] 已提供 `start_web.bat` 作为依赖现有 Python 环境的可读备用启动方式；它不被宣称为独立 `.exe` 成品。
- [x] 已增加 `TraceDocRAG.spec`、`web_launcher.py` 和独立构建依赖文件，并完成实际 onedir 构建；当前试运行包约 938.9 MiB。
- [ ] 已在本机构建目录验证启动、首页及管理接口、读取配置和正常退出；仍需复制到全新目录并在另一台 Windows 电脑验证。
- [x] 已在发行目录 `BUILD_INFO.txt` 记录大小、SHA256、Python 版本和 Windows 架构，并在 `DISTRIBUTION.md` 记录构建与运行方法。

### 6.3 介绍材料

- [x] 已更新介绍文档 `docs/PROJECT_INTRO.md` 和 Word 版 `docs/TraceDoc_RAG_项目介绍.docx`：写入 18 题正式报告、原始与第 12 题人工复核后的两套指标、6 PASS / 1 FAIL 验收及答案证据不足限制。2026-10-29 通过本机 Word 导出 PDF（11 页）核验封面和指标页；逐页图示与截图仍待补齐。
- [ ] 为介绍材料准备并插入真实项目截图：文档解析结果、Build 完整 snapshot、Query 最终 Top 5、DeepSeek 带引用回答、Web 首页与问答页、本机模型配置状态、正式评测报告。
- [ ] 至少加入一张系统流程示意图，并让截图标题说明对应功能、输入和结果；Word 中的图片应保持清晰、比例正确，不出现跨页截断或难以辨认的小字。
- [ ] 截图前统一使用演示数据并检查脱敏：不得暴露 API Key、`.env` 内容、DPAPI 文件、私人路径、个人文档、会话隐私或无关桌面信息。
- [ ] 从已保存的 18 题正式指标和 Acceptance 报告制作脱敏截图，插入 Word；标注生成日期和 snapshot。
- [x] Word 已分别展示原始正式指标与第 12 题人工复核补充指标，并明确原始标签下 Hybrid 低于 Dense；没有只展示较高的复核结果。
- [ ] README 与 Markdown 介绍材料已记录第 12 题新快照的完整数值证据，以及旧快照的缺口；提交前仍需同步更新 Word 版，保留原始成绩、复核补充成绩和新快照成绩三者的区别。扫描页 OCR 误识及视觉图表描述 `unreviewed` 的限制继续保留。
- [x] 介绍材料已包含 3–5 分钟演示脚本：启动系统 → 选择完整 snapshot → 提问 → 查看 Top 5 → 查看 DeepSeek 回答及引用 → 打开来源；现场演练仍待完成。
- [x] 介绍材料已包含无额度演示方案：使用已有截图、正式报告和保存的评测产物；不伪造现场模型调用成功。
- [x] 介绍材料末尾已列出环境要求、API 服务依赖、隐私边界、已知问题和提交物；最终截图与版本号仍待补齐。

### 6.4 提交前总验收

- [ ] 源码包、可执行发行包、介绍材料三个交付物使用一致的项目名称和版本号。
- [ ] 对源码压缩包和可执行发行包执行密钥扫描及恶意／误报检查。
- [ ] 在不修改仓库文件的情况下按最终演示脚本完整走一遍。
- [ ] 检查所有链接、文件名、截图、版本号、SHA256 和 GitHub 地址。
- [ ] 保留一份本地备份，并在提交平台下载回传文件再次核对可打开性。

**验收：** 老师获得源码、Windows 发行包和介绍材料后，能够理解系统、启动演示入口、查看真实评测与已知限制；所有交付物均不包含真实 API Key 或私人数据。

## 7. 后续扩展（基础验收后再决定）

- [ ] 根据评测和实际使用证据，再决定是否加入查询扩展、检索缓存、长期记忆、多轮 Agent、GraphRAG 或质量仪表盘。
- [ ] 每项扩展先定义基线和回退方式，单独验证收益与数据保存风险。

## 实施约束

- DeepSeek 只作最终问答模型，不作 Embedding；Qwen3 相关模型调用模力方舟。
- 模力方舟处理 Embedding、视觉描述或精排时，会收到相应文档片段或图片；接入前确认资料可发送及账号额度。
- 模型切换必须新建 snapshot，不能直接改旧索引元数据。任何质量指标必须来自真实标注和实际运行。
