# 模力方舟模型接入验收

日期：2026-10-29
服务地址：`https://api.moark.com/v1`

## 角色约束

| 项目角色 | 服务商 | 模型 | 验收结果 |
| --- | --- | --- | --- |
| Dense Embedding | 模力方舟 | `Qwen3-Embedding-8B` | PASS |
| Reranker | 模力方舟 | `Qwen3-Reranker-8B` | PASS |
| 图片描述 | 模力方舟 | `Qwen3-VL-30B-A3B-Instruct` | PASS |
| 最终问答 | DeepSeek | `deepseek-chat` | 角色保持不变，本次未调用 |

DeepSeek 不用于 Embedding。bge-m3 保持本地模型，不静默切换服务商。

## 最小真实调用

- Embedding 返回 4096 维有限向量，L2 归一化检查通过。
- Reranker 返回 2 个有限分数，相关文本排在无关文本前。
- 视觉模型返回非空描述，测试只记录字符数，不保存响应正文。
- 三个模力方舟角色均从同一个 `MOARK_API_KEY` 安全来源读取密钥；密钥值未打印或写入报告。

## 端到端 Query

使用完整 snapshot `20261026T093858Z-8329980e` 运行一次 `--retrieval-only` Query：

1. BM25 Top 20 完成。
2. Qwen3 Dense Top 20 完成。
3. 本地 bge-m3 Top 20 完成。
4. RRF 候选融合完成。
5. Qwen3 Reranker 完成。
6. 最终输出 5 个 Chunk。

该验证没有调用 DeepSeek，也没有把 API Key、Prompt、文档文本或原始模型响应写入本报告。

## 配置与安全

- 模型、服务商、HTTPS 主机和超时由 `src/utils/model_settings.py` 校验。
- `.env` 已被 Git 忽略；`.env.example` 只保存占位符。
- Windows Web 配置使用当前用户 DPAPI，把密钥保存在仓库外。
- Snapshot 记录 Embedding 服务商、模型、维度和构建校验值；Query 拒绝模型配置不匹配的 snapshot。
- Embedding 失败不写全零向量，单个失败记录到错误日志。

## 后续额度使用

接入已经完成。后续需要额度的工作是使用修正后的 18 题 Gold 重新运行正式评测，不属于接入功能缺失。
