# TraceDoc RAG Windows 发行包

此目录是 TraceDoc RAG 的 Windows onedir 发行包。它提供 Web 启动入口，不包含模型权重、索引、原始文档、历史会话或真实 API Key。

## 启动

在当前目录打开 PowerShell：

```powershell
.\TraceDocRAG.exe
```

浏览器访问 `http://127.0.0.1:8000`。

仅在可信局域网中共享时，显式指定本机局域网地址：

```powershell
.\TraceDocRAG.exe --host 192.168.1.10 --port 8000
```

不要把无登录的服务直接暴露到互联网。

## 数据与配置

- 把待解析文件放入 `data/raw/`，或从 Web 上传。
- 解析产物、索引和共享会话保存在当前目录的 `data/` 下。
- 本地 bge-m3 权重可放入 `data/models/bge-m3/`。
- `config.yaml` 是默认问答配置；不要在其中写 API Key。
- API Key 应通过服务电脑上的模型设置页面配置，或使用本机环境变量。
- `.env.example` 仅为变量名示例，不含真实密钥。

完整 Build 和 Query 仍依赖 MoArk、DeepSeek 网络服务以及相应额度。视觉描述未经人工复核时只作为辅助检索文本。

## 已知限制

- 发行包体积较大，因为包含 PyTorch、ONNX Runtime、OpenCV、FAISS 和本地 OCR 运行库。
- 模型权重与已有 FAISS snapshot 不随发行包提供，需要单独放置或重新构建。
- 当前 Web 无登录，局域网用户共享文档和会话。
- CLI 的完整开发与评测命令继续通过源码版 `python app.py ...` 使用。
