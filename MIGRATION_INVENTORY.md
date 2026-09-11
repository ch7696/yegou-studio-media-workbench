# 云端媒体工作台迁移清单

统计时间：2026-09-11（当前 WSL 环境）

## 迁移边界

云端只运行媒体处理链路：视频/音频上传、ASR、说话人和时间戳处理、CosyVoice 配音、MiniCPM-V 视觉分析、HunyuanOCR 和 PaddleOCR-VL。RAGFlow、知识库、Embedding、数据库、搜索服务和文章库继续留在本机。

## 当前本机占用

| 类别 | 大小 | 处理方式 |
|---|---:|---|
| Docker 镜像存储 | 130.9 GB | 含旧版本、重复标签和悬空镜像，不能直接视为迁移包 |
| Docker 构建缓存 | 40.03 GB | 不迁移，云端重新构建或从镜像仓库拉取 |
| 容器可写层 | 2.8 GB | 不迁移 |
| RAGFlow 数据卷 | 538.4 MB | 留在本机，不迁移 |
| 已下载模型 | 32.2 GB | 按媒体链路选择性迁移 |
| 工作台代码 | 约 1.2 MB | 通过 Git 迁移 |
| `G:\\天涯` 原始文章 | 约 24 GB | 留在本机，除非云端也要处理这些文章 |
| ComfyUI 输出 | 约 9.6 GB | 按需迁移原始媒体 |

## 云端实际需要的内容

### 语音配音基础包

Docker 镜像名义大小约 **61.8GB**：

- `ragflow-cosyvoice-speaker:0.0.2`：32.1GB
- `ragflow-qwen-asr:0.0.6`：29.7GB

模型约 **12.7GB**：

- Fun-CosyVoice3-0.5B：9.1GB
- Qwen3-ASR-0.6B：1.8GB
- Qwen3-ForcedAligner-0.6B：1.8GB

语音配音基础包合计约 **74.5GB**，加上系统、临时音频和输出文件，云端建议准备 **150GB SSD**。

### 完整媒体分析包

在语音配音基础包上，再加入视觉服务镜像约 **52.7GB**：

- vLLM：28.1GB
- PaddleOCR-VL Pipeline：4.66GB
- PaddleOCR-VL API：1.3GB
- PaddleOCR GenAI vLLM Server：18.6GB

视觉模型约 **9.2GB**：

- MiniCPM-V 4.5 GPTQ 4-bit：6.7GB
- HunyuanOCR-1.5：2.5GB

完整媒体处理环境约 **136.4GB**，不包含原始视频和音频。云端建议准备 **250GB SSD**，持续处理大量媒体时准备 **300GB 以上**更合适。

## 明确留在本机的内容

- RAGFlow 镜像及其 MySQL、Elasticsearch、MinIO、Valkey 服务。
- RAGFlow 数据卷和本机知识库。
- Qwen3-Embedding-4B 原版与 NVFP4 版；云端不做向量检索时不需要迁移。
- `G:\\天涯` 文章库，除非后续决定让云端处理文章。
- Docker 构建缓存、旧版 CosyVoice 镜像和悬空镜像。
- AnythingLLM、Ollama 等本机辅助服务，除非另行决定迁移。

## 推荐迁移步骤

1. 云端创建带 NVIDIA 驱动、Docker 和 CUDA 运行时的 GPU 实例。
2. 通过 Git 迁移 `voice-asset-workbench`、CosyVoice 和 3D-Speaker 代码。
3. 模型放在独立数据盘的 `/data/models`，不要写进镜像层。
4. 先部署 ASR 和 CosyVoice，确认配音链路后再部署视觉/OCR服务。
5. 将 Windows 路径改成 Linux 路径，例如 `F:\\voice` 改为 `/data/voice`。
6. 原始视频、音频和生成结果单独放 `/data/media`，不要和 Docker 根盘混在一起。

## 机器规格建议

- 只做语音配音：24GB 显存、32GB 内存起步。
- 同时做视频视觉分析和 OCR：建议 24GB 以上显存、64GB 内存。
- 追求速度时，ASR/TTS 与视觉模型拆到不同 GPU；16GB 显存适合串行处理，不适合全部模型同时常驻。
