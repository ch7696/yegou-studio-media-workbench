# 云迁移清单

统计时间：2026-09-11（当前 WSL 环境）

## 当前占用

| 类别 | 大小 | 说明 |
|---|---:|---|
| Docker 镜像存储 | 130.9 GB | 30 个镜像，包含旧版本、重复标签和悬空构建层 |
| Docker 构建缓存 | 40.03 GB | 不需要迁移，其中约 34.15 GB 可回收 |
| 容器可写层 | 2.8 GB | 主要来自已停止容器，不作为迁移资产 |
| Docker 数据卷 | 538.4 MB | RAGFlow 的 MySQL、Elasticsearch、MinIO、Redis 数据等 |
| 已下载模型 | 32.2 GB | 模型权重，不包含在 Docker 镜像大小中 |
| 工作台代码 | 约 1.2 MB | 已提交 Git，可直接复制或重新拉取 |

## 建议迁移分层

### 第一阶段：核心服务

需要迁移的 Docker 镜像名义大小约 **76.5GB**：

- `ragflow-cosyvoice-speaker:0.0.2`：32.1GB
- `ragflow-qwen-asr:0.0.6`：29.7GB
- RAGFlow：11.4GB
- Elasticsearch：2.07GB
- MySQL、MinIO、Valkey：约 1.2GB

加上选定运行模型约 **24.6GB**，以及 RAGFlow 数据卷，核心迁移包约 **101GB**。如果保留原始 Qwen3-Embedding-4B 版本，再增加约 7.6GB。

### 第二阶段：视觉和知识库扩展

视觉与向量服务还需要以下镜像，名义大小约 **52.7GB**：

- vLLM：28.1GB
- PaddleOCR-VL Pipeline：4.66GB
- PaddleOCR-VL API：1.3GB
- PaddleOCR GenAI vLLM Server：18.6GB

这些服务目前没有全部常驻运行，云端应按显存逐个启动。加入核心服务和模型后，完整运行环境建议按 **150GB 以上**准备。

## 不要直接打包的内容

- Docker 构建缓存 40.03GB：云端重新构建或从镜像仓库拉取即可。
- 旧版 CosyVoice 镜像和悬空镜像：当前工作台使用 `ragflow-cosyvoice-speaker:0.0.2`。
- `G:\\天涯` 原始文章：约 24GB，属于业务资产，单独上传或挂载对象存储。
- ComfyUI 输出目录：约 9.6GB，按需迁移，不属于服务镜像。

## 云端落地方式

1. 创建带 NVIDIA 驱动和 Docker 的 GPU 实例，代码用 Git 迁移。
2. 模型权重放在独立数据盘或对象存储挂载目录，不塞进镜像。
3. 核心镜像优先使用镜像仓库或 `docker save` 导出；不要传构建缓存。
4. 恢复 RAGFlow 的 MySQL、Elasticsearch、MinIO、Redis 数据卷。
5. 将 Windows 路径改成云端 Linux 路径，例如 `F:\\voice` 改为 `/data/voice`。
6. 先启动 RAGFlow、ASR、CosyVoice，再按需启动 OCR、MiniCPM-V、HunyuanOCR 和 Embedding 服务。

## 机器规格建议

- 只跑 RAGFlow、ASR、CosyVoice：24GB 显存更适合并发测试，32GB 系统内存起步。
- 同时做视觉分析、OCR、Embedding：建议 24GB 以上显存；追求速度时拆成两个 GPU 服务更稳。
- 磁盘：核心迁移建议 200GB SSD；包含视觉服务和原始素材建议 300GB 起步，并预留增长空间。

