# 野构 Studio · 视频导演拉片分析工作台

<p align="center">
  <img src="assets/brand/yegou-studio-logo.png" width="260" alt="野构 Studio 创意工作室标志">
</p>

<p align="center"><strong>视频导演拉片分析工作台</strong><br>把视频拆成画面、声音和时间线，方便回看、复盘与继续创作。</p>

<p align="center"><a href="https://space.bilibili.com/248935331?spm_id_from=333.1007.0.0">B站主页：野构 Studio</a></p>

## 项目简介

这是野构 Studio 用来做视频导演拉片的工作台。上传视频后，系统会分别分析画面和声音，再把它们合成一条完整时间线，最后输出一套方便回看、复盘和继续创作的资料。WebUI 支持视频上传、区间截取、任务进度和 GPU 状态查看；刷新页面后会自动恢复最近任务并继续跟进。

## 核心能力

- 每秒提取一张高清视频帧，并结合连续时间上下文分析镜头。
- 用 ASR 和 ForcedAligner 把语音整理成带时间戳的文本。
- 将画面变化、语音内容和时间码合并成一条可回看的分析时间线。
- 同时保留 Markdown、JSONL、SRT、VTT、逐帧图像和联系图等资料。
- 页面会显示任务进度、运行日志和 GPU 状态，任务结束后也可以释放显存。
- MiniCPM-V 4.7-1B 模式复用常驻 ASR 与 ForcedAligner，连续任务无需反复加载或停止视觉模型。
- 任务状态写入服务端并在浏览器保存任务索引，刷新页面或切回标签页后可恢复进度、日志和输出。

## 批量处理

主页提供“批量导演拉片”和“批量字幕提取”两个入口。批量工作台支持连续添加视频、读取整个文件夹，并在提交前通过拖动或上下按钮调整顺序。提交后由服务端单 GPU FIFO 队列逐项执行；等待中的任务仍可调整顺序，某个视频报错时会记录失败原因并自动跳过，继续下一个视频。

当前流程没有固定时长上限，两小时视频可以直接提交。按每秒一帧计算，约需处理 7,200 张画面，任务耗时和输出占用会明显增加，适合放入队列后台运行。

## 分析输出

每个任务会生成四份主要文档：

| 文件 | 内容 |
| --- | --- |
| `01_纯视觉分析.md` | 只看画面的镜头时间线 |
| `02_纯ASR与时间戳.md` | 语音转写、分段文本和时间戳 |
| `03_代码综合时间线.md` | 画面与声音合并后的完整时间线 |
| `04_最终导演分析.md` | 用于继续整理和深化的导演分析稿 |

原始帧、联系图、索引文件和 ASR 中间结果也会一并保留，方便随时回看和二次处理。

## 使用方式

准备好 Docker、NVIDIA Container Toolkit 和模型文件后，执行：

```bash
cp .env.example .env
./scripts/link_models.sh
./scripts/run_model.sh
./scripts/run_webui.sh
```

启动后打开 <http://localhost:7877>。模型目录通过 `MODEL_ROOT` 配置，结构如下：

```text
${MODEL_ROOT}/MiniCPM-V-4_5-GPTQ
${MODEL_ROOT}/Qwen3-ASR-0.6B
${MODEL_ROOT}/Qwen3-ForcedAligner-0.6B
```

不需要打开 WebUI 时，也可以直接运行流水线：

```bash
python3 media_analysis_pipeline.py \
  --source /path/to/video.mp4 \
  --output /path/to/output
```

加上 `--max-duration 60` 可以只分析前 60 秒；已有产物时，可以用 `--package-only` 重新整理文档。

## Agent CLI（最小入口）

运行中的 WebUI 服务也是 agent 的任务服务，无需打开浏览器。在 `video/` 目录调用：

```bash
# 提交并等待完成；stdout 仅一条 JSON，进度写入 stderr。
python3 agent_cli.py analyze /mnt/c/path/to/video.mp4 > result.json

# 后台提交，返回 JSON 中的 id 可用于后续查询或等待。
python3 agent_cli.py analyze /mnt/c/path/to/video.mp4 --no-wait
python3 agent_cli.py status JOB_ID
python3 agent_cli.py wait JOB_ID --timeout 7200

# 可选区间、字幕模式；默认导演拉片，完成后保留模型常驻。
python3 agent_cli.py analyze /mnt/c/path/to/video.mp4 --start-sec 10 --end-sec 20
python3 agent_cli.py analyze /mnt/c/path/to/video.mp4 --mode subtitle
```

客户端依赖现有 `requirements.txt` 中的 requests；默认连接 `http://127.0.0.1:7877`，
可用 `YEGOU_URL` 或命令前的 `--url` 覆盖。使用服务当前选中的视觉模型，
和 WebUI 共用单 GPU 队列。输入路径属于运行 CLI 的机器，视频通过 HTTP 上传。
Windows agent 可通过 WSL 调用：

```powershell
wsl.exe -d Ubuntu-26.04 --exec python3 /home/administrator/voice-asset-workbench/video/agent_cli.py analyze /mnt/c/path/to/video.mp4
```

结果 JSON 包含 `id`、`status`、`output_path`（服务端路径）以及 `files` 的绝对下载 URL。
agent 可读取 `03_代码综合时间线.md` 继续分析；`04_最终导演分析.md` 仍是待深化稿。
退出码：0 表示成功或成功提交/查询，1 表示任务失败/取消，2 表示输入或 HTTP 错误，
124 表示等待超时。超时不会取消服务端任务，可凭任务 ID 再次 `wait`。
`--timeout` 限制轮询等待时间（不含上传，每次 HTTP 请求超时 60 秒）。
仅在明确需要释放三个模型时添加 `--release-gpu`。

工具命令同样返回 JSON，模型操作默认等待完成：

```bash
python3 agent_cli.py model-status       # 最近模型操作、模型注册表、显存和活动任务
python3 agent_cli.py queue              # 当前任务和排队列表
python3 agent_cli.py release            # 停止本项目视觉、ASR、对齐模型，WebUI 保持在线
python3 agent_cli.py restart            # 重启视觉模型；常驻策略启用时加载 ASR 和对齐模型
python3 agent_cli.py restart --no-wait  # 后台启动，用 model-status 查看操作结果
python3 agent_cli.py retry JOB_ID       # 复用服务端原视频、模式和区间，创建新任务并等待
python3 agent_cli.py retry JOB_ID --no-wait
```

`release` / `restart` 同样支持 `--timeout` 和 `--poll-interval`，超时不会终止后台操作。
`model-status` 的 `model.status` 是最近一次管理操作的状态，并非实时容器健康检查。
运行中的任务会限制模型操作，服务端拒绝时 CLI 返回 JSON 错误和退出码 2。
`retry` 仅允许已完成、失败或取消的任务；原上传必须仍在服务端，旧任务与产物保持不变，
返回的新任务含 `retry_of`。重试使用当前选中的视觉模型，默认保持常驻，
可添加 `--release-gpu` 要求新任务结束后释放。批量中的单项重试会成为独立任务。

## MiniCPM-V 4.7-1B

4.7 需要包含原生适配的 vLLM 开发版。当前注册表绑定官方构建
`0.32.0rc1.dev277+g642c18d64`（提交 `642c18d64`）与 Transformers `5.19.0`，
可在 WSL 内独立 Docker 中构建并启动：

```bash
./scripts/build_minicpm47_runtime.sh
MODEL_ID=minicpm-v-4_7-1b ./scripts/run_model.sh
```

权重目录为 `${MODEL_ROOT}/MiniCPM-V-4.7-1B`，本地镜像名为
`yegou-vllm:minicpm47-642c18d64`。构建脚本会校验官方 wheel 的 SHA256。
配置为 16K 上下文、每次最多 5 张图片、显存利用率 0.40，并默认关闭思考输出，
适用于现有的 5 帧导演分析和字幕提取。WebUI 模型页可切换到此模型。
当前使用图片帧输入，将原生视频输入设为 0，避免启动时预热默认的视频负载。

如果 `.env` 中已有 `MINICPM_IMAGE`、`MINICPM_MODEL` 或性能覆盖项，先按需移除；
这些变量的优先级高于模型注册表。其他模型继续使用各自的注册表配置。

2026-10-16 在 RTX 5060 Ti 16GB 上验证了文本、单图、5 图请求与原导演分析脚本。
5 张 1280×720 测试帧的简短描述请求在预热后耗时约 1.14 秒；首次请求包含内核编译，耗时更长。
1B 在复杂导演提示下仍会误判运动方向和画面细节，分析正文需要人工复核。

同日使用本机 ComfyUI 的 8 段真实素材完成性能测试。初轮 eager 基线详见
[素材实测报告](docs/minicpm47-performance-20261016.md)。后续对照取消强制 eager，
启用默认 Torch 编译与 CUDA Graph：固定 256 tokens 的解码从约 57 提升至 231–236 tokens/秒；
原图 5 帧长描述中位耗时 3.99→1.89 秒，960 像素 5 帧短描述 1.24→0.60 秒；
同一段 8 秒视频的原始完整流程 14.34→7.15 秒。详见
[编译与 CUDA Graph 对照报告](docs/minicpm47-graph-20261016.md)。

4.7 的注册表已设置 `enforce_eager=false`，其他模型保留原先默认。
编译缓存通过 `~/.cache/minicpm-v47` 持久保存；首次启动需要编译、预热与图捕获。
可用 `MINICPM_ENFORCE_EAGER=1` 回退调试，用 `MINICPM_RUNTIME_CACHE_DIR` 指定缓存目录。
单任务队列适合当前配置；精确动作、景别与时间码仍需复核。

4.7-1B 的导演拉片已接入 `evidence-v1` 提示词试用版：保留原尺寸逐秒帧、
20 秒联系图和 5 秒细节组，优先描述主体、动作前后状态、关键道具与可确认的画面变化。
片段范围由程序标记，不再要求模型编排逐秒时间码；联系图允许包含多个镜头，
细节请求使用 temperature=0。每次输出的 `测试视频信息.md` 与 `视觉分析配置.json`
记录所用模型和提示词版本。它仍可能漏报或误述动作，原帧是复核依据。
可在 `.env` 设置 `DIRECTOR_PROMPT_PROFILE=legacy` 回退旧版；WebUI 读取该环境设置需重启。
字幕提取使用独立提示词。第四份最终导演分析仍为人工/后续大模型解释层，未自动完成创作判断。

## 三模型常驻与连续分析

MiniCPM-V 4.7-1B 默认启用常驻音频模式：视觉服务保持运行，
Qwen3-ASR-0.6B 和 Qwen3-ForcedAligner-0.6B 在独立的
`vision-audio-resident` 容器内加载一次，后续视频复用同一进程。
任务仍按单 GPU 队列顺序执行，音频和视觉阶段依次推理，避免并发峰值。
默认任务完成后保留模型；勾选“分析完成后释放显存”或点击释放按钮，会停掉视觉和常驻音频容器，WebUI 保持在线。

`ASR_EXECUTION_MODE=auto` 仅为 4.7-1B 启用该模式；切换较大的视觉模型时，
自动停止常驻音频并回到原先分阶段释放的流程。
可用 `ASR_EXECUTION_MODE=sequential` 回退旧流程；`resident` 可显式启用常驻模式，需自行留足显存预算。
音频服务仅绑定本地 `127.0.0.1:8003`，模型路径与最大输出由现有 `MODEL_ROOT`、
`ASR_MAX_NEW_TOKENS` 设置提供；服务会检查运行时、模型路径和代码版本以决定是否重建。
音频产物格式仍为 JSON/SRT/VTT/Markdown，继续由原有代码合并时间线。

## 视觉字幕提取变体

在 WebUI 中选择“视觉字幕提取”或“批量字幕提取”，即可只调用 MiniCPM-V 检查画面中的对白/旁白字幕，不启动 ASR。它会为每秒证据帧生成结构化观察，并合并为可交给其他配音项目的字幕时间轴：

```bash
python3 subtitle_extraction_pipeline.py \
  --source /path/to/video.mp4 \
  --output /path/to/subtitle-output
```

输出包括 `字幕提取时间轴.json`、`字幕提取时间轴.srt`、`字幕提取时间轴.vtt`、`字幕提取报告.md` 和 `字幕提取原始结果.jsonl`。当前边界依据每秒画面采样估计；JSON 保留逐帧判断，方便后续配音项目复核和微调。

## 项目结构

```text
app/                         WebUI 与视觉分析模块
media_analysis_pipeline.py   分析流程总控
subtitle_extraction_pipeline.py  视觉字幕提取流程
audio_timeline.py            ASR 与 ForcedAligner 流程
scripts/                     模型链接与服务启动脚本
deploy/                      GPU 部署配置
assets/brand/                野构 Studio 品牌素材
```

## 许可证与品牌

本项目源代码采用 [Apache License 2.0](LICENSE)。`assets/brand/` 中的野构 Studio 标识、Logo 及相关品牌素材不随 Apache-2.0 授权，具体说明见 [NOTICE](NOTICE)。
