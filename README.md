# Voice Asset Workbench

本项目把语音处理整理成两个入口：先用“识别并提取语音资产”整理说话人音色，再用“按说话人替换视频配音”完成时间轴配音和导出，运行在当前 WSL Docker 环境中。

识别链路负责上传音频或视频、抽取完整音轨、可调清洁、Sortformer / 3D-Speaker 说话人识别、Qwen3-ASR-1.7B + ForcedAligner 时间戳识别，并把每个说话人保存为可读的资产目录。TTS 链路只读取已经生成的说话人资产，不会因为打开页面就重新分析源视频；重新配音工作台则按任务需要调用两条链路。

## 目录

```text
app/
  audio.py            音轨清洁、样本切片、可选目标时长后处理
  reference.py        参考音频扫描、VAD 候选、说话人筛选和质量记录
  diarization.py      Sortformer / 3D-Speaker 说话人识别和统一 frame activity
  asr.py              Qwen3-ASR 参考台词和词级时间戳
  asr_server.py       隔离的 Qwen3-ASR + ForcedAligner 服务
  dubbing.py          全片时间轴、参考音色绑定、说话人替换、两路混音和视频导出
  timeline.py         SRT / VTT / 视觉项目 JSON 时间线导入
  timeline_dubbing.py 无声视频的时间线配音、音轨渲染和视频封装
  separation.py       Hybrid Demucs 人声 / 背景两路分离
  identification.py   识别链路和资产清单
  tts.py              CosyVoice3 官方调用封装
  main.py             FastAPI 接口
web/index.html        两条流水线的主页
web/assets.html       识别、清洁、切片和保存语音资产页面
web/studio.html       重新配音工作台页面
web/studio.js         时间轴分析和批量配音交互
web/timeline.html     无声视频时间线配音页面
scripts/run_webui.sh  Docker 启动脚本
```

运行时数据在仓库外的 `/home/administrator/ragflow-managed/state/speaker-webui`，模型也不进入 Git。Windows 默认输出目录是 `F:\voice`，在容器中对应 `/mnt/f/voice`。

## 启动

```bash
bash /home/administrator/voice-asset-workbench/scripts/run_webui.sh
curl http://127.0.0.1:7866/api/health
```

需要根据 Dockerfile 重建镜像时执行：

```bash
BUILD_IMAGE=1 bash /home/administrator/voice-asset-workbench/scripts/run_webui.sh
```

主页：`http://127.0.0.1:7866/`。

主页右上角的“释放显存”按钮会卸载 7866 工作台中已缓存的 CosyVoice3 / HDEMUCS，并通知独立的 18763 ASR 服务卸载 Qwen3-ASR 和 ForcedAligner；服务端口保持在线，下一次使用时模型会自动重新加载。正在生成或分离时按钮会返回忙碌提示，不会中断任务。

语音资产流水线：`http://127.0.0.1:7866/assets`。操作顺序是：上传音频或视频，填写已知说话人数（不知道就保持 0），点击“开始识别并提取资产”，等待识别、清洁、切片和保存完成。结果按说话人显示，资产目录和清洁参考音频会写入 `F:\voice`。

音视频重新配音流水线：`http://127.0.0.1:7866/studio`。页面按从上到下的顺序组织：导入视频，分析声音并分离人声 / 背景，筛选说话人和台词片段，绑定参考音色并试听，最后替换目标说话人的全部台词并导出。分析会自动生成一条人声轨和一条背景 / 音效轨；导出时默认压低目标说话人的原声、保留其他人声和背景 / 音效。

独立 Qwen3-ASR 服务使用本机端口 `18763`，由工作台内部调用；日常操作只需要打开上面的 7866 页面。

当前默认 ASR 是 `Qwen3-ASR-1.7B`，时间戳使用官方 `Qwen3-ForcedAligner-0.6B`。旧的 `Qwen3-ASR-0.6B` 保留为回退版本，临时回退时执行：

```bash
ASR_MODEL_NAME=Qwen3-ASR-0.6B \
ASR_MODEL_DIR=/home/administrator/models/Qwen3-ASR-0.6B \
bash /home/administrator/voice-asset-workbench/scripts/run_webui.sh
```

直接 TTS 工作台地址：`http://127.0.0.1:7866/tts`。它接收音频或视频，先扫描指定范围内的完整音轨，使用轻量 VAD 找出清晰人声，拼接成约 8 秒的参考音频；长音频或疑似多人素材会调用当前说话人后端筛选单一说话人，并在结果中记录候选片段和质量提示。默认的“快速音色复刻”走官方 `inference_cross_lingual`，不需要参考台词；Zero-shot 会自动调用 Qwen3-ASR 提供参考台词，但生成前应确认文字与参考音频完全一致；Instruct2 只需要填写 `instruct_text`。

直接 TTS 页面默认扫描 60 秒、生成约 8 秒参考音频。需要保留原始起点和片段时，可以关闭“自动筛选清晰语音”；这个兼容路径仍会把最多 30 秒的指定片段整理成单声道参考音频。生成结果会同时显示本次真正送入 CosyVoice 的参考音频，便于先判断音色素材是否干净。

重新配音工作台会把结果写入 `F:\voice\<原文件名>__全片配音项目__<任务号>`，其中包含 `时间轴.json`、`说话人时间轴.md`、`音频/人声.wav`、`音频/背景声.wav`、参考音色、每句配音片段、`新配音人声.wav`、`配音混音.wav` 和可选的 `配音视频.mp4`。原始文件不会被覆盖。

无声视频时间线入口：`http://127.0.0.1:7866/timeline`。它接收视频、SRT/VTT/JSON 时间线和一段参考音色，按字幕起止时间逐段调用 CosyVoice3，输出 `时间线配音.wav`、`时间线配音视频.mp4`、`时间线.json` 和逐段音频。视觉字幕项目导出的 `字幕提取时间轴.json` 可以直接导入；其中 `segments[].start_sec`、`end_sec`、`text` 会被保留，视觉模型的估计精度提示也会写入结果。默认保留自然语速；只有主动勾选温和适配且自然时长接近字幕槽（0.80–1.25 倍）时才做变速，严重不匹配的片段会保留原声并顺延后续片段，避免变成快进、慢放或叠音。这个入口当前使用一段参考音色生成整条时间线，后续再增加按 `speaker_id` 绑定多套音色。

当前分离采用 [Demucs 官方仓库](https://github.com/facebookresearch/demucs) 对应的 Hybrid Demucs 权重，将 `vocals` 保留为人声，将 `drums`、`bass` 和 `other` 合成为背景 / 音效。3D-Speaker 使用分离后的人声轨做说话人聚类；ASR 和时间戳使用清洁原轨，以减少动画音效和分离伪影对字音的影响。它适合当前演示素材和大多数单人说话场景；两个真人在同一时刻重叠说话时，仍然会落在同一条人声轨里，不能只从混音中精确抽出其中一个人的原声。

## V3 说话人时间轴

重新配音工作台默认使用 NVIDIA NeMo 的 `nvidia/diar_streaming_sortformer_4spk-v2.1`。运行脚本会把本地 `/home/administrator/models/diar_streaming_sortformer_4spk-v2.1` 挂载为只读模型目录，因此启动后不需要再次联网下载。Sortformer 保留约 80ms 的 speaker posterior；台词字符时间戳会对整个字符时间范围做加权归属。快速接话不再依赖 0.35 秒静音，重叠或低置信度结果会标成 `unknown` / `需人工复核`。

页面的“分析选项”可以把同一素材切换到 `3D-Speaker（旧版 A/B）`。3D-Speaker 的 overlap 运行会把 `hf_access_token` 明确传入；如果官方 overlap 初始化失败，时间轴会写入请求后端、实际后端、`overlap_enabled`、`fallback_reason`，日志也会打印真实 pipeline。Sortformer 完成后会显式释放模型和 CUDA 缓存，便于在 16GB 显存上继续调用 ASR 或 CosyVoice。

V3 镜像默认名为 `ragflow-cosyvoice-speaker:0.0.4`，重建命令仍然是：

```bash
BUILD_IMAGE=1 bash /home/administrator/voice-asset-workbench/scripts/run_webui.sh
```

如果只想快速启动已经构建好的镜像，直接运行：

```bash
bash /home/administrator/voice-asset-workbench/scripts/run_webui.sh
```

## TTS 参数依据

运行时依赖固定为 CosyVoice 官方要求的 `transformers==4.51.3` 和
`tokenizers==0.21.4`。不要把工作台升级到 Transformers 4.57 或更高版本；该版本组合在当前
CosyVoice3 权重上会导致语音生成乱码。

页面只暴露 CosyVoice3 官方 CLI 模板中实际存在的模式和参数：

| 页面参数 | 官方调用 | 说明 |
| --- | --- | --- |
| `cross_lingual` | `inference_cross_lingual(tts_text, prompt_wav, ...)` | 不需要参考台词；CosyVoice3 将前置指令和目标文本放在 `tts_text` 中 |
| `zero_shot` | `inference_zero_shot(tts_text, prompt_text, prompt_wav, ...)` | 官方 3s 极速复刻路径，参考台词必须和参考音频一致 |
| `instruct2` | `inference_instruct2` | 使用参考音频并传入指令 |
| `speed` | `speed` | 直接传给 CosyVoice3，默认 `1.0` |
| `output_gain_db` / `voice_gain_db` | 工作台后处理 | 以活动语音 RMS 做响度标准化，默认 `+6 dB`，峰值过高时自动限峰 |
| `text_frontend` | `text_frontend` | 是否启用文本前端，默认开启（官方函数默认值） |
| `seed` | `set_all_random_seed` | 官方 WebUI 使用的随机种子，默认 `0` |
| `+` / `-` / `±` | 工作台文本预处理 | 送入模型前按上下文读作“加 / 减 / 正负”；字幕原文保持不变 |
| 轻度自然停顿 | CosyVoice3 `[breath]` | 可选地在内部逗号、分号等位置加入少量官方韵律标记 |
| 严格适配目标时长 | ffmpeg 后处理 | 不是 CosyVoice 参数；批量配音页面默认开启 |

批量配音会先按目标时间槽估算新的 `speed`，再在 0.5–2.0 范围内重新生成；若仍然偏长或偏短，使用 ffmpeg 做最后的时长适配，并在质量报告中留下警告。每个要被替换的说话人必须先绑定参考音色；Zero-shot 模式还需要参考音频对应的台词，页面默认自动识别并允许手工修改。页面的“目标说话人原声比例”“其他说话人原声比例”“背景 / 音效比例”和“新配音响度增益”用于控制最终混音，默认值分别为 `0`、`1`、`1` 和 `+6 dB`。

CosyVoice3 zero-shot 的提示词和音频按官方示例直接传入 `inference_zero_shot`，组织为：
`You are a helpful assistant.<|endofprompt|>参考台词`。

CosyVoice2 示例中的 `add_zero_shot_spk` 缓存调用没有套用到 CosyVoice3，因为当前 Fun-CosyVoice3 模型是 24 kHz 前端，官方 CosyVoice3 示例直接传入 `prompt_text` 和 `prompt_wav`。

参考音频样本限制为 30 秒以内，这是 [官方前端](https://github.com/FunAudioLLM/CosyVoice/blob/main/cosyvoice/cli/frontend.py) 提取 speech token 时的硬限制。Zero-shot 目标文本如果明显短于参考台词，官方前端也会给出质量警告；页面会把这个警告显示出来。

官方 CLI 没有向这个接口暴露 `temperature`、`top_p` 等采样参数，因此页面没有凭经验添加这些字段。依据是 [CosyVoice 官方 example.py](https://github.com/FunAudioLLM/CosyVoice/blob/main/example.py) 和 [CosyVoice 官方 CLI 实现](https://github.com/FunAudioLLM/CosyVoice/blob/main/cosyvoice/cli/cosyvoice.py)。

识别部分采用 [3D-Speaker 官方 diarization 脚本](https://github.com/modelscope/3D-Speaker/blob/main/speakerlab/bin/infer_diarization.py) 和 [Qwen3-ASR 官方仓库](https://github.com/QwenLM/Qwen3-ASR) 的调用形态。多人动画或配音素材如果已知说话人数，建议在折叠的“分析选项”中填写人数，例如四人素材填写 `4`，可以减少自动聚类漏分；时间轴同时会合并无停顿的词内边界。

## 接口

- `POST /api/analyze`：只做识别和资产生成。
- `POST /api/tts/generate`：只对已有说话人资产生成试音。
- `POST /api/tts/direct`：上传参考音频或视频，直接生成 TTS。
- `POST /api/dubbing/analyze`：创建全片说话人和词级时间轴分析任务。
- `GET /api/dubbing/jobs/{job_id}`：查询分析或批量配音任务状态。
- `GET /api/dubbing/jobs/{job_id}/timeline`：获取可编辑时间轴 JSON。
- `POST /api/dubbing/jobs/{job_id}/timeline`：暂存台词、说话人和时间修改。
- `POST /api/dubbing/jobs/{job_id}/speakers/{speaker_id}/reference`：绑定一个说话人的参考音频或视频。
- `POST /api/dubbing/jobs/{job_id}/generate`：按目标说话人筛选时间轴，逐句生成配音并导出两路混音 / 视频。
- `GET /api/dubbing/jobs/{job_id}/files/{path}`：访问任务输出文件。
- `POST /api/timeline/jobs`：导入无声视频和 SRT/VTT/JSON 时间线，创建逐段配音任务。
- `GET /api/timeline/jobs/{job_id}`：查询无声视频时间线配音任务。
- `GET /api/timeline/jobs/{job_id}/files/{path}`：访问无声视频配音任务输出文件。
- `GET /api/speaker/{asset_id}/sample`：获取清洁参考样本。
- `GET /api/speaker/{asset_id}/latest`：获取最近一次试音。
- `GET /api/tts/direct/{job_id}/output`：获取直接 TTS 结果。
- `GET /api/health`：查看服务和 TTS 是否已懒加载。
- `GET /api/system/memory`：查看 7866 工作台的模型和 CUDA cache 状态。
- `POST /api/system/release-memory`：释放模型显存，保持 7866 WebUI 和 18763 ASR 服务在线。

每个资产目录包含清洁参考音色、`资产卡片.md`、`说话人分析.json` 或试音结果；原始上传文件和任务状态保留在外部 state 目录，不会被 Git 跟踪。
