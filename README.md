# Voice Asset Workbench

本项目把“多人说话人识别/参考音色资产生成”和“CosyVoice3 TTS 试音”拆成两条独立链路，运行在当前 WSL Docker 环境中。

识别链路负责上传音频或视频、抽取完整音轨、可调清洁、3D-Speaker 说话人聚类、Qwen3-ASR 参考台词识别，并把每个说话人保存为可读的资产目录。TTS 链路只读取已经生成的说话人资产，不会因为打开页面就重新分析源视频。

## 目录

```text
app/
  audio.py            音轨清洁、样本切片、可选目标时长后处理
  diarization.py      3D-Speaker 说话人识别
  asr.py              Qwen3-ASR 参考台词识别
  identification.py   识别链路和资产清单
  tts.py              CosyVoice3 官方调用封装
  main.py             FastAPI 接口
web/index.html        识别区与 TTS 区分离的页面
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

页面地址：`http://127.0.0.1:7866/`。

## TTS 参数依据

页面只暴露 CosyVoice3 官方 CLI 模板中实际存在的模式和参数：

| 页面参数 | 官方调用 | 说明 |
| --- | --- | --- |
| `zero_shot` | `inference_zero_shot(tts_text, prompt_text, prompt_wav, ...)` | 使用参考音频和参考台词复刻音色 |
| `instruct2` | `inference_instruct2` | 使用参考音频并传入指令 |
| `speed` | `speed` | 直接传给 CosyVoice3，默认 `1.0` |
| `text_frontend` | `text_frontend` | 是否启用文本前端，默认开启（官方函数默认值） |
| 严格适配目标时长 | ffmpeg 后处理 | 不是 CosyVoice 参数，默认关闭以保留自然时长 |

CosyVoice3 zero-shot 的提示词和音频按官方示例直接传入 `inference_zero_shot`，组织为：
`You are a helpful assistant.<|endofprompt|>参考台词`。

CosyVoice2 示例中的 `add_zero_shot_spk` 缓存调用没有套用到 CosyVoice3，因为当前 Fun-CosyVoice3 模型是 24 kHz 前端，官方 CosyVoice3 示例直接传入 `prompt_text` 和 `prompt_wav`。

参考音频样本限制为 30 秒以内，这是 [官方前端](https://github.com/FunAudioLLM/CosyVoice/blob/main/cosyvoice/cli/frontend.py) 提取 speech token 时的硬限制。Zero-shot 目标文本如果明显短于参考台词，官方前端也会给出质量警告；页面会把这个警告显示出来。

官方 CLI 没有向这个接口暴露 `temperature`、`top_p` 等采样参数，因此页面没有凭经验添加这些字段。依据是 [CosyVoice 官方 example.py](https://github.com/FunAudioLLM/CosyVoice/blob/main/example.py) 和 [CosyVoice 官方 CLI 实现](https://github.com/FunAudioLLM/CosyVoice/blob/main/cosyvoice/cli/cosyvoice.py)。

识别部分采用 [3D-Speaker 官方 diarization 脚本](https://github.com/modelscope/3D-Speaker/blob/main/speakerlab/bin/infer_diarization.py) 和 [Qwen3-ASR 官方仓库](https://github.com/QwenLM/Qwen3-ASR) 的调用形态。

## 接口

- `POST /api/analyze`：只做识别和资产生成。
- `POST /api/tts/generate`：只对已有说话人资产生成试音。
- `GET /api/speaker/{asset_id}/sample`：获取清洁参考样本。
- `GET /api/speaker/{asset_id}/latest`：获取最近一次试音。
- `GET /api/health`：查看服务和 TTS 是否已懒加载。

每个资产目录包含清洁参考音色、`资产卡片.md`、`说话人分析.json` 或试音结果；原始上传文件和任务状态保留在外部 state 目录，不会被 Git 跟踪。
