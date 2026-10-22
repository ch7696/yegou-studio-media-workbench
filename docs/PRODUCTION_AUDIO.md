# 已确认的音频制作方案

2026-10-22 用户实测确定：保留第二版 **HDEMUCS_HIGH_MUSDB 去人声方案**，配音升级 **Fish S2-Pro**。生产配置见 `configs/production-profile.json`。

- HDEMUCS 丢弃 vocals，合并 drums+bass+other，保留音乐、环境音和音效。背景增益默认 1.0，最终整体防削波。
- Fish 使用 ComfyUI-FishAudioS2 的 s2-pro-bnb-nf4，BF16 计算、SDPA，不编译；复用原始权重，加载时量化。
- BandIt、多轮 BandIt 和频谱强降噪未被采用，不以降低环境音量冒充去人声。不将原始混合音轨加入背景。
- 每个角色绑定准确参考音色、参考台词；逐句情绪写在 tts_text，字幕仍用 text。

## 配置与命令

复制 `configs/production-runtime.example.json` 为 Git 忽略的 `configs/production.local.json`，填写现有 ComfyUI Python、用户目录、模型目录和 torch 缓存。也可设置 `YEGOU_PRODUCTION_RUNTIME`。

`model_root` 是包含 s2-pro 文件夹的父目录。需要 config.json、两个 safetensors 分片及 codec.pth。`torch_home` 下可复用原环境的 `hub/torchaudio/models/hdemucs_high_musdbhq_only.pt`，避免重新下载。

```powershell
& .\scripts\yegou-production.ps1 status
& .\scripts\yegou-production.ps1 run --config 'C:\media\project\agent_work\01_config\job.json'
```

WSL 入口（Windows Python 通过 interop 调用，参数按数组传递）：

```bash
bash scripts/yegou production status
bash scripts/yegou production run --config 'C:\media\project\agent_work\01_config\job.json'
```

也可在仓库执行 `python -m app.production`。若 UNC 上的 PowerShell 脚本被系统签名策略阻止，可使用 WSL 入口或直接 Python 模块，无需修改系统执行策略。

| 动作 | 行为 |
|---|---|
| status | 文件就绪检查，不加载模型；不等于 CUDA 实测 |
| run | 原 HDEMUCS 分离 → Fish 逐句生成 → 对齐 → 原背景+新配音 → 视频及字幕 |
| tts | 只生成逐句音频，保存 resolved-job.json |
| render | 使用已有音频和 HDEMUCS 背景重混音，不重新生成或分离 |

**新生产入口独立运行。7866 WebUI、7868 HTTP 和原 dub 入口仍使用 CosyVoice**，未假装它们已经支持 Fish。新入口不启动 Docker、不要求 ComfyUI 界面常驻。

## 依赖

已验证节点 <https://github.com/Saganaki22/ComfyUI-FishAudioS2.git> v0.5.3，commit `521f33fe79c081da314dc905ce399c62edb24749`。固定节点版本，按上游安装依赖；不要 pip 安装 fish-speech 覆盖 ComfyUI 核心包。codec/audiotools 使用 --no-deps。

实测 Python 3.12.11、PyTorch 2.10.0+cu130、RTX 5060 Ti 16GB，numpy 2.5.2 / numba 0.68 / llvmlite 0.50。需要 torchaudio、numpy、soundfile、scipy、imageio-ffmpeg 和节点依赖。worker 使用 TensorBoard 自带 no-TensorFlow 模式规避本机残缺 TensorFlow 命名空间，不改共享 TensorFlow。模型适用 Fish Audio Research License，模型、许可、缓存和第三方源码不进入 Git。

HDEMUCS 直接调用原 app.separation 算法，保留 24kHz 单声道、30 秒分块及 0.5 秒重叠；仅注入 SoundFile 读写以兼容新 torchaudio 后端。先卸载分离模型，再加载 Fish。

## 任务与复用

参照 `configs/production-job.example.json`。workspace 必填，输入与参考使用绝对路径，按 `AGENT_WORKSPACE_STANDARD.md` 保存。references 按 speaker_id 提供 audio/text，text 必须与参考音频一致。

segments 必填 start_sec/end_sec/text，按时间排序且不重叠；tts_text 可填写 `[excited][shouting]消音泡泡！`，情绪标签不进入字幕。逐句可指定 seed、temperature、top_p、repetition_penalty、max_new_tokens 和 gain。默认采样 0.7/0.7、重复惩罚 1.2、seed 17、最大 token 256。独立单字、嗝和尖叫仍需实听复核。

任何句可提供 audio_path，复用已验证音频。tts 的 `05_logs/<job_id>/resolved-job.json` 包含生成后路径。run 可提供 background_audio 复用原分离结果，但必须声明 background_method 为 HDEMUCS_HIGH_MUSDB；render 必须提供背景及每句 audio_path。burn_subtitles 默认为 true。

每次任务创建独立 ID，配置、音频、日志和交付分别进入 01_config/02_work/05_logs/04_outputs。模型和用户材料不提交 Git。同模型目录下生产任务用文件锁互斥，但不管理外部 ComfyUI、旧 gateway 或早期实验脚本，避免同时运行 GPU 作业。Fish 仅批次内缓存，任务完成或失败卸载；worker 有超时退出，结束释放本进程显存，不停止外部服务。

保持自然语速，必要时有限加速；超过 1.55 倍要求重配或修改时间轴，不截断。报告记录实际对齐、增益、峰值、耗时、配置哈希及 CUDA/PyTorch 版本；导出后全量解码。程序检查不替代听感或 ASR，不保证分离完全无残留。

历史实测：Fish 四段含首次加载 56.14 秒、峰值分配显存 9.08 GiB；15 句批次 77.19 秒。用户确认 Fish 配音好，第二版 HDEMUCS 背景优于本次 BandIt 试验。耗时仅供本机参考。

仓库新入口已通过 WSL → Windows 完整实测：6 秒视频实际运行原 HDEMUCS、Fish 克隆、配音对齐、混音、硬字幕和全量解码，峰值分配显存 8.98 GiB，退出后显卡总占用回到约 700 MiB。无截断或叠音，输出峰值 0.679。该检查验证执行链路，不是对所有台词的听感背书。
