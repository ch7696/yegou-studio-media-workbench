# Agent 自动配音与资源管理

本机项目：WSL `Ubuntu-26.04`，`/home/administrator/voice-asset-workbench`。

项目文件默认集中到素材目录内的 `agent_work`。使用 `workspace init --root PATH` 初始化，配音提交增加 `--workspace PATH`；详见 [任务文件存放标准](AGENT_WORKSPACE_STANDARD.md)。

轻量控制服务：`http://127.0.0.1:7868`。音频工作台：7866；内部 ASR：18763。
控制服务运行在 WSL 宿主，不加载 torch / 模型。模型容器关闭后，控制服务仍接受任务、查询和成片下载。
视频分析台 7877 不属于这套音频启停操作。

## Windows 使用

推荐用本仓库 `scripts/yegou.ps1`，或桌面部署的 `yegou.ps1`：

```powershell
& "$env:USERPROFILE\Desktop\yegou.ps1" voices add --id narrator --name 旁白 --reference 'F:\voice\旁白参考.wav' --seconds 10
& "$env:USERPROFILE\Desktop\yegou.ps1" dub --input 'F:\视频\原片.mp4' --voice narrator --background remove --wait --stop-after
```

参数以数组传给 WSL，不拼接执行 shell。文件路径支持 Windows 和 WSL；相对路径按 WSL 仓库目录解析，推荐绝对路径。
CLI 总是向 stdout 输出 JSON，进度写 stderr。退出码：0 成功/已排队，1 失败，2 待复核。
Windows 访问 WSL loopback 是否转发由 WSL 配置决定；CLI 在 WSL 内访问服务，不依赖该转发。

## Linux / WSL CLI

```bash
bash scripts/yegou service status
bash scripts/yegou service start
bash scripts/yegou voices list
bash scripts/yegou dub --config /mnt/f/voice/job.json --idempotency-key project-001 --wait
bash scripts/yegou jobs get JOB_ID
bash scripts/yegou jobs wait JOB_ID
bash scripts/yegou jobs download JOB_ID --output /mnt/f/voice/final.mp4
bash scripts/yegou jobs download JOB_ID --file quality --output /mnt/f/voice/quality.json
bash scripts/yegou jobs cancel QUEUED_JOB_ID
bash scripts/yegou jobs retry FAILED_JOB_ID
bash scripts/yegou jobs resume REVIEW_JOB_ID --config /mnt/f/voice/resume.json
bash scripts/yegou service release
bash scripts/yegou service stop
bash scripts/yegou policy --idle-seconds 300
bash scripts/yegou capabilities
```

CLI 默认自动启动轻量控制服务。`--no-autostart` 禁止自动启动。
服务管理命令针对本项目两个容器，不删除容器、镜像、模型或输出，不停止 Docker 引擎与 WSL。
`service release` 仅卸载模型，HTTP 工作台保持在线；`service stop` 关闭两个模型容器。
`--stop-after` 需要 `--wait`，完成后尝试立即停止；若其他任务仍在运行/排队，返回 shutdown 错误而不强制中断。

## HTTP 与参数发现

`GET /api/agent/capabilities` 返回机器可读 JSON Schema 和功能入口。直接用 HTTP 工具调用即可，无需浏览器或额外 agent SDK。

| 接口 | 用途 |
|---|---|
| `GET /api/agent/health` | 轻量服务健康检查 |
| `GET /api/agent/capabilities` | 完整提交参数 Schema |
| `GET /api/agent/voices` | 列出持久化音色 |
| `POST /api/agent/voices` | 注册音色；参考媒体复制保存，不随源文件移动丢失 |
| `GET /api/agent/voices/{id}/reference` | 获取原始参考媒体，供试听 |
| `POST /api/agent/jobs` | 提交完整配音任务 |
| `GET /api/agent/jobs` | 列出任务 |
| `GET /api/agent/jobs/{id}` | 状态、进度、子任务、音色映射、结果与质量报告 |
| `POST /api/agent/jobs/{id}/resume` | 用修订映射/台词恢复待复核任务 |
| `POST /api/agent/jobs/{id}/retry` | 重试失败/中断任务 |
| `POST /api/agent/jobs/{id}/cancel` | 取消排队任务 |
| `GET /api/agent/jobs/{id}/files/{key}` | `video`、`voice_track`、`mix_track`、`quality`；模型关闭后仍可下载 |
| `GET /api/agent/service` | 两个模型容器状态、队列与空闲时间 |
| `POST /api/agent/service/start` | 启动模型服务并等待健康检查 |
| `POST /api/agent/service/stop` | 确认空闲、停止接收新任务，再关闭模型容器 |
| `POST /api/agent/service/release` | 释放工作台模型显存 |
| `GET/POST /api/agent/policy` | 查询/持久化 `idle_seconds`；0 禁用自动关闭 |

输入文件和参考文件需要已存在于此机器；这里的 HTTP 提交使用本地路径，不是远程 URL 下载或 multipart 上传接口。
已有 7866 multipart 接口仍保留。7868 会流式向 7866 上传视频，避免把大视频整体读入内存。

注册音色：

```json
{"id":"narrator","name":"旁白","reference":"F:\\voice\\旁白参考.wav","start":0,"seconds":10,"text":"参考音频的实际台词，可不填"}
```

提交任务：

```json
{
  "input": "F:\\视频\\原片.mp4",
  "voice": "narrator",
  "voices": {"speaker_02": "character_b"},
  "background": "remove",
  "output": "F:\\voice\\项目输出",
  "speaker_num": 2,
  "language": "Chinese",
  "mode": "cross_lingual",
  "speed": 1.0,
  "noise_reduction": 8,
  "voice_gain_db": 6,
  "fit_to_slot": true,
  "adaptive_speed": true,
  "seed": 0,
  "max_retries": 1,
  "timeout_seconds": 21600
}
```

`voice` 是所有说话人的默认音色，`voices` 对指定说话人覆盖；也可只传 `voices`。
说话人 ID 来自识别结果，不应凭姓名猜测。仅指定多人映射时，如未覆盖全部说话人，任务进入 `needs_review` 并附完整时间轴；agent 查看后 resume。
只指定全片默认音色、没有逐人覆盖时，unknown 片段会明确归入第一个已识别说话人，全部使用默认音色；数量写入质量报告。这不代表确定了原说话人身份。
多人 unknown 不会自动猜身份，需 resume 的 `unknown_speaker_id` 或逐片段 `edits` 指定。
如希望 agent 先审核识别文本再生成，在提交时设置 `review_after_analysis=true`。分析结束进入 needs_review，agent 读取时间轴、修正文本/音色后调用 resume；也可以不改参数直接 resume。

```json
{
  "voices": {"speaker_01":"narrator","speaker_02":"character_b"},
  "unknown_speaker_id": "speaker_01",
  "edits": [{"segment_id":"seg_0003","dubbing_text":"校正后的台词","speaker_id":"speaker_02"}]
}
```

`background=remove` 表示最终仅使用新人声，所有原始人声、背景/音乐均为 0；`keep` 保留分离背景但替换全部已识别台词。
原始文件不会覆盖。CosyVoice 音色由参考音频指定，不保证复刻质量或原片口型一致。
Zero-shot 参考文字必须与音频一致；未提供时沿用已有参考绑定接口自动识别。

提交时建议带 `Idempotency-Key`。同键同参数返回原任务，不重复处理；同键不同参数返回 409。
原任务参数的指纹不会因 resume 改动而改变，因此重复投递原始提交仍能找到同一任务。

## 从影视参考提取多角色音色并按剧本配音

控制服务启动后，`engine` 子命令可调用现有 7866 媒体接口，支持流式上传和保存 JSON 响应，便于完整 CLI 编排：

```bash
bash scripts/yegou service start
bash scripts/yegou engine --method POST --route /api/dubbing/analyze --file /path/source.mp4 --config /path/analysis.json --save /path/submit.json
bash scripts/yegou engine --route /api/dubbing/jobs/JOB_ID
bash scripts/yegou engine --route /api/dubbing/jobs/JOB_ID/timeline --save /path/timeline.json
bash scripts/yegou media extract-reference --input /path/isolated-vocals.wav --start 33 --seconds 3.5 --output /path/role-reference.wav
bash scripts/yegou voices add --id character_a --reference /path/role-reference.wav --text '参考音频实际台词'
```

`engine` 直接访问媒体引擎，调用者应先 `service start`；它不进入 7868 任务队列，需按顺序编排。不要与 gateway 的自动配音任务并发提交。
人工/agent 校正的角色不能直接沿用自动聚类标签猜测。原片声线与目标视频原声不同，应分别核对画面和角色称呼。

`POST /api/dubbing/jobs/{id}/script` 可替换错误分割的 ASR 时间轴，并新增角色，供有独立剧本的视频使用。原分析保留在 `original_analysis`。
请求包含 `speakers: [{speaker_id, speaker_label}]` 与 `segments: [{speaker_id,start_sec,end_sec,text}]`；片段按时间排序、不重叠、不超出视频时长。
该接口仅接受已完成分析的任务；导入后为每个角色调用 `/speakers/{speaker_id}/reference` 绑定参考，再调用 `/generate`。

```bash
bash scripts/yegou engine --method POST --route /api/dubbing/jobs/JOB_ID/script --config /path/curated-script.json
bash scripts/yegou engine --method POST --route /api/dubbing/jobs/JOB_ID/speakers/character_a/reference --file /path/role-reference.wav --file-field reference_media --config /path/reference-fields.json
bash scripts/yegou engine --method POST --route /api/dubbing/jobs/JOB_ID/generate --config /path/generate.json
bash scripts/yegou media caption --input /path/dubbed.mp4 --timeline /path/curated-script.json --output /path/captioned.mp4 --srt /path/subtitles.srt
bash scripts/yegou media mux --input /path/original.mp4 --audio /path/background.wav --output /path/no-vocals.mp4
```

`media` 子命令自动唤起模型服务，执行期间持有并续期租约，避免转码/字幕渲染被空闲关闭。它使用已部署容器内的 FFmpeg，宿主无需额外安装。字幕来自已确认的剧本时间轴，不对合成语音重新识别后把识别错误写入字幕。
最终可另外通过 `/api/subtitles/jobs` 识别合成音轨复核漏字、错字；发现问题应针对重生成，不把 ASR 结果等同于听感保证。

逐字时间戳可直接通过 `media transcribe --input AUDIO --output JSON` 获取。局部修复后可用 `media render --config PLAN --output DIR` 重组已有逐句音频：PLAN 包含 `source`、`timeline`、`segments`、`background_volume`；segments 采用配音结果的片段格式（包含音频路径）。可选 `outputs` 映射自定义 voice_track、mix_track、video 文件名，仅允许文件名，不允许路径穿越。默认输出 `新配音人声.wav`、`配音混音.wav`、`配音视频.mp4`。

## 情绪配音与 agent 自主交付

有喊叫、兴奋、怒斥等表演要求时，应先比较参考音色和表达模式，再合成整片。`cross_lingual` 可作为普通对白候选；`instruct2` 配合具体表演指令可用于情绪台词。模型效果需逐句检查，不能仅凭模式名保证音色或表现力。

批量生成接口的 `prompt_instruction` 作用于整个生成请求。不同句子需要不同表演时，用 `engine` 调用 `/api/tts/direct` 分别生成，再通过 `media render` 组合选中的音频，不要把一条情绪指令套到所有角色和台词。

例如将以下内容保存为项目 `01_config/shout.json`：

```json
{
  "target_text": "不是叫你们起飞！",
  "mode": "instruct2",
  "prompt_instruction": "You are a helpful assistant. 请急躁地怒吼这句话，像发现闯祸后着急制止，保持参考人物的音色。",
  "reference_seconds": 10,
  "reference_target_seconds": 10,
  "auto_select_reference": false,
  "detect_reference_speakers": false,
  "trim_leading_silence": false,
  "noise_reduction": 0,
  "speed": 1,
  "output_gain_db": 3,
  "fit_duration": false,
  "seed": 17,
  "output_dir": "/path/project/agent_work/02_work/expressive-take"
}
```

```bash
bash scripts/yegou service start
bash scripts/yegou engine --method POST --route /api/tts/direct --file /path/project/agent_work/03_voices/role.wav --file-field reference_media --config /path/project/agent_work/01_config/shout.json --save /path/project/agent_work/05_logs/shout-result.json
```

参考应为已截取的同角色干净音频；`reference_seconds` 超过实际长度不会补出新的参考素材。需要比较时用不同 seed、指令或参考生成候选，并记录实际选择的音频与参数。一次只改变一个因素，便于判断问题来自角色身份、克隆相似度还是表演方式。

尖叫、笑声等非台词应单独处理。可尝试简短的大声指令、`<strong>啊——！</strong>` 等已有控制标记；重复很多“啊”可能生成额外音节。不要靠提高增益代替喊叫，也不要把过短的声音强行拉长成持续尖叫。保留自然起落，仅在必要时有限调整语速，并检查是否与下一句重叠。

交付前由 agent 自行筛选候选、复核台词和角色参考、检查重叠与削波、同步字幕，再输出带版本的成片。用户要求直接交付时，不把逐句试听或确认设成必经步骤。原片参考、候选、最终选择及局部修复参数保存在项目目录；不要将用户的素材、音色或测试成片提交到 Git。

ASR 用于辅助检查漏字、额外台词和读音，不代表音色或表演验收。短字、非语言发声及同音字可能误识别，应结合单句与上下文复核并保留结果；无法判断的质量问题应明确记录，不能标成已完成听感审核。

## 批量任务命令

配置文件为任务 JSON 数组，提交后由单 worker 串行处理 GPU 流程：

```bash
bash scripts/yegou batch --config /mnt/f/voice/batch.json --idempotency-prefix episode-pack-001 --wait --stop-after
```

数组索引会附加到幂等前缀。同一批次重投请保持顺序；内容改变请换前缀。
个别任务提交失败会记录在结果中，其他任务仍会提交。

## 状态、恢复和资源策略

`queued → running → done`，异常可进入 `failed`、`needs_review`、`interrupted`、`cancelled`。
阶段包含 starting、waiting_for_workbench、analysis、references、generation、complete。
按句数量、输出存在性、WAV 时长、采样峰值/静音/削波、视频音画流及源片/成片时长检查，不满足条件不会报告成功。视频导出给短音轨末尾补静音，保留完整片尾画面。既有配音时长警告也保留在质量报告中。
这些检查不能证明 ASR 台词或音色听感正确；需要语义/听感审核的业务，应由 agent 继续检查报告与试听。

控制服务重启：保留排队任务，把原运行任务标记为 interrupted，避免不确定状态下重复提交 GPU 作业。
工作台重启：旧后台线程对应的遗留任务标成 failed。分析完成后重试会复用时间轴；分析失败会重新分析。生成重试会重新生成全部选中台词，不是逐句断点续传。
仅对已确认失败的子任务做有上限自动重试。网络错误和超时不会盲目重提；子任务可能仍在运行，请查询 `child_id` 后决定是否 retry。
取消仅作用于排队任务；运行任务不强杀。长任务超时后也不会立刻关闭尚在运行的模型任务。

默认空闲 300 秒自动关闭。只有被控制服务显式启动/接管的模型服务才自动关闭；不会因为控制服务刚启动就关闭手工启动的 WebUI。
租约在每次模型接口调用时续期，600 秒无请求过期；实际模型后台线程仍阻止关闭。
关闭前检查本项目 WebUI 请求、后台线程、模型锁与租约，再 drain 拒绝新写入。另有 WebUI 任务时延后关闭。

## 部署与运维

```bash
bash scripts/install_agent_gateway.sh
systemctl --user start yegou-agent.service
systemctl --user status yegou-agent.service
journalctl --user -u yegou-agent.service -n 100
```

用户服务在 WSL 用户会话启动时运行，异常自动重启。若曾用 nohup 启动，请先停止那个轻量 gateway 进程再切换到 systemd，状态目录上的文件锁会拒绝双实例。
Docker 引擎需可用，建议启用本机 Docker socket 按需激活。模型容器创建仍沿用已有脚本、镜像和挂载。
已存在的容器用 `docker start` 恢复，避免每次唤起都重建或丢失任务。

可选环境文件：`~/.config/yegou-agent.env`（systemd），或手动启动时设置环境变量：

| 变量 | 默认 |
|---|---|
| `YEGOU_AGENT_STATE` | `/home/administrator/ragflow-managed/state/agent-gateway` |
| `YEGOU_BACKEND_URL` | `http://127.0.0.1:7866` |
| `YEGOU_ASR_HEALTH_URL` | `http://127.0.0.1:18763/health` |
| `YEGOU_AGENT_URL` | CLI 控制服务地址，默认 `http://127.0.0.1:7868` |
| `YEGOU_AGENT_TOKEN` | 默认空；配置后 HTTP 和 CLI 需相同 Bearer token |

默认只监听 loopback，不支持浏览器跨域写操作。对外监听必须配置 token；此接口可读取提交者指定的本地文件，应仅给可信 agent 使用。
控制状态、音色媒体和任务记录不进入 Git；输出仍在配置的 voice 目录。

验证：

```bash
python3 -m unittest discover -s tests -p test_agent_gateway.py -v
docker exec -e YEGOU_GATE_TEST=1 -e VOICE_STATE_DIR=/tmp/yegou-gate-test -e VOICE_DEFAULT_OUTPUT=/tmp/yegou-gate-output ragflow-model-speaker-webui python3 -m unittest discover -s tests -v
```
## 已确认的新配音入口

需要 Fish S2-Pro 时使用 `bash scripts/yegou production status/run/tts/render`。默认去人声继续使用 HDEMUCS_HIGH_MUSDB，保留音乐和环境音效；BandIt 不作为生产方案。配置与接口边界见 [PRODUCTION_AUDIO.md](PRODUCTION_AUDIO.md)。原 `dub`/HTTP 仍使用 CosyVoice，不要混淆。
