# MiniCPM-V4.7-1B 显存预算调整

2026-10-16：显存预算由0.8降至0.4，保留16K上下文、单并发、image5/video0及编译/CUDA Graph。

降预算前空闲显存占用13510MiB；降预算后完成文本与五图测试，占用7106MiB（约6.94GiB），剩余9205MiB（约8.99GiB）。观测值包含vLLM缓存与其他GPU占用，会随实际负载波动。

12次固定256token请求全部成功（每种输入预热1次、正式2次），解码约230–235tokens/秒，与调整前相近。另一次12007输入token长上下文请求成功。保留16K配置，但未测试恰好16K边界或与ComfyUI同时渲染。

原始记录：/mnt/g/tmp/minicpm47-graph-20261016/lowmem40.jsonl；本目录summary.json、validation.json。
