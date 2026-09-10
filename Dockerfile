FROM ragflow-cosyvoice:0.0.1

ENV PYTHONUNBUFFERED=1

# 这些依赖来自 Qwen3-ASR 与 3D-Speaker 的实际导入链路。
# 使用国内 PyPI 源，避免运行时才发现缺包。
# Keep this layer compatible with the existing base image cache.  The
# CosyVoice runtime is pinned immediately below to the official versions.
RUN python3 -m pip install --no-cache-dir \
    addict==2.4.0 \
    simplejson==3.19.3 \
    sortedcontainers==2.4.0 \
    kaldiio==2.18.1 \
    datasets==4.0.0 \
    oss2==2.19.1 \
    funasr==1.4.15 \
    transformers==4.57.6 \
    tokenizers==0.22.1 \
    soynlp==0.0.493 \
    qwen-omni-utils==0.0.9 \
    sox==1.5.0 \
    pytz==2025.2 \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    --trusted-host mirrors.aliyun.com

# CosyVoice's official requirements pin these versions.  Newer Transformers
# releases change the Qwen2 generation path and produce garbled speech with
# this CosyVoice3 checkpoint.  --no-deps avoids re-downloading the ASR stack.
RUN python3 -m pip install --no-cache-dir --no-deps \
    transformers==4.51.3 \
    tokenizers==0.21.4 \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    --trusted-host mirrors.aliyun.com
