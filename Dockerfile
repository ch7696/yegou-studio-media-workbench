FROM ragflow-cosyvoice:0.0.1

ENV PYTHONUNBUFFERED=1

# Keep the existing Qwen3-ASR / CosyVoice dependency layer reproducible from
# the original local base image, then add only the official NeMo ASR runtime
# needed by the four-speaker Streaming Sortformer checkpoint.
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

# NeMo is installed without dependency resolution so it cannot replace the
# pinned CosyVoice transformers/tokenizers pair below.
RUN python3 -m pip install --no-cache-dir \
    Cython \
    packaging \
    'nemo_toolkit[asr]' \
    --no-deps \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    --trusted-host mirrors.aliyun.com

# The Sortformer inference path uses NeMo's audio dataloader and alignment
# helpers. Install only the runtime pieces absent from the base image;
# training-only extras such as wandb are unnecessary.
RUN python3 -m pip install --no-cache-dir \
    lhotse==1.33.0 \
    kaldialign==0.12.0 \
    braceexpand==0.1.7 \
    wrapt \
    smart-open \
    text-unidecode \
    aistore \
    sacrebleu \
    whisper_normalizer \
    tensorboard \
    webdataset \
    nv_one_logger_core \
    nv_one_logger_training_telemetry \
    nv_one_logger_pytorch_lightning_integration \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    --trusted-host mirrors.aliyun.com

# Keep the CosyVoice3-compatible pair after NeMo's dependency layer has
# finished.
RUN python3 -m pip install --no-cache-dir --no-deps \
    transformers==4.51.3 \
    tokenizers==0.21.4 \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    --trusted-host mirrors.aliyun.com
