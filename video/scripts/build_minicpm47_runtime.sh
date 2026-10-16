#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_DIR="${MINICPM_RUNTIME_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/minicpm-v47-upgrade}"
WHEEL="vllm-0.32.0rc1.dev277+g642c18d64-cp38-abi3-manylinux_2_28_x86_64.whl"
SHA256="4c079ae97d48afc904fe1a3acffca32e3b56bcfde30c28704b26e6185ef3eb0a"
URL="https://wheels.vllm.ai/642c18d64c4cb6db3ded8fbe1786718cd4005fcb/${WHEEL/+/%2B}"
IMAGE="${MINICPM47_IMAGE:-yegou-vllm:minicpm47-642c18d64}"
mkdir -p "$CACHE_DIR"
if ! (cd "$CACHE_DIR" && printf '%s  %s\n' "$SHA256" "$WHEEL" | sha256sum -c - >/dev/null 2>&1); then
  curl --fail --location --retry 3 --connect-timeout 20 "$URL" -o "$CACHE_DIR/$WHEEL.part"
  mv "$CACHE_DIR/$WHEEL.part" "$CACHE_DIR/$WHEEL"
fi
(cd "$CACHE_DIR" && printf '%s  %s\n' "$SHA256" "$WHEEL" | sha256sum -c -)
TORCH_WHEEL="torch-2.14.0-cp312-cp312-manylinux_2_28_x86_64.whl"
TORCH_SHA256="fecffb58f51fd643d213acd68da21cc3fc19bea05a3bc64b4ee55128f47a4963"
TORCH_URL="https://mirrors.aliyun.com/pypi/packages/15/71/49399acd41f750a906c686bd23c08a2001ccb8dd25f2971003c2ed89c1dd/$TORCH_WHEEL"
if ! (cd "$CACHE_DIR" && printf '%s  %s\n' "$TORCH_SHA256" "$TORCH_WHEEL" | sha256sum -c - >/dev/null 2>&1); then
  curl --fail --location --retry 3 --connect-timeout 20 "$TORCH_URL" -o "$CACHE_DIR/$TORCH_WHEEL.part"
  (cd "$CACHE_DIR" && printf '%s  %s.part\n' "$TORCH_SHA256" "$TORCH_WHEEL" | sha256sum -c -)
  mv "$CACHE_DIR/$TORCH_WHEEL.part" "$CACHE_DIR/$TORCH_WHEEL"
fi
(cd "$CACHE_DIR" && printf '%s  %s\n' "$TORCH_SHA256" "$TORCH_WHEEL" | sha256sum -c -)
docker build \
  --file "$REPO_ROOT/deploy/Dockerfile.minicpm47" \
  --build-arg "PIP_INDEX_URL=${PIP_INDEX_URL:-https://pypi.org/simple}" \
  --tag "$IMAGE" "$CACHE_DIR"
printf 'MiniCPM-V 4.7 runtime: %s\n' "$IMAGE"
