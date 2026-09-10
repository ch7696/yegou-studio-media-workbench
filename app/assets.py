from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import to_windows_path


def save_card(manifest: dict[str, Any]) -> None:
    root = Path(manifest["asset_root"])
    lines = [
        f"# {manifest.get('speaker_label', '说话人')} 参考音色资产",
        "",
        f"- 原始文件：{manifest.get('source_filename', '')}",
        f"- 说话人累计语音：{float(manifest.get('speaker_speech_duration_sec', 0)):.3f} 秒",
        f"- 样本时长：{float(manifest.get('sample_duration_sec', 0)):.3f} 秒",
        f"- 样本文件：{to_windows_path(Path(manifest['sample_audio']))}",
        f"- 来源 SHA256：{manifest.get('source_sha256', '')}",
        "- 处理：24 kHz 单声道、可调降噪、可选首部静音裁剪；保留中间静音和完整音轨",
        "",
        "## 参考台词",
        "",
        manifest.get("reference_text") or "（自动识别失败，请在网页中手工填写）",
        "",
        "## 选取时间段",
        "",
    ]
    lines.extend(
        f"- {float(start):.3f}s - {float(end):.3f}s"
        for start, end in manifest.get("selected_ranges", [])
    )
    lines.extend(["", "## 试音记录", ""])
    for item in manifest.get("generations", []):
        lines.append(
            f"- {item.get('created_at', '')} · {item.get('text', '')} · "
            f"模式 {item.get('mode', 'zero_shot')} · 目标 {item.get('target_duration_sec', '')} 秒 · "
            f"实际 {item.get('final_duration_sec', '')} 秒 · {to_windows_path(Path(item['output']))}"
        )
    root.mkdir(parents=True, exist_ok=True)
    (root / "资产卡片.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def speaker_card(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "asset_id": manifest["asset_id"],
        "speaker_label": manifest["speaker_label"],
        "sample_duration_sec": manifest["sample_duration_sec"],
        "speaker_speech_duration_sec": manifest["speaker_speech_duration_sec"],
        "reference_text": manifest.get("reference_text", ""),
        "sample_url": f"/api/speaker/{manifest['asset_id']}/sample",
        "asset_dir": to_windows_path(Path(manifest["asset_root"])),
    }
