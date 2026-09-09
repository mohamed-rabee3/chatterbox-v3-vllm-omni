"""Official Chatterbox Multilingual V3 reference runner (pinned revision).

Profile `official_loader_v3`: T3 = t3_mtl23ls_v3.safetensors, S3Gen = s3gen.pt,
exactly as the pinned official loader selects them (plan section 2.2).

This runner exists to produce golden artifacts for the vLLM-Omni port. It must
not be "improved": every behaviour here is the reference the port is compared to.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

REPO_ID = "ResembleAI/chatterbox"
REVISION = "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18"

EXPECTED_SHA256 = {
    "t3_mtl23ls_v3.safetensors": "5abca8321ede76f8e61f1cc0d19aea6c946b28871017ce8726f8a69203f05953",
    "s3gen.pt": "9b9ff07e60b20c136e2b1b3d7563a24604e8d2c4c267888d1ee929dd0151d2a3",
    "s3gen_v3.pt": "f7abce4b196dae2d08d9296cbebc6521b046079577643b42a19a03499d08721e",
    "s3gen_v3.safetensors": "4a46190f3dccc2230fbb3488a930bccc925862ee68f2662433dfcfe93ce6c2cb",
    "ve.pt": "4b16d836bc598509860f6fa068165a8bb5e9ac84f05582dfcf278a5a372879f1",
    "grapheme_mtl_merged_expanded_v1.json": (
        "69632f47220a788a52ce2661d096453c5655e9bf25289d89a8d832c46ee07dbf"
    ),
}


def snapshot_dir() -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="model",
            revision=REVISION,
            allow_patterns=[
                "ve.pt",
                "t3_mtl23ls_v3.safetensors",
                "s3gen.pt",
                "s3gen_v3.pt",
                "s3gen_v3.safetensors",
                "grapheme_mtl_merged_expanded_v1.json",
                "conds.pt",
                "Cangjie5_TC.json",
                "*.json",
            ],
        )
    )


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def verify_manifest(ckpt_dir: Path) -> dict[str, str]:
    """Fail loudly rather than silently synthesising audio from wrong weights."""
    got: dict[str, str] = {}
    for name, expected in EXPECTED_SHA256.items():
        path = ckpt_dir / name
        if not path.exists():
            raise FileNotFoundError(f"pinned artifact missing: {path}")
        digest = sha256_file(path.resolve())
        got[name] = digest
        if digest != expected:
            raise ValueError(f"{name}: sha256 {digest} != pinned {expected}")
    return got


def load_reference(device: str = "cuda"):
    """Load the official multilingual wrapper on the `official_loader_v3` profile."""
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    ckpt_dir = snapshot_dir()
    verify_manifest(ckpt_dir)
    model = ChatterboxMultilingualTTS.from_local(ckpt_dir, device, t3_model="v3")
    return model, ckpt_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="Hello, this is the reference implementation speaking.")
    ap.add_argument("--language", default="en")
    ap.add_argument("--voice", default="/workspace/refvoices/en_ex01.wav")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="/workspace/port/artifacts/ref_out.wav")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    model, ckpt_dir = load_reference(args.device)
    print(f"loaded reference from {ckpt_dir}")
    print("t3 param dtype:", next(model.t3.parameters()).dtype)
    print("s3gen param dtype:", next(model.s3gen.parameters()).dtype)

    wav = model.generate(args.text, language_id=args.language, audio_prompt_path=args.voice)
    arr = wav.squeeze(0).numpy()
    os.makedirs(Path(args.out).parent, exist_ok=True)
    import soundfile as sf

    sf.write(args.out, arr, model.sr)
    print(json.dumps({"out": args.out, "sr": model.sr, "samples": int(arr.shape[-1]),
                      "seconds": round(float(arr.shape[-1]) / model.sr, 3)}, indent=2))


if __name__ == "__main__":
    main()
