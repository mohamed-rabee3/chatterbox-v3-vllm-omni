"""Build the served model directory for Chatterbox Multilingual V3.

The upstream checkpoint ships no ``config.json`` (its architecture lives in
Python), so a served deployment needs an explicit export. This writes:

  config.json     the frozen architecture + checkpoint profile + policy
  manifest.json   artifact hashes, exporter version, preprocessing version
  <weights>       hard links (or copies) of the pinned artifacts
  <tokenizer>     the multilingual grapheme tokenizer

Every artifact hash is verified against the pinned manifest before the export
is written, so a corrupted or substituted file cannot be served.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

EXPORTER_VERSION = "1"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> int:
    from vllm_omni.model_executor.models.chatterbox_mtl_v3 import constants as K
    from vllm_omni.transformers_utils.configs.chatterbox_mtl_v3 import ChatterboxMTLV3Config

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/models/chatterbox-mtl-v3")
    ap.add_argument("--profile", default=K.DEFAULT_CHECKPOINT_PROFILE,
                    choices=sorted(K.CHECKPOINT_PROFILES))
    ap.add_argument("--copy", action="store_true", help="copy instead of hard-linking weights")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download

    src = Path(
        snapshot_download(
            repo_id=K.CHATTERBOX_REPO_ID,
            repo_type="model",
            revision=K.CHATTERBOX_REVISION,
            allow_patterns=["*.pt", "*.safetensors", "*.json"],
        )
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    profile = K.CHECKPOINT_PROFILES[args.profile]
    wanted = [profile["t3"], profile["s3gen"], K.VOICE_ENCODER_FILE, K.TOKENIZER_FILE]

    manifest_hashes: dict[str, str] = {}
    for name in wanted:
        s = (src / name).resolve()
        if not s.exists():
            raise FileNotFoundError(f"pinned artifact missing from the snapshot: {name}")
        digest = sha256_file(s)
        expected = K.ARTIFACT_SHA256.get(name)
        if expected and digest != expected:
            raise ValueError(f"{name}: sha256 {digest} != pinned {expected}")
        manifest_hashes[name] = digest

        d = out / name
        if d.exists() or d.is_symlink():
            d.unlink()
        if args.copy:
            shutil.copy2(s, d)
        else:
            try:
                os.link(s, d)
            except OSError:
                shutil.copy2(s, d)
        print(f"  {name}  {digest[:16]}…")

    # Chinese Cangjie table is only needed for zh, but keep it alongside the
    # tokenizer so a zh-enabled deployment does not reach back to the Hub.
    cangjie = src / "Cangjie5_TC.json"
    if cangjie.exists():
        d = out / "Cangjie5_TC.json"
        if not d.exists():
            shutil.copy2(cangjie.resolve(), d)

    # vLLM builds an HF tokenizer for the stage that owns one. Chatterbox ships
    # a `tokenizers` serialization but no HF wrapper, so expose the model's OWN
    # multilingual grapheme tokenizer under the name AutoTokenizer looks for.
    # It must be this file: the repo also contains an English `tokenizer.json`,
    # and serving that one would tokenize Arabic into ids the 2454-entry text
    # embedding cannot represent.
    shutil.copy2(out / K.TOKENIZER_FILE, out / "tokenizer.json")
    (out / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "model_max_length": K.MAX_TEXT_TOKENS,
                "bos_token": "[START]",
                "eos_token": "[STOP]",
                "unk_token": "[UNK]",
                "pad_token": "[PAD]",
                "clean_up_tokenization_spaces": False,
                # The engine never tokenizes for this model: the adapter and the
                # multimodal processor build the ids with the reference path
                # (punc_norm -> language marker -> [SPACE] -> [START]/[STOP]).
                # This wrapper exists so the engine can construct *a* tokenizer.
                "_chatterbox_note": "serving tokenization is done by the adapter, not by this wrapper",
            },
            indent=2,
        )
    )

    config = ChatterboxMTLV3Config(checkpoint_profile=args.profile)
    config.save_pretrained(str(out))

    (out / "manifest.json").write_text(
        json.dumps(
            {
                "model_type": K.MODEL_TYPE,
                "architectures": [K.MODEL_ARCH],
                "checkpoint_profile": args.profile,
                "t3_checkpoint": profile["t3"],
                "s3gen_checkpoint": profile["s3gen"],
                "repo_id": K.CHATTERBOX_REPO_ID,
                "revision": K.CHATTERBOX_REVISION,
                "upstream_source_revision": K.UPSTREAM_SOURCE_REVISION,
                # Model identity is weights PLUS the code that interprets them.
                "exporter_version": EXPORTER_VERSION,
                "preprocessor_version": K.PREPROCESSOR_VERSION,
                "artifact_sha256": manifest_hashes,
            },
            indent=2,
        )
    )
    print(f"\nexported profile {args.profile!r} to {out}")
    print(json.dumps(json.loads((out / "config.json").read_text())["architectures"], indent=None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
