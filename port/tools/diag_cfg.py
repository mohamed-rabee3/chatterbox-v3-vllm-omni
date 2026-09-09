"""Isolate a truncation by sweeping guidance weight and text length.

Compares the served duration against the official reference's duration for the
same text, seed and voice, so the question "is the guided decode faithful?" is
answered with numbers rather than by inspection.
"""
from __future__ import annotations

import base64, io, json, sys
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

BASE = "http://127.0.0.1:18091"
VOICE = "/workspace/refvoices/en_ex01.wav"
REF_B64 = "data:audio/wav;base64," + base64.b64encode(Path(VOICE).read_bytes()).decode()

TEXTS = {
    "t15": "Hello, this is the reference implementation speaking.",
    "t40": "The quick brown fox jumps over the lazy dog, and then it turns around.",
    "t70": "The quick brown fox jumps over the lazy dog, and then it turns around and does it "
           "again because the sentence needs to be longer.",
    "t102": "The quick brown fox jumps over the lazy dog, and then it turns around and does it "
            "again because the sentence needs to be long enough to exercise a real decode loop.",
}


def serve(model_id, text, cfg_weight, seed=1234):
    body = {"model": model_id, "input": text, "language": "en", "ref_audio": REF_B64,
            "response_format": "wav", "seed": seed,
            "extra_params": {"cfg_weight": cfg_weight}}
    r = requests.post(f"{BASE}/v1/audio/speech", json=body, timeout=600)
    if r.status_code != 200:
        return None, r.text[:160]
    a, sr = sf.read(io.BytesIO(r.content), dtype="float32")
    return len(a) / sr, ""


def main():
    model_id = requests.get(f"{BASE}/v1/models", timeout=10).json()["data"][0]["id"]

    sys.path.insert(0, "/workspace/port/reference")
    import torch
    from run_reference import load_reference

    ref_model, _ = load_reference("cuda")
    ref_model.prepare_conditionals(VOICE, exaggeration=0.5)

    print(f"{'text':<6}{'ntok':>6}{'reference':>11}" + "".join(f"{f'w={w}':>10}" for w in (0.0, 0.25, 0.5, 1.0)))
    rows = []
    for key, text in TEXTS.items():
        torch.manual_seed(1234)
        wav = ref_model.generate(text, language_id="en", audio_prompt_path=VOICE)
        ref_sec = wav.shape[-1] / ref_model.sr
        from chatterbox.mtl_tts import punc_norm
        ntok = len(ref_model.tokenizer.text_to_tokens(punc_norm(text), language_id="en")[0]) + 2

        line = f"{key:<6}{ntok:>6}{ref_sec:>11.2f}"
        row = {"text": key, "n_text_tokens": int(ntok), "reference_seconds": round(ref_sec, 2)}
        for w in (0.0, 0.25, 0.5, 1.0):
            sec, err = serve(model_id, text, w)
            line += f"{(sec if sec is not None else float('nan')):>10.2f}"
            row[f"cfg_{w}"] = round(sec, 2) if sec is not None else err
        print(line)
        rows.append(row)

    Path("/workspace/port/artifacts/diag_cfg.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
