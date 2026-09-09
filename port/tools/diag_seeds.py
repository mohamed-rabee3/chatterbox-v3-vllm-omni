"""Is premature EOS a port defect or the model's own sampling variance?

Runs the SAME texts through the port and the official reference across many
seeds and compares the distribution of output durations. A defect would show
as the port truncating where the reference never does; sampling variance shows
as both truncating at similar rates.

Duration is expressed relative to each text's median reference duration, so
"truncated" means "much shorter than this text normally is", not an absolute.
"""
from __future__ import annotations

import base64, io, json, statistics, sys
from pathlib import Path

import numpy as np
import requests
import soundfile as sf
import torch

BASE = "http://127.0.0.1:18091"
VOICE = "/workspace/refvoices/en_ex01.wav"
REF_B64 = "data:audio/wav;base64," + base64.b64encode(Path(VOICE).read_bytes()).decode()
SEEDS = [1234, 7, 42, 99, 2024, 555, 8888, 31337]
TEXTS = {
    "en_short": ("en", "Hello, this is the reference implementation speaking."),
    "en_long": ("en",
        "The quick brown fox jumps over the lazy dog, and then it turns around and does it "
        "again because the sentence needs to be long enough to exercise a real decode loop."),
    "ar_long": ("ar",
        "أهلاً وسهلاً بك في خدمة العملاء، نحن سعداء بتواصلك معنا اليوم، "
        "وسوف نقوم بمراجعة طلبك والرد عليك في أقرب وقت ممكن."),
}
# A run shorter than this fraction of the text's median duration is treated as
# a truncation for counting purposes.
TRUNCATION_FRACTION = 0.5


def serve(model_id, text, language, seed):
    body = {"model": model_id, "input": text, "language": language, "ref_audio": REF_B64,
            "response_format": "wav", "seed": seed}
    r = requests.post(f"{BASE}/v1/audio/speech", json=body, timeout=600)
    if r.status_code != 200:
        return None
    a, sr = sf.read(io.BytesIO(r.content), dtype="float32")
    return len(a) / sr


def main():
    model_id = requests.get(f"{BASE}/v1/models", timeout=10).json()["data"][0]["id"]
    sys.path.insert(0, "/workspace/port/reference")
    from run_reference import load_reference

    ref_model, _ = load_reference("cuda")
    ref_model.prepare_conditionals(VOICE, exaggeration=0.5)

    rows = []
    for key, (language, text) in TEXTS.items():
        port, ref = [], []
        for seed in SEEDS:
            port.append(serve(model_id, text, language, seed))
            torch.manual_seed(seed)
            wav = ref_model.generate(text, language_id=language, audio_prompt_path=VOICE)
            ref.append(wav.shape[-1] / ref_model.sr)
        port_ok = [x for x in port if x is not None]
        median = statistics.median(ref)
        cut = TRUNCATION_FRACTION * median
        row = {
            "text": key,
            "seeds": len(SEEDS),
            "reference_median_s": round(median, 2),
            "reference_durations": [round(x, 2) for x in ref],
            "port_durations": [round(x, 2) if x is not None else None for x in port],
            "reference_truncations": sum(1 for x in ref if x < cut),
            "port_truncations": sum(1 for x in port_ok if x < cut),
            "port_errors": len(SEEDS) - len(port_ok),
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    Path("/workspace/port/artifacts/diag_seeds.json").write_text(json.dumps(rows, indent=2))
    print("\n== summary: truncations out of "
          f"{len(SEEDS)} seeds (shorter than {TRUNCATION_FRACTION:.0%} of the median) ==")
    print(f"{'text':<10}{'ref':>6}{'port':>6}{'ref median s':>14}")
    for r in rows:
        print(f"{r['text']:<10}{r['reference_truncations']:>6}{r['port_truncations']:>6}"
              f"{r['reference_median_s']:>14.2f}")


if __name__ == "__main__":
    main()
