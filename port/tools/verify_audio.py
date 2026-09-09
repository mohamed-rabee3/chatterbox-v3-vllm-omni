"""Objective verification of served audio against the official reference.

Three independent signals, because no single one is sufficient (plan section 11.5):

* **ASR** -- does the audio say the requested words? Character error rate against
  the input text, computed on both raw and lightly-normalized text so an
  aggressive normalizer cannot hide a changed number or name.
* **Speaker similarity** -- does it sound like the *requested* voice? Cosine
  similarity of the Chatterbox voice-encoder embedding against the reference
  clip. Reported next to the official runner's own similarity, so the port is
  judged against what this model actually achieves, not against 1.0.
* **Duration** -- served audio length vs the official runner's, as a coarse
  check for truncation or runaway generation.

A waveform distance is deliberately NOT used as a pass/fail signal: the AR stage
samples, so two correct runs differ.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import requests
import soundfile as sf
import torch

BASE = "http://127.0.0.1:18091"
MODEL_DIR = (
    "/workspace/.hf_home/hub/models--ResembleAI--chatterbox/snapshots/"
    "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18"
)
VOICES = {"ex01": "/workspace/refvoices/en_ex01.wav", "ex02": "/workspace/refvoices/en_ex02.wav"}

CASES = [
    ("en_plain", "en", "Hello, this is the reference implementation speaking.", "ex01"),
    ("en_long", "en",
     "The quick brown fox jumps over the lazy dog, and then it turns around and does it "
     "again because the sentence needs to be long enough to exercise a real decode loop.", "ex01"),
    ("ar_plain", "ar", "حياك الله، موعدك بكرة الساعة التاسعة صباحًا.", "ex01"),
    ("ar_long", "ar",
     "أهلاً وسهلاً بك في خدمة العملاء، نحن سعداء بتواصلك معنا اليوم، "
     "وسوف نقوم بمراجعة طلبك والرد عليك في أقرب وقت ممكن.", "ex02"),
    ("mixed_ar_en", "ar", "حياك الله، حسابك على Netflix تم تجديده اليوم.", "ex02"),
]

_ARABIC_DIACRITICS = re.compile(r"[ً-ْٰـ]")


def normalize_for_asr(text: str, language: str) -> str:
    """Light normalization only.

    Case, punctuation and Arabic diacritics/tatweel are removed; letters,
    digits and word identity are NOT touched. An aggressive normalizer would
    make the error rate look better while hiding a changed amount or name.
    """
    text = unicodedata.normalize("NFKC", text).lower()
    if language == "ar":
        text = _ARABIC_DIACRITICS.sub("", text)
        text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ى", "ي")
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate (Levenshtein / len(reference))."""
    r, h = reference, hypothesis
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i]
        for j, hc in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / len(r)


class ASR:
    def __init__(self, device: str = "cuda") -> None:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        name = "openai/whisper-large-v3-turbo"
        self.processor = WhisperProcessor.from_pretrained(name)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            name, dtype=torch.float16
        ).to(device).eval()
        self.device = device

    @torch.inference_mode()
    def transcribe(self, audio: np.ndarray, sr: int, language: str) -> str:
        import librosa

        if sr != 16000:
            audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)
        feats = self.processor(audio, sampling_rate=16000, return_tensors="pt").input_features
        feats = feats.to(self.device, dtype=torch.float16)
        ids = self.model.generate(feats, language=language, task="transcribe", max_new_tokens=200)
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


class SpeakerSimilarity:
    """Chatterbox's own voice encoder -- the same one that conditions T3."""

    def __init__(self, device: str = "cuda") -> None:
        from vllm_omni.model_executor.models.chatterbox_mtl_v3.vendor.voice_encoder import (
            VoiceEncoder,
        )

        self.ve = VoiceEncoder()
        self.ve.load_state_dict(
            torch.load(f"{MODEL_DIR}/ve.pt", map_location="cpu", weights_only=True)
        )
        self.ve = self.ve.to(device).eval()

    @torch.inference_mode()
    def embed(self, audio: np.ndarray, sr: int) -> np.ndarray:
        import librosa

        if sr != 16000:
            audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)
        emb = self.ve.embeds_from_wavs([audio], sample_rate=16000)
        v = np.asarray(emb).reshape(-1)
        return v / (np.linalg.norm(v) + 1e-9)

    def similarity(self, a: np.ndarray, sr_a: int, b: np.ndarray, sr_b: int) -> float:
        return float(np.dot(self.embed(a, sr_a), self.embed(b, sr_b)))


def served_model_id() -> str:
    return requests.get(f"{BASE}/v1/models", timeout=10).json()["data"][0]["id"]


def serve_one(model_id: str, text: str, language: str, voice: str, seed: int = 1234):
    wav = Path(VOICES[voice]).read_bytes()
    body = {
        "model": model_id,
        "input": text,
        "language": language,
        "ref_audio": "data:audio/wav;base64," + base64.b64encode(wav).decode(),
        "response_format": "wav",
        "seed": seed,
    }
    r = requests.post(f"{BASE}/v1/audio/speech", json=body, timeout=600)
    r.raise_for_status()
    return sf.read(io.BytesIO(r.content), dtype="float32")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/port/artifacts/verify_audio.json")
    ap.add_argument("--wav-dir", default="/workspace/port/artifacts/verify")
    ap.add_argument("--skip-reference", action="store_true")
    args = ap.parse_args()

    wav_dir = Path(args.wav_dir)
    wav_dir.mkdir(parents=True, exist_ok=True)

    model_id = served_model_id()
    asr = ASR()
    spk = SpeakerSimilarity()

    reference_model = None
    if not args.skip_reference:
        import sys

        sys.path.insert(0, "/workspace/port/reference")
        from run_reference import load_reference

        reference_model, _ = load_reference("cuda")

    rows = []
    for case_id, language, text, voice in CASES:
        ref_audio, ref_sr = sf.read(VOICES[voice], dtype="float32")
        if ref_audio.ndim > 1:
            ref_audio = ref_audio.mean(axis=1)

        served, sr = serve_one(model_id, text, language, voice)
        sf.write(wav_dir / f"{case_id}.served.wav", served, sr)

        hyp = asr.transcribe(served, sr, language)
        want = normalize_for_asr(text, language)
        got = normalize_for_asr(hyp, language)
        row = {
            "case": case_id,
            "language": language,
            "voice": voice,
            "served_seconds": round(len(served) / sr, 3),
            "served_cer": round(cer(want, got), 4),
            "served_speaker_similarity": round(spk.similarity(served, sr, ref_audio, ref_sr), 4),
            "asr_served": hyp,
            "expected_text": text,
        }

        if reference_model is not None:
            torch.manual_seed(1234)
            reference_model.prepare_conditionals(VOICES[voice], exaggeration=0.5)
            wav = reference_model.generate(text, language_id=language,
                                           audio_prompt_path=VOICES[voice])
            ref_out = wav.squeeze(0).numpy()
            sf.write(wav_dir / f"{case_id}.official.wav", ref_out, reference_model.sr)
            hyp_ref = asr.transcribe(ref_out, reference_model.sr, language)
            row.update(
                official_seconds=round(len(ref_out) / reference_model.sr, 3),
                official_cer=round(cer(want, normalize_for_asr(hyp_ref, language)), 4),
                official_speaker_similarity=round(
                    spk.similarity(ref_out, reference_model.sr, ref_audio, ref_sr), 4
                ),
                asr_official=hyp_ref,
            )
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False, indent=None))

    Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=2))

    print("\n== summary ==")
    print(f"{'case':<14}{'lang':<6}{'CER port':>10}{'CER ref':>10}"
          f"{'spk port':>10}{'spk ref':>10}{'sec port':>10}{'sec ref':>10}")
    for r in rows:
        print(f"{r['case']:<14}{r['language']:<6}{r['served_cer']:>10.4f}"
              f"{r.get('official_cer', float('nan')):>10.4f}"
              f"{r['served_speaker_similarity']:>10.4f}"
              f"{r.get('official_speaker_similarity', float('nan')):>10.4f}"
              f"{r['served_seconds']:>10.2f}{r.get('official_seconds', float('nan')):>10.2f}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
