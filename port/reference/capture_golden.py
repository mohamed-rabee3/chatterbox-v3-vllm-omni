"""Capture golden artifacts from the official Chatterbox Multilingual V3 runner.

These artifacts are the ground truth for the port's release gates:

  Gate A  tokenizer + conditioning + prefill embedding layout
  Gate B  T3 logits on fixed (teacher-forced) speech histories
  Gate D  S3Gen mel/waveform for fixed codec sequences and fixed noise

Everything captured here is produced by the *unmodified* official modules; the
only additions are read-only taps. Nothing in this file may "fix" reference
behaviour -- the duplicate BOS, the zeroed unconditional text content and the
final `max(1, N-1)*960` crop are all reference behaviour and are captured as-is.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from corpus import CASES  # noqa: E402  (script-relative import)
from run_reference import load_reference

OUT = Path("/workspace/port/artifacts/golden")


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().cpu().numpy()


def build_text_tokens(model, text: str, language_id: str) -> tuple[str, torch.Tensor]:
    """Reproduce the wrapper's text path exactly: punc_norm -> tokenize -> SOT/EOT."""
    import torch.nn.functional as F
    from chatterbox.mtl_tts import punc_norm

    normed = punc_norm(text)
    ids = model.tokenizer.text_to_tokens(normed, language_id=language_id.lower()).to(model.device)
    sot, eot = model.t3.hp.start_text_token, model.t3.hp.stop_text_token
    ids = F.pad(ids, (1, 0), value=sot)
    ids = F.pad(ids, (0, 1), value=eot)
    return normed, ids


def build_prefill(model, text_ids_1: torch.Tensor) -> dict[str, torch.Tensor]:
    """Reproduce `T3.inference`'s prefill for the CFG pair (cond row, uncond row).

    Returns the pieces the port must match position-for-position.
    """
    t3 = model.t3
    hp = t3.hp
    device = model.device

    text_tokens = torch.cat([text_ids_1, text_ids_1], dim=0)  # CFG pair
    initial_speech = hp.start_speech_token * torch.ones_like(text_tokens[:, :1])

    embeds, len_cond = t3.prepare_input_embeds(
        t3_cond=model.conds.t3,
        text_tokens=text_tokens,
        speech_tokens=initial_speech,
        cfg_weight=0.5,
    )
    bos_token = torch.tensor([[hp.start_speech_token]], dtype=torch.long, device=device)
    bos_embed = t3.speech_emb(bos_token) + t3.speech_pos_emb.get_fixed_embedding(0)
    bos_embed = torch.cat([bos_embed, bos_embed])
    prefill = torch.cat([embeds, bos_embed], dim=1)

    cond_emb = t3.prepare_conditioning(model.conds.t3)
    return {
        "cond_emb": cond_emb,          # (1, 34, 1024)
        "prefill": prefill,            # (2, len, 1024)
        "len_cond": torch.tensor(len_cond),
        "text_tokens": text_tokens,
    }


@torch.inference_mode()
def fixed_history_logits(model, text_ids_1: torch.Tensor, history: list[int]) -> dict[str, np.ndarray]:
    """Raw cond/uncond speech logits for a teacher-forced speech history.

    `history` are generated speech ids (excluding the two prefill BOS). Learned
    speech position of generated token k (0-based) is k+1, matching the
    reference decode loop.
    """
    t3 = model.t3
    parts = build_prefill(model, text_ids_1)
    embeds = parts["prefill"]

    if history:
        ids = torch.tensor(history, dtype=torch.long, device=model.device).unsqueeze(0)
        emb = t3.speech_emb(ids)
        pos = t3.speech_pos_emb.get_fixed_embedding(
            torch.arange(1, len(history) + 1, device=model.device)
        )
        emb = emb + pos
        emb = torch.cat([emb, emb], dim=0)
        embeds = torch.cat([embeds, emb], dim=1)

    out = t3.tfmr(inputs_embeds=embeds, use_cache=False, output_hidden_states=True, return_dict=True)
    hidden = out.hidden_states[-1]
    logits = t3.speech_head(hidden[:, -1, :])  # (2, 8194)
    return {
        "logits_cond": _np(logits[0]),
        "logits_uncond": _np(logits[1]),
        "prefill_len": np.int64(embeds.shape[1]),
    }


@torch.inference_mode()
def capture_case(model, case_id: str, language: str, text: str, voice: str, seed: int) -> dict:
    normed, text_ids = build_text_tokens(model, text, language)
    parts = build_prefill(model, text_ids)

    rec: dict[str, np.ndarray] = {
        "text_ids": _np(text_ids[0]).astype(np.int64),
        "cond_emb": _np(parts["cond_emb"]),
        "prefill_cond": _np(parts["prefill"][0]),
        "prefill_uncond": _np(parts["prefill"][1]),
        "len_cond": np.int64(int(parts["len_cond"])),
    }

    # Deterministic generation with a fixed seed, then fixed-history logits at
    # several depths of the resulting history.
    torch.manual_seed(seed)
    codes = model.t3.inference(
        t3_cond=model.conds.t3,
        text_tokens=torch.cat([text_ids, text_ids], dim=0),
        max_new_tokens=300,
        temperature=0.8,
        cfg_weight=0.5,
        repetition_penalty=1.2,
        min_p=0.05,
        top_p=1.0,
    )[0]
    from chatterbox.models.s3tokenizer import drop_invalid_tokens

    valid = drop_invalid_tokens(codes).to(model.device)
    rec["gen_ids_raw"] = _np(codes).astype(np.int64)
    rec["gen_ids_valid"] = _np(valid).astype(np.int64)

    hist = [int(x) for x in _np(valid).astype(np.int64).tolist()]
    for depth in (0, 1, 5, min(20, len(hist)), min(50, len(hist))):
        fh = fixed_history_logits(model, text_ids, hist[:depth])
        rec[f"fh{depth}_cond"] = fh["logits_cond"]
        rec[f"fh{depth}_uncond"] = fh["logits_uncond"]
        rec[f"fh{depth}_len"] = fh["prefill_len"]

    # Acoustic stage on the fixed codec sequence, with a pinned noise tensor so
    # the port can reproduce the mel bit-for-bit modulo kernel differences.
    n_tok = int(valid.shape[-1])
    torch.manual_seed(seed + 1)
    wav, _ = model.s3gen.inference(speech_tokens=valid, ref_dict=model.conds.gen)
    wav_np = wav.squeeze(0).detach().cpu().numpy()
    st_len = max(1, n_tok - 1)
    cropped = wav_np[: st_len * 960]
    rec["wav_raw"] = wav_np.astype(np.float32)
    rec["wav_cropped"] = cropped.astype(np.float32)
    rec["n_valid_tokens"] = np.int64(n_tok)

    meta = {
        "case_id": case_id,
        "language": language,
        "text": text,
        "normalized_text": normed,
        "voice": voice,
        "seed": seed,
        "text_ids": rec["text_ids"].tolist(),
        "len_cond": int(rec["len_cond"]),
        "prefill_shape": list(rec["prefill_cond"].shape),
        "n_generated_raw": int(rec["gen_ids_raw"].shape[0]),
        "n_valid_tokens": n_tok,
        "wav_raw_samples": int(wav_np.shape[-1]),
        "wav_cropped_samples": int(cropped.shape[-1]),
    }
    return {"arrays": rec, "meta": meta}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="/workspace/refvoices/en_ex01.wav")
    ap.add_argument("--voice-id", default="ex01")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cases", default="")
    args = ap.parse_args()

    model, ckpt = load_reference(args.device)
    model.prepare_conditionals(args.voice, exaggeration=0.5)

    outdir = OUT / args.voice_id
    outdir.mkdir(parents=True, exist_ok=True)

    # Conditioning is voice-level, not case-level: capture it once.
    np.savez_compressed(
        outdir / "conditioning.npz",
        speaker_emb=_np(model.conds.t3.speaker_emb),
        cond_prompt_speech_tokens=_np(model.conds.t3.cond_prompt_speech_tokens).astype(np.int64),
        emotion_adv=_np(model.conds.t3.emotion_adv),
        prompt_token=_np(model.conds.gen["prompt_token"]).astype(np.int64),
        prompt_token_len=_np(model.conds.gen["prompt_token_len"]).astype(np.int64),
        prompt_feat=_np(model.conds.gen["prompt_feat"]),
        embedding=_np(model.conds.gen["embedding"]),
    )

    wanted = set(args.cases.split(",")) if args.cases else None
    metas = []
    for case_id, language, text in CASES:
        if wanted and case_id not in wanted:
            continue
        print(f"[capture] {case_id} ({language}) ...", flush=True)
        got = capture_case(model, case_id, language, text, args.voice_id, args.seed)
        np.savez_compressed(outdir / f"{case_id}.npz", **got["arrays"])
        metas.append(got["meta"])
        print(f"    normalized={got['meta']['normalized_text'][:70]!r}")
        print(f"    text_ids={len(got['meta']['text_ids'])} tokens, "
              f"prefill={got['meta']['prefill_shape']}, "
              f"gen={got['meta']['n_valid_tokens']} codes, "
              f"wav={got['meta']['wav_cropped_samples']} samples")

    (outdir / "manifest.json").write_text(json.dumps({
        "checkpoint_dir": str(ckpt),
        "profile": "official_loader_v3",
        "voice": args.voice,
        "voice_id": args.voice_id,
        "seed": args.seed,
        "torch": torch.__version__,
        "cases": metas,
    }, ensure_ascii=False, indent=2))
    print(f"\nwrote {len(metas)} cases to {outdir}")


if __name__ == "__main__":
    main()
