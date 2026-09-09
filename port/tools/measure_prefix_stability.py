"""Prefix-stability gate for incremental (real) acoustic streaming.

Incremental streaming means the server emits audio for codes 0..k while codes
k+1.. are still being generated. That is only honest if the audio already sent
would not have changed had the decoder waited for the whole utterance. This
script measures exactly that drift, because two parts of this checkpoint make
it a real question:

1. **Noise.** ``CausalConditionalCFM.rand_noise`` is ``None`` in this
   checkpoint, so the flow draws ``randn`` shaped to the CURRENT length on every
   call. ``randn((80, L))`` fills row-major, so the value at mel position j is
   different for every L -- decoding a prefix twice at two lengths gives two
   different waveforms for the same region, for reasons that have nothing to do
   with the model. A per-request fixed noise bank sliced to the current length
   removes this term entirely. That is what CosyVoice's ``rand_noise`` does and
   what this script installs.

2. **The token encoder is not causal.** ``UpsampleConformerEncoder`` is built
   without ``static_chunk_size``/``use_dynamic_chunk``, so its self-attention is
   fully bidirectional: h[:, :k] genuinely can move when tokens after k arrive.
   Only the ``PreLookaheadLayer`` and the CFM estimator are causal. The
   ``pre_lookahead_len=3`` trim (``finalize=False``) exists to absorb the worst
   of it. Whether what is left is audible is the question this measures.

Reported per case: the drift of the region a streaming server would ALREADY
HAVE SENT, against the same region of the one-shot decode, in mel and in
samples, with and without the fixed noise bank so the two causes are separated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/workspace/repos/vllm-omni")

from vllm_omni.model_executor.models.chatterbox_mtl_v3 import constants as K
from vllm_omni.model_executor.models.chatterbox_mtl_v3.conditioning import ReferenceConditioning
from vllm_omni.model_executor.models.chatterbox_mtl_v3.s3gen import ChatterboxS3Gen
from vllm_omni.model_executor.models.chatterbox_mtl_v3.vendor.s3gen.utils.mask import make_pad_mask

MODEL_DIR = (
    "/workspace/.hf_home/hub/models--ResembleAI--chatterbox/snapshots/"
    "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18"
)
GOLDEN = Path("/workspace/port/artifacts/golden")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_MELS = 80


def load_conditioning(voice: str) -> ReferenceConditioning:
    ref = np.load(GOLDEN / voice / "conditioning.npz")
    t = {k: torch.from_numpy(ref[k]).to(DEVICE) for k in ref.files}
    return ReferenceConditioning(
        cache_key=f"golden-{voice}",
        speaker_emb=t["speaker_emb"],
        cond_prompt_speech_tokens=t["cond_prompt_speech_tokens"].long(),
        prompt_token=t["prompt_token"].long(),
        prompt_token_len=t["prompt_token_len"].long(),
        prompt_feat=t["prompt_feat"],
        embedding=t["embedding"],
    )


@torch.inference_mode()
def decode_prefix(
    s3gen: ChatterboxS3Gen,
    codes: torch.Tensor,
    cond: ReferenceConditioning,
    noise_bank: torch.Tensor | None,
    finalize: bool,
) -> torch.Tensor:
    """Mel for the GENERATED region of ``codes``.

    With ``finalize=False`` the last ``pre_lookahead_len`` codes are dropped, so
    the result covers codes ``0 .. len(codes)-pre_lookahead_len``.
    """
    flow = s3gen.s3gen.flow
    ratio = flow.token_mel_ratio
    dtype = flow.prompt_feat_dtype if hasattr(flow, "prompt_feat_dtype") else cond.prompt_feat.dtype

    prompt_token = cond.prompt_token.reshape(1, -1).to(DEVICE, torch.long)
    declared = int(cond.prompt_token_len.reshape(-1)[0].item())
    declared = max(0, min(declared, int(prompt_token.shape[1])))
    prompt_token = prompt_token[:, :declared]
    prompt_feat = cond.prompt_feat.reshape(1, -1, N_MELS)[:, : declared * ratio].to(DEVICE)

    tokens = torch.cat([prompt_token, codes.reshape(1, -1).to(DEVICE, torch.long)], dim=1)
    token_lens = torch.tensor([tokens.shape[1]], dtype=torch.long, device=DEVICE)

    embedding = torch.nn.functional.normalize(
        cond.embedding.reshape(1, -1).to(DEVICE, dtype), dim=1
    )
    embedding = flow.spk_embed_affine_layer(embedding)

    mask = (~make_pad_mask(token_lens)).unsqueeze(-1).to(embedding)
    h, _ = flow.encoder(flow.input_embedding(tokens) * mask, token_lens)
    if not finalize:
        h = h[:, : -flow.pre_lookahead_len * ratio]
    h = flow.encoder_proj(h)

    mel_len1 = prompt_feat.shape[1]
    mel_total = h.shape[1]
    conds = torch.zeros((1, mel_total, N_MELS), device=DEVICE, dtype=h.dtype)
    conds[:, :mel_len1] = prompt_feat.to(h.dtype)
    conds = conds.transpose(1, 2)
    mel_mask = torch.ones((1, 1, mel_total), device=DEVICE, dtype=h.dtype)

    mu = h.transpose(1, 2).contiguous()
    noise = None
    if noise_bank is not None:
        noise = noise_bank[:, :, :mel_total].to(device=DEVICE, dtype=mu.dtype)

    feat = flow.decoder(
        mu=mu,
        mask=mel_mask,
        spks=embedding,
        cond=conds,
        n_timesteps=s3gen.cfm_timesteps,
        noise=noise,
    )[0]
    return feat[:, :, mel_len1:]


def drift(a: torch.Tensor, b: torch.Tensor) -> dict:
    n = min(a.shape[-1], b.shape[-1])
    x, y = a[..., :n].float(), b[..., :n].float()
    peak = float(y.abs().max().item()) or 1.0
    d = (x - y).abs()
    return {
        "frames": int(n),
        "max_abs": round(float(d.max().item()), 6),
        "mean_abs": round(float(d.mean().item()), 6),
        "pct_of_peak": round(100.0 * float(d.max().item()) / peak, 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="ex01")
    ap.add_argument("--cases", default="en_plain,ar_plain,mixed_ar_en")
    ap.add_argument("--block", type=int, default=25, help="codec tokens per streamed chunk")
    ap.add_argument("--seed", type=int, default=1235)
    ap.add_argument("--out", default="/workspace/port/artifacts/prefix_stability.json")
    args = ap.parse_args()

    s3gen = ChatterboxS3Gen(apply_watermark=False)
    s3gen.load_weights_from_dir(MODEL_DIR)
    s3gen = s3gen.to(DEVICE).eval()
    cond = load_conditioning(args.voice)
    ratio = s3gen.s3gen.flow.token_mel_ratio
    lookahead = s3gen.s3gen.flow.pre_lookahead_len

    report = {
        "voice": args.voice,
        "block_tokens": args.block,
        "pre_lookahead_tokens": lookahead,
        "token_mel_ratio": ratio,
        "cases": {},
    }

    for case in args.cases.split(","):
        codes_path = GOLDEN / args.voice / f"{case}.npz"
        if not codes_path.exists():
            print(f"  {case}: no golden capture, skipped")
            continue
        codes = torch.from_numpy(
            np.load(codes_path)["gen_ids_valid"]
        ).reshape(-1).to(DEVICE, torch.long)
        n = int(codes.shape[0])

        # One fixed noise bank per request, big enough for the whole utterance
        # plus the prompt. Slicing it is what makes a prefix reproducible.
        declared = int(cond.prompt_token_len.reshape(-1)[0].item())
        max_mel = (declared + n) * ratio
        g = torch.Generator(device=DEVICE).manual_seed(args.seed)
        bank = torch.randn((1, N_MELS, max_mel), generator=g, device=DEVICE)

        full = decode_prefix(s3gen, codes, cond, bank, finalize=True)

        per_chunk = []
        k = args.block
        while k < n:
            # What a streaming server would have emitted after k codes.
            partial = decode_prefix(s3gen, codes[:k], cond, bank, finalize=False)
            emitted = partial.shape[-1]              # (k - lookahead) * ratio
            d_fixed = drift(partial, full[:, :, :emitted])

            # Same comparison with the checkpoint's own per-call noise, to show
            # how much of the drift is noise rather than the encoder.
            partial_rnd = decode_prefix(s3gen, codes[:k], cond, None, finalize=False)
            d_rand = drift(partial_rnd, full[:, :, :emitted])

            per_chunk.append({
                "codes_seen": k,
                "codes_emitted": emitted // ratio,
                "fixed_noise": d_fixed,
                "per_call_noise": d_rand,
            })
            print(
                f"  {case:<12} k={k:>4}  emitted={emitted//ratio:>4} codes  "
                f"fixed-noise drift {d_fixed['pct_of_peak']:>7.3f}% of peak   "
                f"per-call-noise drift {d_rand['pct_of_peak']:>8.3f}%"
            )
            k += args.block

        report["cases"][case] = {
            "n_codes": n,
            "mel_frames": int(full.shape[-1]),
            "chunks": per_chunk,
            "worst_fixed_noise_pct": max(
                (c["fixed_noise"]["pct_of_peak"] for c in per_chunk), default=None
            ),
            "worst_per_call_noise_pct": max(
                (c["per_call_noise"]["pct_of_peak"] for c in per_chunk), default=None
            ),
        }

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")
    for case, data in report["cases"].items():
        print(
            f"{case:<14} worst drift: fixed-noise {data['worst_fixed_noise_pct']}%  "
            f"per-call-noise {data['worst_per_call_noise_pct']}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
