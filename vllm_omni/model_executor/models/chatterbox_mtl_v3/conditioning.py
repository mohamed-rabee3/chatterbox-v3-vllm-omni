# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Immutable reference-voice conditioning for Chatterbox Multilingual V3.

The official wrapper stores conditioning on the model instance and mutates it
per call (``self.conds``, and ``T3Cond`` caches derived embeddings in place).
That is not an isolation design: two concurrent HTTP requests would race on one
speaker. Here conditioning is an immutable, content-addressed artifact that a
request *borrows*; nothing request-scoped is ever written back onto the model.

Reference behaviour preserved exactly (plan section 3.2):

* audio is loaded mono at 24 kHz and a 16 kHz copy is derived from it;
* S3Gen reference conditioning uses the first **10 s of the 24 kHz** signal;
* the T3 speech prompt uses the first **6 s of the 16 kHz** signal, capped at
  150 codec tokens, tokenized by the tokenizer bundled with the *loaded S3Gen
  checkpoint*;
* the T3 voice encoder sees the **full 16 kHz** signal, not the 10 s crop.

Those three windows are deliberately different. Collapsing them changes the
voice.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from vllm_omni.model_executor.models.chatterbox_mtl_v3 import constants as K


class ConditioningError(ValueError):
    """A reference voice that must be rejected before any GPU work is admitted."""


@dataclass(frozen=True)
class ReferenceConditioning:
    """Everything derived from one reference voice, for both stages.

    Frozen on purpose: a request holds a reference to this object and must not
    be able to change what another request sees. ``exaggeration`` is *not* part
    of it -- exaggeration is a per-request scalar applied at prefill, so one
    cached voice serves every exaggeration value.
    """

    cache_key: str

    # T3 conditioning
    speaker_emb: torch.Tensor          # (1, 256)
    cond_prompt_speech_tokens: torch.Tensor  # (1, <=150) int64

    # S3Gen conditioning
    prompt_token: torch.Tensor         # (1, P) int64
    prompt_token_len: torch.Tensor     # (1,) int64
    prompt_feat: torch.Tensor          # (1, 2P, 80)
    embedding: torch.Tensor            # (1, 80)

    # Provenance
    source_seconds: float = 0.0
    preprocessor_version: str = K.PREPROCESSOR_VERSION
    checkpoint_profile: str = K.DEFAULT_CHECKPOINT_PROFILE

    def to(self, device: torch.device | str) -> "ReferenceConditioning":
        """Device-local view. Returns ``self`` when already on ``device``."""
        if self.speaker_emb.device == torch.device(device):
            return self
        moved = {
            f: (v.to(device) if torch.is_tensor(v) else v)
            for f, v in self.__dict__.items()
        }
        return ReferenceConditioning(**moved)

    def s3gen_ref_dict(self) -> dict[str, Any]:
        """The ``ref_dict`` shape the vendored S3Gen flow expects.

        ``prompt_feat_len`` is ``None`` in the reference (``embed_ref`` sets it
        so); it is passed through rather than invented.
        """
        return {
            "prompt_token": self.prompt_token,
            "prompt_token_len": self.prompt_token_len,
            "prompt_feat": self.prompt_feat,
            "prompt_feat_len": None,
            "embedding": self.embedding,
        }


def conditioning_cache_key(
    *,
    audio_content_hash: str,
    checkpoint_profile: str,
    preprocessor_version: str = K.PREPROCESSOR_VERSION,
    model_revision: str = K.CHATTERBOX_REVISION,
) -> str:
    """Content-addressed key for one reference voice.

    Keyed by decoded audio *content*, never by path: a path-keyed cache happily
    survives the file being replaced, which silently serves the wrong voice.
    The model revision, checkpoint profile and preprocessing revision are folded
    in so a weight or preprocessing change cannot reuse stale conditioning.
    """
    h = hashlib.sha256()
    for part in (audio_content_hash, checkpoint_profile, preprocessor_version, model_revision):
        h.update(b"\x00")
        h.update(part.encode("utf-8"))
    return h.hexdigest()[:32]


def audio_content_hash(wav: np.ndarray, sample_rate: int) -> str:
    """Hash of the decoded waveform itself, plus its rate."""
    arr = np.ascontiguousarray(np.asarray(wav, dtype=np.float32).reshape(-1))
    h = hashlib.sha256()
    h.update(str(int(sample_rate)).encode("ascii"))
    h.update(b"\x00")
    h.update(arr.tobytes())
    return h.hexdigest()


class ConditioningEncoder:
    """Builds :class:`ReferenceConditioning` from raw audio.

    Owns the small models the reference wrapper uses for conditioning: the T3
    voice encoder (``ve.pt``), and -- taken from the *loaded S3Gen checkpoint*
    so the codes match the acoustic model -- the S3 speech tokenizer, the
    CAMPPlus speaker encoder and the 24 kHz mel extractor.
    """

    def __init__(self, model_dir: str, *, device: str | torch.device = "cpu",
                 checkpoint_profile: str = K.DEFAULT_CHECKPOINT_PROFILE) -> None:
        import os

        from vllm_omni.model_executor.models.chatterbox_mtl_v3.vendor.s3gen.utils.mel import (
            mel_spectrogram,
        )
        from vllm_omni.model_executor.models.chatterbox_mtl_v3.vendor.s3gen.xvector import CAMPPlus
        from vllm_omni.model_executor.models.chatterbox_mtl_v3.vendor.s3tokenizer import S3Tokenizer
        from vllm_omni.model_executor.models.chatterbox_mtl_v3.vendor.voice_encoder import (
            VoiceEncoder,
        )

        if checkpoint_profile not in K.CHECKPOINT_PROFILES:
            raise ConditioningError(f"unknown checkpoint profile {checkpoint_profile!r}")
        self.checkpoint_profile = checkpoint_profile
        self.device = torch.device(device)

        s3gen_file = K.CHECKPOINT_PROFILES[checkpoint_profile]["s3gen"]
        s3gen_path = os.path.join(model_dir, s3gen_file)
        # weights_only: these are third-party .pt artifacts; never unpickle code.
        state = torch.load(s3gen_path, map_location="cpu", weights_only=True)

        self.tokenizer = S3Tokenizer()
        tok_state = {k[len("tokenizer.") :]: v for k, v in state.items() if k.startswith("tokenizer.")}
        _load_strict_except(self.tokenizer, tok_state, allowed_missing=("_mel_filters", "window"))
        self.tokenizer.to(self.device).eval()

        self.speaker_encoder = CAMPPlus(memory_efficient=False)
        spk_state = {
            k[len("speaker_encoder.") :] : v for k, v in state.items() if k.startswith("speaker_encoder.")
        }
        _load_strict_except(self.speaker_encoder, spk_state)
        self.speaker_encoder.to(self.device).eval()

        self.mel_extractor = mel_spectrogram

        self.voice_encoder = VoiceEncoder()
        ve_state = torch.load(
            os.path.join(model_dir, K.VOICE_ENCODER_FILE), map_location="cpu", weights_only=True
        )
        _load_strict_except(self.voice_encoder, ve_state)
        self.voice_encoder.to(self.device).eval()

        self._resamplers: dict[tuple[str, str], Any] = {}
        del state

    def _resample_24_to_16(self, wav24: torch.Tensor) -> torch.Tensor:
        """torchaudio 24 kHz -> 16 kHz, matching the reference ``embed_ref``.

        Built once per device and reused: constructing the resampler builds a
        sinc kernel, which is pure overhead on the per-voice path.
        """
        import torchaudio

        key = (str(wav24.device), str(wav24.dtype))
        resampler = self._resamplers.get(key)
        if resampler is None:
            resampler = torchaudio.transforms.Resample(K.S3GEN_SR, K.S3_SR).to(wav24.device)
            self._resamplers[key] = resampler
        return resampler(wav24)

    @torch.inference_mode()
    def encode(
        self,
        wav: np.ndarray,
        sample_rate: int,
        *,
        max_seconds: float = 30.0,
    ) -> ReferenceConditioning:
        """Compute conditioning from a decoded mono waveform.

        ``wav`` may be at any rate; it is resampled exactly as the reference
        wrapper does (librosa to 24 kHz, then librosa 24 kHz -> 16 kHz).
        """
        import librosa

        arr = np.asarray(wav, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            raise ConditioningError("reference audio is empty")
        if not np.isfinite(arr).all():
            raise ConditioningError("reference audio contains non-finite samples")

        if sample_rate != K.S3GEN_SR:
            wav24 = librosa.resample(arr, orig_sr=sample_rate, target_sr=K.S3GEN_SR)
        else:
            wav24 = arr
        duration = float(wav24.shape[0]) / K.S3GEN_SR
        if duration > max_seconds:
            raise ConditioningError(
                f"reference audio is {duration:.1f}s; the limit is {max_seconds:.1f}s"
            )
        if duration < 1.0:
            raise ConditioningError(f"reference audio is {duration:.2f}s; at least 1.0s is required")

        wav16 = librosa.resample(wav24, orig_sr=K.S3GEN_SR, target_sr=K.S3_SR)

        # --- S3Gen reference: first 10 s of the 24 kHz signal ---------------
        dec_len = K.DEC_COND_SECONDS * K.S3GEN_SR
        ref24 = torch.from_numpy(wav24[:dec_len]).float().to(self.device).unsqueeze(0)
        ref_mels_24 = self.mel_extractor(ref24).transpose(1, 2)

        # The reference resamples the S3Gen window with torchaudio (inside
        # ``embed_ref``) while the T3 window above uses librosa. Two different
        # resamplers is reference behaviour; unifying them changes the voice.
        ref16_for_gen = self._resample_24_to_16(ref24)
        ref_x_vector = self.speaker_encoder.inference(ref16_for_gen)
        ref_tokens, ref_token_lens = self.tokenizer(ref16_for_gen)

        # Reference invariant: mel_len == 2 * token_len. The upstream code
        # truncates tokens when a non-40 ms-multiple input breaks it.
        if ref_mels_24.shape[1] != K.MEL_FRAMES_PER_CODEC_TOKEN * ref_tokens.shape[1]:
            keep = ref_mels_24.shape[1] // K.MEL_FRAMES_PER_CODEC_TOKEN
            ref_tokens = ref_tokens[:, :keep]
            ref_token_lens = ref_token_lens.clone()
            ref_token_lens[0] = ref_tokens.shape[1]

        # --- T3 speech prompt: first 6 s of the 16 kHz signal, <=150 codes ---
        enc_len = K.ENC_COND_SECONDS * K.S3_SR
        t3_prompt_wav = torch.from_numpy(wav16[:enc_len]).float().to(self.device).unsqueeze(0)
        t3_prompt_tokens, _ = self.tokenizer.forward(
            [t3_prompt_wav.squeeze(0)], max_len=K.SPEECH_COND_PROMPT_LEN
        )
        t3_prompt_tokens = torch.atleast_2d(t3_prompt_tokens).to(self.device)

        # --- T3 speaker embedding: the FULL 16 kHz signal --------------------
        ve_embed = torch.from_numpy(
            self.voice_encoder.embeds_from_wavs([wav16], sample_rate=K.S3_SR)
        ).mean(axis=0, keepdim=True).to(self.device)

        key = conditioning_cache_key(
            audio_content_hash=audio_content_hash(wav24, K.S3GEN_SR),
            checkpoint_profile=self.checkpoint_profile,
        )
        return ReferenceConditioning(
            cache_key=key,
            speaker_emb=ve_embed.contiguous(),
            cond_prompt_speech_tokens=t3_prompt_tokens.to(torch.long).contiguous(),
            prompt_token=ref_tokens.to(torch.long).contiguous(),
            prompt_token_len=ref_token_lens.to(torch.long).contiguous(),
            prompt_feat=ref_mels_24.contiguous(),
            embedding=ref_x_vector.contiguous(),
            source_seconds=duration,
            checkpoint_profile=self.checkpoint_profile,
        )


def _load_strict_except(module: torch.nn.Module, state: dict, allowed_missing: tuple[str, ...] = ()) -> None:
    """Load a state dict, tolerating only an explicitly listed missing set.

    Buffers such as the tokenizer's ``_mel_filters`` are rebuilt in ``__init__``
    and legitimately absent from some checkpoints. Anything else missing or
    unexpected is a real mismatch and must fail loudly rather than silently
    leaving a randomly initialized submodule in the voice path.
    """
    missing, unexpected = module.load_state_dict(state, strict=False)
    unexplained = [m for m in missing if not any(m.endswith(a) for a in allowed_missing)]
    if unexplained or unexpected:
        raise ConditioningError(
            f"{type(module).__name__}: state-dict mismatch "
            f"(missing={unexplained[:8]}, unexpected={list(unexpected)[:8]})"
        )


@dataclass
class _Entry:
    value: ReferenceConditioning
    leases: int = 0


class ConditioningCache:
    """Bounded LRU of reference conditioning with in-flight leases.

    A request takes a lease for as long as it may still touch the artifact.
    Leased entries are never evicted, so an eviction cannot pull the voice out
    from under a request that is mid-synthesis (which would either crash the
    acoustic stage or, worse, swap a speaker mid-utterance).
    """

    def __init__(self, capacity: int = 256) -> None:
        if capacity < 1:
            raise ValueError("conditioning cache capacity must be >= 1")
        self.capacity = capacity
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, key: str) -> ReferenceConditioning | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            return entry.value

    def put(self, value: ReferenceConditioning) -> ReferenceConditioning:
        """Insert (or return the already-cached equivalent) and evict if needed."""
        with self._lock:
            existing = self._entries.get(value.cache_key)
            if existing is not None:
                self._entries.move_to_end(value.cache_key)
                return existing.value
            self._entries[value.cache_key] = _Entry(value)
            self._evict_locked()
            return value

    def acquire(self, key: str) -> ReferenceConditioning | None:
        """Take a lease. The entry cannot be evicted until :meth:`release`."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            entry.leases += 1
            self._entries.move_to_end(key)
            self.hits += 1
            return entry.value

    def acquire_or_put(self, value: ReferenceConditioning) -> ReferenceConditioning:
        with self._lock:
            entry = self._entries.get(value.cache_key)
            if entry is None:
                entry = _Entry(value)
                self._entries[value.cache_key] = entry
            entry.leases += 1
            self._entries.move_to_end(value.cache_key)
            self._evict_locked()
            return entry.value

    def release(self, key: str) -> None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return
            entry.leases = max(0, entry.leases - 1)
            self._evict_locked()

    def _evict_locked(self) -> None:
        if len(self._entries) <= self.capacity:
            return
        for key in list(self._entries.keys()):
            if len(self._entries) <= self.capacity:
                return
            if self._entries[key].leases == 0:
                self._entries.pop(key)
                self.evictions += 1
        # If every entry is leased the cache is allowed to exceed capacity:
        # dropping a leased voice is a correctness bug, over-shooting a cache
        # bound is only a memory cost, and it self-corrects on release.

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "size": len(self._entries),
                "capacity": self.capacity,
                "leased": sum(1 for e in self._entries.values() if e.leases),
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
            }
