"""Local ASR diagnostic for rendered TTS samples; run outside load tests."""
import argparse
import json
from pathlib import Path
import time
import librosa
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('audio', nargs='+', type=Path)
    p.add_argument('--model', default='/workspace/models/whisper-large-v3-turbo')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    processor = WhisperProcessor.from_pretrained(args.model, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model, local_files_only=True, dtype=torch.float16,
        attn_implementation='sdpa').cuda().eval()
    records = []
    for path in args.audio:
        audio, sr = sf.read(path, dtype='float32')
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        language = 'ar' if 'ar_' in path.name or 'mixed_ar' in path.name else 'en'
        inputs = processor(audio, sampling_rate=16000, return_tensors='pt', return_attention_mask=True)
        started = time.perf_counter()
        with torch.inference_mode():
            tokens = model.generate(
                input_features=inputs.input_features.cuda().half(),
                attention_mask=inputs.attention_mask.cuda(),
                language=language, task='transcribe', max_new_tokens=256,
                do_sample=False)
        row = dict(path=str(path), language=language,
                   transcript=processor.batch_decode(tokens, skip_special_tokens=True)[0],
                   audio_s=len(audio)/16000, wall_s=time.perf_counter()-started)
        records.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(records, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
