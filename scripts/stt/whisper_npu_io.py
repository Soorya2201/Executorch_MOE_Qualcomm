"""Prepare inputs for qnn_whisper_runner.exe and score its transcripts.

prep:  audio -> Whisper log-mel features, float32 [1, 80, 3000] raw files + input_list.txt
       (LF line endings: the runner doesn't strip '\\r'), plus tokenizer.json for the runner.
       Sources: --librispeech N (with reference texts) and/or --wav file.wav ... (any rate, mono-mixed).
score: read outputs/output_<i>.txt, print each transcript, and compute WER where references exist.
"""
import argparse
import json
import os
import re

import numpy as np


def prep(args):
    from transformers import AutoProcessor, AutoTokenizer

    processor = AutoProcessor.from_pretrained(args.hf_model)
    AutoTokenizer.from_pretrained(args.hf_model).save_pretrained(args.run_dir)

    items = []  # (name, audio_16k_float32, reference_or_None)
    if args.librispeech:
        from datasets import load_dataset

        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        # Last N clips: the export calibrates on the first 20.
        for row in list(ds)[-args.librispeech:]:
            a = row["audio"]
            items.append((f"librispeech_{row['id']}", a["array"], a["sampling_rate"], row["text"]))
    for path in args.wav or []:
        import librosa

        audio, sr = librosa.load(path, sr=None, mono=True)
        items.append((os.path.basename(path), audio, sr, None))

    inputs_dir = os.path.join(args.run_dir, "inputs")
    os.makedirs(inputs_dir, exist_ok=True)
    os.makedirs(os.path.join(args.run_dir, "outputs"), exist_ok=True)
    lines, manifest = [], []
    for i, (name, audio, sr, ref) in enumerate(items):
        if sr != 16000:
            import librosa

            audio = librosa.resample(np.asarray(audio, dtype=np.float32), orig_sr=sr, target_sr=16000)
        seconds = len(audio) / 16000
        if seconds > 30:
            print(f"warning: {name} is {seconds:.1f}s; Whisper sees only the first 30s")
        feats = processor(audio, sampling_rate=16000, return_tensors="np").input_features
        feats.astype(np.float32).tofile(os.path.join(inputs_dir, f"input_{i}.raw"))
        lines.append(f"inputs/input_{i}.raw")
        manifest.append({"index": i, "name": name, "seconds": round(seconds, 2), "reference": ref,
                         "shape": list(feats.shape)})
    with open(os.path.join(args.run_dir, "input_list.txt"), "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(args.run_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    print(f"prepared {len(items)} inputs, feature shape {manifest[0]['shape'] if manifest else '-'}")


def score(args):
    import jiwer
    from transformers import AutoTokenizer

    normalize = AutoTokenizer.from_pretrained(args.hf_model).normalize  # Whisper English normalizer
    with open(os.path.join(args.run_dir, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    refs, hyps = [], []
    for m in manifest:
        path = os.path.join(args.run_dir, "outputs", f"output_{m['index']}.txt")
        hyp = open(path, encoding="utf-8").read() if os.path.exists(path) else "<missing>"
        # The runner emits Whisper's control tokens (<|en|><|transcribe|>...<|endoftext|>).
        hyp = re.sub(r"<\|[^|]*\|>", "", hyp).strip()
        print(f"[{m['index']}] {m['name']} ({m['seconds']}s)\n  NPU: {hyp}")
        if m["reference"]:
            print(f"  REF: {m['reference']}")
            refs.append(normalize(m["reference"]))
            hyps.append(normalize(hyp))
    if refs:
        print(f"\nWER over {len(refs)} clips (Whisper-normalized): {jiwer.wer(refs, hyps) * 100:.1f}%")


parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=["prep", "score"])
parser.add_argument("--run_dir", required=True)
parser.add_argument("--hf_model", default="openai/whisper-tiny")
parser.add_argument("--librispeech", type=int, default=0, help="number of LibriSpeech dummy clips")
parser.add_argument("--wav", nargs="*", help="your own audio files")
args = parser.parse_args()
prep(args) if args.mode == "prep" else score(args)
