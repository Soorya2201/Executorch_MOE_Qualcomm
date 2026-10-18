import importlib

for name in ("soundfile", "librosa", "torchaudio", "scipy", "transformers", "datasets", "jiwer"):
    try:
        mod = importlib.import_module(name)
        print(f"{name:12s} OK  {getattr(mod, '__version__', '')}")
    except Exception as e:
        print(f"{name:12s} MISSING ({type(e).__name__}: {e})")
