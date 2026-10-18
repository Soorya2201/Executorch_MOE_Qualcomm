"""Whisper audio front-end in plain numpy (no torch/transformers on the device).

Matches Hugging Face's WhisperFeatureExtractor: 16 kHz mono, 25 ms Hann window (n_fft 400), 10 ms hop,
80 Slaney-normalised mel bands, log10, clamp to (max - 8), then (x + 4) / 4. One 30 s window gives a
float32 [1, 80, 3000] tensor: exactly the encoder input exported for the NPU.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
N_FFT = 400
HOP = 160
N_MELS = 80
WINDOW_SAMPLES = 30 * SAMPLE_RATE  # Whisper always sees 30 s


def _hz_to_mel(f):
    f = np.asanyarray(f, dtype=np.float64)
    lin = 3.0 * f / 200.0
    return np.where(f >= 1000.0, 15.0 + np.log(np.maximum(f, 1e-10) / 1000.0) / (np.log(6.4) / 27.0), lin)


def _mel_to_hz(m):
    m = np.asanyarray(m, dtype=np.float64)
    lin = 200.0 * m / 3.0
    return np.where(m >= 15.0, 1000.0 * np.exp((np.log(6.4) / 27.0) * (m - 15.0)), lin)


def _mel_filters() -> np.ndarray:
    """[n_fft//2+1, n_mels] triangular Slaney filters, 0-8 kHz."""
    fft_freqs = np.linspace(0, SAMPLE_RATE / 2, N_FFT // 2 + 1)
    mel_pts = np.linspace(_hz_to_mel(0.0), _hz_to_mel(SAMPLE_RATE / 2), N_MELS + 2)
    hz_pts = _mel_to_hz(mel_pts)
    diff = np.diff(hz_pts)
    slopes = hz_pts[None, :] - fft_freqs[:, None]
    down = -slopes[:, :-2] / diff[:-1]
    up = slopes[:, 2:] / diff[1:]
    fb = np.maximum(0.0, np.minimum(down, up))
    fb *= 2.0 / (hz_pts[2:N_MELS + 2] - hz_pts[:N_MELS])  # Slaney area normalisation
    return fb.astype(np.float64)


_FILTERS = _mel_filters()
_WINDOW = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT))  # periodic Hann


def log_mel(audio: np.ndarray) -> np.ndarray:
    """30 s (or shorter, zero-padded) of 16 kHz mono float audio -> float32 [1, 80, 3000]."""
    x = np.zeros(WINDOW_SAMPLES, dtype=np.float64)
    a = np.asarray(audio, dtype=np.float64)[:WINDOW_SAMPLES]
    x[:len(a)] = a
    x = np.pad(x, (N_FFT // 2, N_FFT // 2), mode="reflect")  # center=True
    n_frames = 1 + (len(x) - N_FFT) // HOP
    frames = np.lib.stride_tricks.as_strided(x, (n_frames, N_FFT), (x.strides[0] * HOP, x.strides[0]))
    power = np.abs(np.fft.rfft(frames * _WINDOW, axis=1)) ** 2  # [frames, 201]
    mel = power[:-1] @ _FILTERS  # drop the last frame -> 3000 frames, like HF
    logm = np.log10(np.maximum(mel, 1e-10))
    logm = np.maximum(logm, logm.max() - 8.0)
    return ((logm + 4.0) / 4.0).T[None].astype(np.float32)  # [1, 80, 3000]


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """PCM 16/24/32-bit or 32-bit float WAV -> (mono float32 in [-1, 1], sample rate)."""
    data = Path(path).read_bytes()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a WAV file; convert the recording to .wav first")
    pos, fmt, raw = 12, None, None
    while pos + 8 <= len(data):
        cid, size = data[pos:pos + 4], struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            fmt = struct.unpack("<HHIIHH", body[:16])
        elif cid == b"data":
            raw = body
        pos += 8 + size + (size & 1)
    if fmt is None or raw is None:
        raise ValueError("WAV file has no fmt/data chunk")
    tag, channels, rate, _, _, bits = fmt
    if tag == 0xFFFE:  # WAVE_FORMAT_EXTENSIBLE: treat as PCM/float by bit depth
        tag = 3 if bits == 32 and b"\x03\x00\x00\x00\x00\x00\x10\x00" in data[:200] else 1
    if tag == 3 and bits == 32:
        x = np.frombuffer(raw, "<f4").astype(np.float32)
    elif tag == 1 and bits == 16:
        x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
    elif tag == 1 and bits == 32:
        x = np.frombuffer(raw, "<i4").astype(np.float32) / 2147483648.0
    elif tag == 1 and bits == 24:
        b = np.frombuffer(raw[:len(raw) // 3 * 3], np.uint8).reshape(-1, 3)
        x = ((b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16))
             << 8 >> 8).astype(np.float32) / 8388608.0
    else:
        raise ValueError(f"unsupported WAV encoding (format {tag}, {bits}-bit)")
    x = x[: len(x) // channels * channels].reshape(-1, channels).mean(axis=1)
    return x, rate


def to_16k(audio: np.ndarray, rate: int) -> np.ndarray:
    """Linear-interpolation resample to 16 kHz (the recorder already writes 16 kHz)."""
    if rate == SAMPLE_RATE:
        return audio
    n = int(round(len(audio) * SAMPLE_RATE / rate))
    return np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)
