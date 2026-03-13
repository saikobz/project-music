# -*- coding: utf-8 -*-
"""
Auto-EQ inference helpers compatible with the current training checkpoint.

Supports:
- Current FiLM checkpoint from train.py (genre-conditioned, normalized mel input)
- Legacy CNN checkpoint format used by older inference code
"""

import logging
import os
from functools import lru_cache
from typing import Any, Mapping

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

SR = 44100
N_FFT = 2048
HOP = 512
N_MELS = 128
TOP_DB = 80.0
SEGMENT_SECONDS = 5
OVERLAP_SECONDS = 0.50
MEL_DB_MIN = -80.0
MEL_DB_MAX = 0.0
DELTA_CLAMP_DB = 2.0
DELTA_SMOOTH_FREQ_BINS = 7
DELTA_SMOOTH_TIME_FRAMES = 7
OUTPUT_BLEND = 0.20
MAX_RMS_GAIN_DB = 1.0
PEAK_LIMIT = 0.98
HF_TAPER_START = 0.65
HF_TAPER_END = 0.35
MIN_BLEND_SCALE = 0.45
MIN_DELTA_SCALE = 0.50
LOW_RMS_THRESHOLD = 0.035
HIGH_HF_RATIO_THRESHOLD = 0.22

GENRES = ["trap", "pop", "rock", "soul", "country"]
GENRE2ID = {genre: idx for idx, genre in enumerate(GENRES)}
DEFAULT_GENRE = "pop"

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "autoeq_cnn_v1.pt")
logger = logging.getLogger(__name__)


class AutoEQModelLoadError(RuntimeError):
    """Raised when an Auto-EQ checkpoint cannot be loaded."""


class AutoEQCNN(nn.Module):
    def __init__(self, ch1: int = 32, ch2: int = 64, ch3: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, ch1, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch1),
            nn.ReLU(),
            nn.Conv2d(ch1, ch2, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch2),
            nn.ReLU(),
            nn.Conv2d(ch2, ch3, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch3),
            nn.ReLU(),
            nn.Conv2d(ch3, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class AutoEQFiLM(nn.Module):
    def __init__(self, n_genres: int, ch: int = 256):
        super().__init__()
        self.emb = nn.Embedding(n_genres, 32)

        self.stem = nn.Sequential(
            nn.Conv2d(1, ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GELU(),
        )

        self.to_gamma = nn.Linear(32, ch)
        self.to_beta = nn.Linear(32, ch)

        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GELU(),
        )

        self.head = nn.Conv2d(ch, 1, 1)

    def forward(self, x: torch.Tensor, gid: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        g = self.emb(gid)

        gamma = self.to_gamma(g).unsqueeze(-1).unsqueeze(-1)
        beta = self.to_beta(g).unsqueeze(-1).unsqueeze(-1)

        h = h * (1 + gamma) + beta
        h = self.body(h)
        return x + self.head(h)


def waveform_to_mel_db(y: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP,
        n_mels=N_MELS,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=1.0)
    mel_db = np.clip(mel_db, MEL_DB_MIN, MEL_DB_MAX)
    return mel_db.astype(np.float32)


def mel_db_to_norm(mel_db: np.ndarray) -> np.ndarray:
    mel_db = np.clip(mel_db, MEL_DB_MIN, MEL_DB_MAX)
    mel_norm = (mel_db + TOP_DB) / TOP_DB
    return np.clip(mel_norm, 0.0, 1.0).astype(np.float32)


def mel_norm_to_db(mel_norm: np.ndarray) -> np.ndarray:
    mel_norm = np.clip(mel_norm, 0.0, 1.0)
    return (mel_norm * TOP_DB - TOP_DB).astype(np.float32)


def mel_db_to_waveform_with_input_phase(mel_db: np.ndarray, y_ref: np.ndarray) -> np.ndarray:
    mel = librosa.db_to_power(np.clip(mel_db, MEL_DB_MIN, MEL_DB_MAX)).astype(np.float32)
    mag_pred = librosa.feature.inverse.mel_to_stft(mel, sr=SR, n_fft=N_FFT)
    stft_ref = librosa.stft(y_ref, n_fft=N_FFT, hop_length=HOP)

    if mag_pred.ndim != 2 or stft_ref.ndim != 2:
        raise ValueError("Invalid spectrogram dimensions for phase reconstruction.")

    time_bins = min(mag_pred.shape[1], stft_ref.shape[1])
    if time_bins <= 0:
        return np.zeros(len(y_ref), dtype=np.float32)

    mag_pred = mag_pred[:, :time_bins]
    phase = np.angle(stft_ref[:, :time_bins])
    stft_new = mag_pred * np.exp(1j * phase)
    y = librosa.istft(stft_new, hop_length=HOP, length=len(y_ref))
    return y.astype(np.float32)


def match_loudness_rms(y_ref: np.ndarray, y_out: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    ref_rms = np.sqrt(np.mean(y_ref**2)) + eps
    out_rms = np.sqrt(np.mean(y_out**2)) + eps
    gain = ref_rms / out_rms
    max_gain = 10.0 ** (MAX_RMS_GAIN_DB / 20.0)
    gain = float(np.clip(gain, 1.0 / max_gain, max_gain))
    return y_out * gain


def limit_peak(y: np.ndarray, peak_limit: float = PEAK_LIMIT) -> np.ndarray:
    peak = float(np.max(np.abs(y))) if y.size > 0 else 0.0
    if peak <= peak_limit or peak <= 0.0:
        return y.astype(np.float32)
    return (y * (peak_limit / peak)).astype(np.float32)


def compute_adaptive_strength(chunk: np.ndarray, mel_db: np.ndarray) -> tuple[float, float]:
    chunk_rms = float(np.sqrt(np.mean(chunk**2))) if chunk.size > 0 else 0.0
    mel_power = librosa.db_to_power(mel_db)
    if mel_power.ndim != 2 or mel_power.shape[0] < 4:
        return 1.0, 1.0

    split_bin = max(int(mel_power.shape[0] * 0.65), 1)
    hf_power = float(np.mean(mel_power[split_bin:, :]))
    total_power = float(np.mean(mel_power)) + 1e-8
    hf_ratio = hf_power / total_power

    blend_scale = 1.0
    delta_scale = 1.0

    if chunk_rms < LOW_RMS_THRESHOLD:
        # Quiet passages expose artifacts more easily, so back off processing.
        quiet_scale = max(MIN_BLEND_SCALE, chunk_rms / max(LOW_RMS_THRESHOLD, 1e-8))
        blend_scale *= quiet_scale
        delta_scale *= max(MIN_DELTA_SCALE, quiet_scale)

    if hf_ratio > HIGH_HF_RATIO_THRESHOLD:
        hf_scale = max(MIN_BLEND_SCALE, HIGH_HF_RATIO_THRESHOLD / max(hf_ratio, 1e-8))
        blend_scale *= hf_scale
        delta_scale *= max(MIN_DELTA_SCALE, hf_scale)

    blend_scale = float(np.clip(blend_scale, MIN_BLEND_SCALE, 1.0))
    delta_scale = float(np.clip(delta_scale, MIN_DELTA_SCALE, 1.0))
    return blend_scale, delta_scale


def _smooth_axis(arr: np.ndarray, axis: int, taps: int) -> np.ndarray:
    if taps <= 1:
        return arr

    taps = max(1, int(taps))
    if taps % 2 == 0:
        taps += 1

    kernel = np.hanning(taps).astype(np.float32)
    kernel_sum = float(kernel.sum())
    if kernel_sum <= 0.0:
        return arr
    kernel /= kernel_sum

    pad = taps // 2
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (pad, pad)
    padded = np.pad(arr, pad_width, mode="edge")

    return np.apply_along_axis(
        lambda v: np.convolve(v, kernel, mode="valid"),
        axis=axis,
        arr=padded,
    ).astype(np.float32)


def smooth_delta_mel(delta_db: np.ndarray) -> np.ndarray:
    delta_db = _smooth_axis(delta_db, axis=0, taps=DELTA_SMOOTH_FREQ_BINS)
    delta_db = _smooth_axis(delta_db, axis=1, taps=DELTA_SMOOTH_TIME_FRAMES)
    return delta_db.astype(np.float32)


def taper_high_freq_delta(delta_db: np.ndarray) -> np.ndarray:
    if delta_db.ndim != 2 or delta_db.shape[0] <= 1:
        return delta_db.astype(np.float32)

    taper = np.ones(delta_db.shape[0], dtype=np.float32)
    start_idx = int(np.clip(HF_TAPER_START, 0.0, 1.0) * (delta_db.shape[0] - 1))
    if start_idx < delta_db.shape[0] - 1:
        taper[start_idx:] = np.linspace(1.0, HF_TAPER_END, delta_db.shape[0] - start_idx, dtype=np.float32)
    return (delta_db * taper[:, None]).astype(np.float32)


def build_chunk_window(chunk_len: int, fade_len: int) -> np.ndarray:
    if chunk_len <= 1:
        return np.ones((max(chunk_len, 1),), dtype=np.float32)

    fade_len = min(max(int(fade_len), 0), chunk_len // 2)
    if fade_len <= 0:
        return np.ones((chunk_len,), dtype=np.float32)

    window = np.ones((chunk_len,), dtype=np.float32)
    fade = np.hanning(fade_len * 2).astype(np.float32)
    fade_in = fade[:fade_len]
    fade_out = fade[fade_len:]
    window[:fade_len] = np.maximum(window[:fade_len] * fade_in, 1e-4)
    window[-fade_len:] = np.maximum(window[-fade_len:] * fade_out, 1e-4)
    return window


def extract_model_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    elif isinstance(checkpoint, Mapping):
        state = checkpoint
    else:
        raise AutoEQModelLoadError("Checkpoint format is invalid (expected mapping or mapping['state_dict']).")

    if not isinstance(state, Mapping):
        raise AutoEQModelLoadError("Checkpoint 'state_dict' is not a mapping.")
    return state


def normalize_state_dict_keys(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not isinstance(key, str) or not torch.is_tensor(value):
            continue

        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module.") :]
        if new_key.startswith("_orig_mod."):
            new_key = new_key[len("_orig_mod.") :]
        if new_key.startswith("net."):
            new_key = "body." + new_key[len("net.") :]

        normalized[new_key] = value

    if not normalized:
        raise AutoEQModelLoadError("No tensor parameters found in checkpoint state dict.")
    return normalized


def infer_legacy_arch_from_state_dict(state: Mapping[str, torch.Tensor]) -> tuple[int, int, int]:
    required = ("body.0.weight", "body.3.weight", "body.6.weight", "body.9.weight")
    missing = [key for key in required if key not in state]
    if missing:
        raise AutoEQModelLoadError(f"Checkpoint missing required layer weights: {', '.join(missing)}")

    w0 = state["body.0.weight"]
    w3 = state["body.3.weight"]
    w6 = state["body.6.weight"]
    w9 = state["body.9.weight"]

    if any(weight.ndim != 4 for weight in (w0, w3, w6, w9)):
        raise AutoEQModelLoadError("Checkpoint contains invalid convolution weight dimensions.")

    ch1 = int(w0.shape[0])
    ch2 = int(w3.shape[0])
    ch3 = int(w6.shape[0])

    if int(w3.shape[1]) != ch1:
        raise AutoEQModelLoadError("Checkpoint shape mismatch: body.3.weight input channels do not match body.0.")
    if int(w6.shape[1]) != ch2:
        raise AutoEQModelLoadError("Checkpoint shape mismatch: body.6.weight input channels do not match body.3.")
    if int(w9.shape[1]) != ch3 or int(w9.shape[0]) != 1:
        raise AutoEQModelLoadError("Checkpoint shape mismatch: body.9.weight is incompatible with inferred channels.")

    return ch1, ch2, ch3


def infer_film_arch_from_state_dict(state: Mapping[str, torch.Tensor]) -> tuple[int, int]:
    required = (
        "emb.weight",
        "stem.0.weight",
        "stem.2.weight",
        "to_gamma.weight",
        "to_beta.weight",
        "body.0.weight",
        "body.2.weight",
        "body.4.weight",
        "head.weight",
    )
    missing = [key for key in required if key not in state]
    if missing:
        raise AutoEQModelLoadError(f"Checkpoint missing required FiLM layer weights: {', '.join(missing)}")

    emb_weight = state["emb.weight"]
    stem0 = state["stem.0.weight"]
    stem2 = state["stem.2.weight"]
    gamma = state["to_gamma.weight"]
    beta = state["to_beta.weight"]
    body0 = state["body.0.weight"]
    body2 = state["body.2.weight"]
    body4 = state["body.4.weight"]
    head = state["head.weight"]

    if any(weight.ndim != 2 for weight in (emb_weight, gamma, beta)):
        raise AutoEQModelLoadError("Checkpoint contains invalid FiLM linear/embedding weight dimensions.")
    if any(weight.ndim != 4 for weight in (stem0, stem2, body0, body2, body4, head)):
        raise AutoEQModelLoadError("Checkpoint contains invalid FiLM convolution weight dimensions.")

    n_genres = int(emb_weight.shape[0])
    emb_dim = int(emb_weight.shape[1])
    ch = int(stem0.shape[0])

    if emb_dim != 32:
        raise AutoEQModelLoadError(f"Unsupported FiLM embedding size: expected 32, got {emb_dim}")
    if int(stem0.shape[1]) != 1 or int(stem2.shape[0]) != ch or int(stem2.shape[1]) != ch:
        raise AutoEQModelLoadError("Checkpoint stem shape mismatch.")
    if int(gamma.shape[0]) != ch or int(gamma.shape[1]) != emb_dim:
        raise AutoEQModelLoadError("Checkpoint to_gamma shape mismatch.")
    if int(beta.shape[0]) != ch or int(beta.shape[1]) != emb_dim:
        raise AutoEQModelLoadError("Checkpoint to_beta shape mismatch.")
    if any(int(weight.shape[0]) != ch or int(weight.shape[1]) != ch for weight in (body0, body2, body4)):
        raise AutoEQModelLoadError("Checkpoint body convolution shape mismatch.")
    if int(head.shape[0]) != 1 or int(head.shape[1]) != ch:
        raise AutoEQModelLoadError("Checkpoint head shape mismatch.")

    return n_genres, ch


@lru_cache(maxsize=1)
def load_auto_eq_model(device: str = "cpu") -> nn.Module:
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        raw_state = extract_model_state_dict(checkpoint)
        state = normalize_state_dict_keys(raw_state)

        if "emb.weight" in state:
            n_genres, ch = infer_film_arch_from_state_dict(state)
            logger.info(
                "Loading Auto-EQ FiLM model path=%s device=%s genres=%d channels=%d",
                MODEL_PATH,
                device,
                n_genres,
                ch,
            )
            model = AutoEQFiLM(n_genres=n_genres, ch=ch)
            model.requires_gid = True
            model.input_representation = "mel_norm"
        else:
            ch1, ch2, ch3 = infer_legacy_arch_from_state_dict(state)
            logger.info(
                "Loading legacy Auto-EQ model path=%s device=%s channels=(%d,%d,%d)",
                MODEL_PATH,
                device,
                ch1,
                ch2,
                ch3,
            )
            model = AutoEQCNN(ch1=ch1, ch2=ch2, ch3=ch3)
            model.requires_gid = False
            model.input_representation = "mel_db"

        model.load_state_dict(state, strict=True)
        model.to(device)
        model.eval()
        return model
    except AutoEQModelLoadError:
        raise
    except Exception as exc:
        raise AutoEQModelLoadError(f"Failed to load Auto-EQ model from {MODEL_PATH}: {exc}") from exc


def resolve_genre_id(model: nn.Module, genre: str | int | None) -> int | None:
    if not getattr(model, "requires_gid", False):
        return None

    if genre is None:
        genre = DEFAULT_GENRE
        logger.warning("No genre provided for FiLM checkpoint. Falling back to default genre=%s", genre)

    if isinstance(genre, int):
        if 0 <= genre < len(GENRES):
            return genre
        raise ValueError(f"Genre id out of range: {genre}. Valid range is 0..{len(GENRES) - 1}.")

    genre_key = str(genre).strip().lower()
    if genre_key not in GENRE2ID:
        raise ValueError(f"Unknown genre '{genre}'. Valid genres: {', '.join(GENRES)}")
    return GENRE2ID[genre_key]


def predict_mel_db(
    model: nn.Module,
    mel_db: np.ndarray,
    device: str,
    genre_id: int | None,
) -> np.ndarray:
    if getattr(model, "input_representation", "mel_db") == "mel_norm":
        model_input = mel_db_to_norm(mel_db)
        mel_tensor = torch.from_numpy(model_input).unsqueeze(0).unsqueeze(0).to(device)
        gid_tensor = torch.tensor([genre_id], dtype=torch.long, device=device)
        with torch.no_grad():
            pred_mel_norm = model(mel_tensor, gid_tensor).squeeze(0).squeeze(0).cpu().numpy()
        return mel_norm_to_db(pred_mel_norm)

    mel_tensor = torch.from_numpy(mel_db).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_mel_db = model(mel_tensor).squeeze(0).squeeze(0).cpu().numpy()
    return pred_mel_db.astype(np.float32)


def apply_auto_eq_file(input_path: str, output_path: str, genre: str | int | None = None) -> str:
    y, _ = librosa.load(input_path, sr=SR, mono=True)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_auto_eq_model(device)
    genre_id = resolve_genre_id(model, genre)

    segment_samples = int(SEGMENT_SECONDS * SR)
    overlap_samples = min(int(OVERLAP_SECONDS * SR), max(segment_samples // 2, 1))
    step_samples = max(segment_samples - overlap_samples, 1)

    mixed = np.zeros(len(y), dtype=np.float32)
    weight = np.zeros(len(y), dtype=np.float32)

    for start in range(0, len(y), step_samples):
        end = min(start + segment_samples, len(y))
        chunk = y[start:end]
        if chunk.size == 0:
            continue

        mel_db = waveform_to_mel_db(chunk)
        pred_mel_db = predict_mel_db(model, mel_db, device, genre_id)
        blend_scale, delta_scale = compute_adaptive_strength(chunk, mel_db)

        delta = pred_mel_db - mel_db
        delta = smooth_delta_mel(delta)
        delta = taper_high_freq_delta(delta)
        delta *= delta_scale
        delta = np.clip(delta, -DELTA_CLAMP_DB, DELTA_CLAMP_DB)
        mel_out = np.clip(mel_db + delta, MEL_DB_MIN, MEL_DB_MAX)

        enhanced_chunk = mel_db_to_waveform_with_input_phase(mel_out, chunk)
        if enhanced_chunk.shape[0] != chunk.shape[0]:
            if enhanced_chunk.shape[0] > chunk.shape[0]:
                enhanced_chunk = enhanced_chunk[: chunk.shape[0]]
            else:
                pad = chunk.shape[0] - enhanced_chunk.shape[0]
                enhanced_chunk = np.pad(enhanced_chunk, (0, pad))

        output_blend = OUTPUT_BLEND * blend_scale
        enhanced_chunk = ((1.0 - output_blend) * chunk + output_blend * enhanced_chunk).astype(np.float32)
        enhanced_chunk = match_loudness_rms(chunk, enhanced_chunk)
        enhanced_chunk = limit_peak(enhanced_chunk)
        enhanced_chunk = np.clip(enhanced_chunk, -1.0, 1.0).astype(np.float32)

        fade = min(overlap_samples, chunk.shape[0] // 2)
        chunk_weight = build_chunk_window(chunk.shape[0], fade)
        if start == 0:
            chunk_weight[:fade] = 1.0
        if end >= len(y):
            chunk_weight[-fade:] = 1.0

        mixed[start:end] += enhanced_chunk * chunk_weight
        weight[start:end] += chunk_weight

    enhanced = np.divide(mixed, np.maximum(weight, 1e-8), out=np.zeros_like(mixed), where=weight > 0)
    sf.write(output_path, enhanced, SR)
    return output_path
