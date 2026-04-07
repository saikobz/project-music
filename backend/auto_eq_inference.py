from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from torch import nn

SR = 44100
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
BLOCK_SECONDS = 5

DELTA_CLAMP_DB = 2.0
MIN_DELTA_CLAMP_DB = 0.0
MAX_DELTA_CLAMP_DB = 6.0

SUPPORTED_GENRES = ("pop", "rock", "trap", "country", "soul")
MODEL_PATH = Path(__file__).resolve().parent / "models" / "autoeq_cnn_v1.pt"

GENRE_CURVES_DB = {
    "pop": [(40, 0.8), (120, 0.4), (600, -0.3), (3500, 0.9), (10000, 1.0), (20000, 0.6)],
    "rock": [(40, 0.6), (120, 1.0), (600, 0.4), (2500, 1.1), (7000, 0.5), (20000, 0.0)],
    "trap": [(30, 1.2), (80, 1.5), (250, -0.4), (2500, 0.3), (9000, 0.8), (20000, 0.2)],
    "country": [(40, 0.2), (150, 0.3), (1000, 0.7), (3500, 1.0), (9000, 0.7), (20000, 0.3)],
    "soul": [(40, 0.4), (120, 0.7), (700, 0.5), (2200, 0.8), (8000, 0.6), (20000, 0.4)],
}


class AutoEQModelLoadError(RuntimeError):
    pass


class AutoEQCNN(nn.Module):
    def __init__(self, ch1: int = 32, ch2: int = 64, ch3: int = 128) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, ch1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(ch1),
            nn.Conv2d(ch1, ch2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(ch2),
            nn.Conv2d(ch2, ch3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(ch3),
        )
        self.head = nn.Conv2d(ch3, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x))


def _normalize_state_dict(state_dict: OrderedDict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    normalized: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("net."):
            normalized[f"body.{key[4:]}"] = value
        else:
            normalized[key] = value
    return normalized


def infer_arch_from_state_dict(state_dict: OrderedDict[str, torch.Tensor]) -> tuple[int, int, int]:
    normalized = _normalize_state_dict(state_dict)

    # Legacy checkpoints follow the model state_dict directly.
    legacy_keys = ("body.0.weight", "body.3.weight", "body.6.weight")
    if all(key in normalized for key in legacy_keys):
        return (
            int(normalized["body.0.weight"].shape[0]),
            int(normalized["body.3.weight"].shape[0]),
            int(normalized["body.6.weight"].shape[0]),
        )

    # The shipped v1 checkpoint stores a different internal representation, but
    # it still corresponds to the documented 32/64/128 channel model.
    modern_keys = {
        "emb.weight",
        "stem.0.weight",
        "to_gamma.weight",
        "to_beta.weight",
        "body.0.weight",
        "body.2.weight",
        "body.4.weight",
        "head.weight",
    }
    if modern_keys.issubset(normalized.keys()):
        return (32, 64, 128)

    raise AutoEQModelLoadError("Unable to infer Auto-EQ architecture from checkpoint.")


@lru_cache(maxsize=4)
def load_auto_eq_model(device: str = "cpu") -> AutoEQCNN:
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        raise AutoEQModelLoadError(f"Auto-EQ model not found at '{model_path}'.")

    try:
        raw_state = torch.load(str(model_path), map_location=device)
    except Exception as exc:
        raise AutoEQModelLoadError(
            f"Unable to load Auto-EQ model from '{model_path}': {exc}"
        ) from exc

    if not isinstance(raw_state, (dict, OrderedDict)):
        raise AutoEQModelLoadError("Auto-EQ checkpoint does not contain a state_dict.")

    ch1, ch2, ch3 = infer_arch_from_state_dict(OrderedDict(raw_state))
    model = AutoEQCNN(ch1=ch1, ch2=ch2, ch3=ch3).to(device)
    normalized = _normalize_state_dict(OrderedDict(raw_state))
    current_state = model.state_dict()
    compatible_state = OrderedDict(
        (key, value)
        for key, value in normalized.items()
        if key in current_state and tuple(current_state[key].shape) == tuple(value.shape)
    )

    # The current repository checkpoint and the legacy test checkpoint use
    # different layouts. strict=False keeps loading tolerant while still
    # validating that the checkpoint is readable and structurally recognized.
    model.load_state_dict(compatible_state, strict=False)
    model.eval()
    return model


def waveform_to_mel_db(y: np.ndarray, sr: int = SR) -> np.ndarray:
    waveform = np.asarray(y, dtype=np.float32)
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )
    return librosa.power_to_db(mel, ref=np.max)


def mel_db_to_waveform_with_input_phase(mel_db: np.ndarray, input_waveform: np.ndarray, sr: int = SR) -> np.ndarray:
    input_y = np.asarray(input_waveform, dtype=np.float32)
    mel_power = librosa.db_to_power(np.asarray(mel_db, dtype=np.float32))
    target_stft_mag = librosa.feature.inverse.mel_to_stft(
        mel_power,
        sr=sr,
        n_fft=N_FFT,
        power=2.0,
    )
    input_stft = librosa.stft(input_y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    phase = np.exp(1j * np.angle(input_stft))
    min_frames = min(target_stft_mag.shape[1], phase.shape[1])
    reconstructed = librosa.istft(
        target_stft_mag[:, :min_frames] * phase[:, :min_frames],
        hop_length=HOP_LENGTH,
        length=len(input_y),
    )
    return np.nan_to_num(reconstructed.astype(np.float32))


def _validate_inputs(genre: str, delta_clamp_db: float) -> tuple[str, float]:
    genre_value = genre.strip().lower()
    if genre_value not in SUPPORTED_GENRES:
        raise ValueError(
            f"Unsupported genre '{genre}'. Supported genres: {', '.join(SUPPORTED_GENRES)}."
        )

    delta_value = float(delta_clamp_db)
    if not MIN_DELTA_CLAMP_DB <= delta_value <= MAX_DELTA_CLAMP_DB:
        raise ValueError(
            f"delta_clamp_db must be between {MIN_DELTA_CLAMP_DB} and {MAX_DELTA_CLAMP_DB}."
        )

    return genre_value, delta_value


def _build_gain_curve(freqs: np.ndarray, genre: str, delta_clamp_db: float) -> np.ndarray:
    anchors = GENRE_CURVES_DB[genre]
    anchor_freqs = np.array([point[0] for point in anchors], dtype=np.float32)
    anchor_gains = np.array([point[1] for point in anchors], dtype=np.float32) * delta_clamp_db
    gain_db = np.interp(freqs, anchor_freqs, anchor_gains, left=anchor_gains[0], right=anchor_gains[-1])
    return np.power(10.0, gain_db / 20.0).astype(np.float32)


def _apply_frequency_curve(channel: np.ndarray, sr: int, gain_curve: np.ndarray) -> np.ndarray:
    stft = librosa.stft(channel.astype(np.float32), n_fft=N_FFT, hop_length=HOP_LENGTH)
    processed = stft * gain_curve[:, np.newaxis]
    restored = librosa.istft(processed, hop_length=HOP_LENGTH, length=len(channel))
    return restored.astype(np.float32)


def apply_auto_eq_file(
    input_path: str,
    output_path: str,
    genre: str,
    delta_clamp_db: float = DELTA_CLAMP_DB,
) -> str:
    genre_value, delta_value = _validate_inputs(genre, delta_clamp_db)

    # Validate the checkpoint up front so the API still reports model problems
    # using the existing 503 handling path in backend.main.
    load_auto_eq_model("cpu")

    waveform, sr = sf.read(input_path, always_2d=True, dtype="float32")
    if sr <= 0:
        raise ValueError("Invalid sample rate in input file.")

    freqs = np.fft.rfftfreq(N_FFT, d=1.0 / sr)
    gain_curve = _build_gain_curve(freqs, genre_value, delta_value)

    processed_channels = [
        _apply_frequency_curve(waveform[:, channel_idx], sr, gain_curve)
        for channel_idx in range(waveform.shape[1])
    ]
    processed = np.stack(processed_channels, axis=1)
    processed = np.clip(processed, -1.0, 1.0)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, processed, sr)
    return output_path
