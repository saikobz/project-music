from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import Any

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
LSTM_TIME_SMOOTH_FRAMES = 17
LSTM_FREQ_SMOOTH_BINS = 9
LSTM_MODEL_BLEND = 0.65
LSTM_HIGH_FREQ_DAMP_START_HZ = 8000.0
LSTM_HIGH_FREQ_DAMP_END_HZ = 16000.0
LSTM_HIGH_FREQ_DAMP_MIN = 0.55

SUPPORTED_GENRES = ("pop", "rock", "trap", "country", "soul")

AUTO_EQ_MODEL_CNN_V1 = "cnn-v1"
AUTO_EQ_MODEL_LSTM_LAST = "lstm-last"
DEFAULT_AUTO_EQ_MODEL_ID = AUTO_EQ_MODEL_CNN_V1

MODEL_PATH = Path(__file__).resolve().parent / "models" / "autoeq_cnn_v1.pt"
MODEL_PATHS = {
    AUTO_EQ_MODEL_CNN_V1: MODEL_PATH,
    AUTO_EQ_MODEL_LSTM_LAST: Path(__file__).resolve().parent / "models" / "autoeq_lstm_last.pt",
}
SUPPORTED_AUTO_EQ_MODELS = tuple(MODEL_PATHS.keys())

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


class AutoEQLSTM(nn.Module):
    def __init__(
        self,
        *,
        n_mels: int = N_MELS,
        num_genres: int = len(SUPPORTED_GENRES),
        emb_dim: int = 32,
        model_ch: int = 256,
    ) -> None:
        super().__init__()
        self.emb = nn.Embedding(num_genres, emb_dim)
        self.input_proj = nn.Linear(n_mels + emb_dim, model_ch)
        self.input_norm = nn.LayerNorm(model_ch)
        self.lstm = nn.LSTM(
            input_size=model_ch,
            hidden_size=model_ch,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.output_proj = nn.Sequential(
            nn.Linear(model_ch * 2, model_ch),
            nn.ReLU(inplace=True),
            nn.Linear(model_ch, n_mels),
        )

    def forward(self, mel_db_frames: torch.Tensor, genre_ids: torch.Tensor) -> torch.Tensor:
        genre_embedding = self.emb(genre_ids)
        genre_embedding = genre_embedding[:, None, :].expand(-1, mel_db_frames.shape[1], -1)
        x = torch.cat((mel_db_frames, genre_embedding), dim=-1)
        x = self.input_proj(x)
        x = self.input_norm(x)
        x, _ = self.lstm(x)
        return self.output_proj(x)


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

    legacy_keys = ("body.0.weight", "body.3.weight", "body.6.weight")
    if all(key in normalized for key in legacy_keys):
        return (
            int(normalized["body.0.weight"].shape[0]),
            int(normalized["body.3.weight"].shape[0]),
            int(normalized["body.6.weight"].shape[0]),
        )

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


def _validate_model_id(model_id: str) -> str:
    model_value = model_id.strip().lower()
    if model_value not in SUPPORTED_AUTO_EQ_MODELS:
        raise ValueError(
            f"Unsupported Auto-EQ model '{model_id}'. Supported models: {', '.join(SUPPORTED_AUTO_EQ_MODELS)}."
        )
    return model_value


def _get_model_path(model_id: str) -> Path:
    if model_id == AUTO_EQ_MODEL_CNN_V1:
        return Path(MODEL_PATH)
    return Path(MODEL_PATHS[model_id])


def _load_cnn_model(raw_state: OrderedDict[str, torch.Tensor], device: str, model_id: str) -> AutoEQCNN:
    ch1, ch2, ch3 = infer_arch_from_state_dict(OrderedDict(raw_state))
    model = AutoEQCNN(ch1=ch1, ch2=ch2, ch3=ch3).to(device)
    normalized = _normalize_state_dict(OrderedDict(raw_state))
    current_state = model.state_dict()
    compatible_state = OrderedDict(
        (key, value)
        for key, value in normalized.items()
        if key in current_state and tuple(current_state[key].shape) == tuple(value.shape)
    )

    model.load_state_dict(compatible_state, strict=False)
    model.eval()
    model.auto_eq_kind = "cnn"
    model.auto_eq_model_id = model_id
    return model


def _load_lstm_model(checkpoint: dict[str, Any], device: str, model_id: str) -> AutoEQLSTM:
    state_dict = checkpoint.get("ema_state_dict") or checkpoint.get("model_state_dict")
    if not isinstance(state_dict, (dict, OrderedDict)):
        raise AutoEQModelLoadError("Auto-EQ LSTM checkpoint does not contain a valid state_dict.")

    config = checkpoint.get("config") or {}
    if not isinstance(config, dict):
        raise AutoEQModelLoadError("Auto-EQ LSTM checkpoint config is invalid.")

    genre2id = config.get("genre2id")
    if not isinstance(genre2id, dict) or not genre2id:
        raise AutoEQModelLoadError("Auto-EQ LSTM checkpoint is missing genre2id mapping.")

    emb_weight = state_dict.get("emb.weight")
    input_proj_weight = state_dict.get("input_proj.weight")
    if emb_weight is None or input_proj_weight is None:
        raise AutoEQModelLoadError("Auto-EQ LSTM checkpoint is missing required weights.")

    emb_dim = int(emb_weight.shape[1])
    n_mels = int(config.get("n_mels", N_MELS))
    model_ch = int(config.get("model_ch", input_proj_weight.shape[0]))
    model = AutoEQLSTM(
        n_mels=n_mels,
        num_genres=len(genre2id),
        emb_dim=emb_dim,
        model_ch=model_ch,
    ).to(device)
    model.load_state_dict(OrderedDict(state_dict), strict=True)
    model.eval()
    model.auto_eq_kind = "lstm"
    model.auto_eq_model_id = model_id
    model.auto_eq_genre2id = {str(key): int(value) for key, value in genre2id.items()}
    model.auto_eq_sample_rate = int(config.get("sample_rate", SR))
    return model


@lru_cache(maxsize=8)
def load_auto_eq_model(device: str = "cpu", model_id: str = DEFAULT_AUTO_EQ_MODEL_ID) -> nn.Module:
    model_key = _validate_model_id(model_id)
    model_path = _get_model_path(model_key)
    if not model_path.exists():
        raise AutoEQModelLoadError(f"Auto-EQ model not found at '{model_path}'.")

    try:
        raw_state = torch.load(str(model_path), map_location=device)
    except Exception as exc:
        raise AutoEQModelLoadError(
            f"Unable to load Auto-EQ model from '{model_path}': {exc}"
        ) from exc

    if isinstance(raw_state, dict) and (
        isinstance(raw_state.get("ema_state_dict"), (dict, OrderedDict))
        or isinstance(raw_state.get("model_state_dict"), (dict, OrderedDict))
    ):
        return _load_lstm_model(raw_state, device, model_key)

    if isinstance(raw_state, (dict, OrderedDict)):
        return _load_cnn_model(OrderedDict(raw_state), device, model_key)

    raise AutoEQModelLoadError("Auto-EQ checkpoint does not contain a supported state_dict.")


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


def _validate_inputs(genre: str, delta_clamp_db: float, model_id: str) -> tuple[str, float, str]:
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

    model_value = _validate_model_id(model_id)
    return genre_value, delta_value, model_value


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


def _smooth_1d(values: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return np.asarray(values, dtype=np.float32)

    kernel_size = max(int(kernel_size), 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    radius = kernel_size // 2
    padded = np.pad(np.asarray(values, dtype=np.float32), (radius, radius), mode="edge")
    kernel = np.ones(kernel_size, dtype=np.float32) / float(kernel_size)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def _smooth_axis(values: np.ndarray, axis: int, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return np.asarray(values, dtype=np.float32)

    return np.apply_along_axis(_smooth_1d, axis, np.asarray(values, dtype=np.float32), kernel_size).astype(
        np.float32
    )


def _match_channel_length(channel: np.ndarray, target_length: int) -> np.ndarray:
    if channel.shape[0] == target_length:
        return channel.astype(np.float32)
    if channel.shape[0] > target_length:
        return channel[:target_length].astype(np.float32)
    return np.pad(channel, (0, target_length - channel.shape[0]), mode="constant").astype(np.float32)


def _predict_lstm_delta_mel_db(model: AutoEQLSTM, mel_db: np.ndarray, genre: str) -> np.ndarray:
    genre2id = getattr(model, "auto_eq_genre2id", None)
    if not isinstance(genre2id, dict) or genre not in genre2id:
        raise ValueError(f"Genre '{genre}' is not supported by the selected Auto-EQ model.")

    device = next(model.parameters()).device
    mel_frames = torch.from_numpy(mel_db.T.astype(np.float32)).unsqueeze(0).to(device)
    genre_ids = torch.tensor([genre2id[genre]], dtype=torch.long, device=device)

    with torch.no_grad():
        predicted = model(mel_frames, genre_ids)

    return predicted.squeeze(0).detach().cpu().numpy().T.astype(np.float32)


def _build_lstm_gain_curve(
    model: AutoEQLSTM,
    channel: np.ndarray,
    sr: int,
    genre: str,
    delta_clamp_db: float,
) -> np.ndarray:
    model_sr = int(getattr(model, "auto_eq_sample_rate", sr))
    working_channel = np.asarray(channel, dtype=np.float32)
    if model_sr != sr:
        working_channel = librosa.resample(working_channel, orig_sr=sr, target_sr=model_sr)

    mel_db = waveform_to_mel_db(working_channel, sr=model_sr)
    predicted_delta_db = _predict_lstm_delta_mel_db(model, mel_db, genre)

    smoothed_delta_db = np.clip(predicted_delta_db, -delta_clamp_db, delta_clamp_db)
    smoothed_delta_db = _smooth_axis(smoothed_delta_db, axis=1, kernel_size=LSTM_TIME_SMOOTH_FRAMES)
    smoothed_delta_db = _smooth_axis(smoothed_delta_db, axis=0, kernel_size=LSTM_FREQ_SMOOTH_BINS)

    mel_curve_db = np.median(smoothed_delta_db, axis=1).astype(np.float32)
    mel_curve_db = _smooth_1d(mel_curve_db, kernel_size=LSTM_FREQ_SMOOTH_BINS)
    mel_curve_db = np.clip(mel_curve_db * LSTM_MODEL_BLEND, -delta_clamp_db, delta_clamp_db)

    mel_freqs = librosa.mel_frequencies(n_mels=mel_curve_db.shape[0], fmin=0.0, fmax=model_sr / 2.0)
    fft_freqs = np.fft.rfftfreq(N_FFT, d=1.0 / sr)
    gain_db = np.interp(fft_freqs, mel_freqs, mel_curve_db, left=mel_curve_db[0], right=mel_curve_db[-1]).astype(
        np.float32
    )

    # Dampen the top-end boost so the LSTM behaves more like EQ and less like resynthesis.
    high_freq_taper = np.ones_like(gain_db, dtype=np.float32)
    band = (fft_freqs >= LSTM_HIGH_FREQ_DAMP_START_HZ) & (fft_freqs <= LSTM_HIGH_FREQ_DAMP_END_HZ)
    if np.any(band):
        high_freq_taper[band] = np.interp(
            fft_freqs[band],
            [LSTM_HIGH_FREQ_DAMP_START_HZ, LSTM_HIGH_FREQ_DAMP_END_HZ],
            [1.0, LSTM_HIGH_FREQ_DAMP_MIN],
        ).astype(np.float32)
    high_freq_taper[fft_freqs > LSTM_HIGH_FREQ_DAMP_END_HZ] = LSTM_HIGH_FREQ_DAMP_MIN
    gain_db *= high_freq_taper

    return np.power(10.0, gain_db / 20.0).astype(np.float32)


def _apply_lstm_auto_eq_channel(
    channel: np.ndarray,
    sr: int,
    model: AutoEQLSTM,
    genre: str,
    delta_clamp_db: float,
) -> np.ndarray:
    gain_curve = _build_lstm_gain_curve(model, channel, sr, genre, delta_clamp_db)
    processed_channel = _apply_frequency_curve(np.asarray(channel, dtype=np.float32), sr, gain_curve)
    return _match_channel_length(processed_channel, len(channel))


def apply_auto_eq_file(
    input_path: str,
    output_path: str,
    genre: str,
    delta_clamp_db: float = DELTA_CLAMP_DB,
    model_id: str = DEFAULT_AUTO_EQ_MODEL_ID,
) -> str:
    genre_value, delta_value, model_key = _validate_inputs(genre, delta_clamp_db, model_id)
    model = load_auto_eq_model("cpu", model_id=model_key)

    waveform, sr = sf.read(input_path, always_2d=True, dtype="float32")
    if sr <= 0:
        raise ValueError("Invalid sample rate in input file.")

    if getattr(model, "auto_eq_kind", "cnn") == "lstm":
        processed_channels = [
            _apply_lstm_auto_eq_channel(waveform[:, channel_idx], sr, model, genre_value, delta_value)
            for channel_idx in range(waveform.shape[1])
        ]
    else:
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
