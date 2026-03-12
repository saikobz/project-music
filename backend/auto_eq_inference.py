"""
หน้าที่ของไฟล์นี้:
- โหลดโมเดล Auto-EQ จาก checkpoint และเตรียมโครงสร้างโมเดลให้พร้อมใช้งาน
- แปลงเสียงเป็น mel spectrogram, รัน inference ทีละช่วง และ reconstruct กลับเป็น waveform
- ทำ post-processing เช่น clamp การปรับ EQ, match loudness และ blend ช่วงที่ซ้อนกัน
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
SEGMENT_SECONDS = 5
OVERLAP_SECONDS = 0.0
MEL_DB_MIN = -80.0
MEL_DB_MAX = 0.0
DELTA_CLAMP_DB = 6.0

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
        # โมเดลทำนายค่าปรับเพิ่มจาก input เดิม ไม่ได้สร้าง mel ใหม่จากศูนย์
        residual = self.body(x)
        return x + residual


def waveform_to_mel_db(y: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP,
        n_mels=N_MELS,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = np.clip(mel_db, MEL_DB_MIN, MEL_DB_MAX)
    return mel_db.astype(np.float32)


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
    return y_out * gain


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
        # รองรับ checkpoint เก่าที่เคยถูก save ผ่าน DataParallel หรือใช้ชื่อชั้นว่า net.*
        if new_key.startswith("module."):
            new_key = new_key[len("module.") :]
        if new_key.startswith("net."):
            new_key = "body." + new_key[len("net.") :]

        normalized[new_key] = value

    if not normalized:
        raise AutoEQModelLoadError("No tensor parameters found in checkpoint state dict.")
    return normalized


def infer_arch_from_state_dict(state: Mapping[str, torch.Tensor]) -> tuple[int, int, int]:
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

    # อ่านจำนวน channel ของแต่ละ block จากน้ำหนักจริง เพื่อไม่ผูกกับสถาปัตยกรรมเดียวตายตัว
    return ch1, ch2, ch3


@lru_cache(maxsize=1)
def load_auto_eq_model(device: str = "cpu") -> AutoEQCNN:
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        raw_state = extract_model_state_dict(checkpoint)
        state = normalize_state_dict_keys(raw_state)
        ch1, ch2, ch3 = infer_arch_from_state_dict(state)

        logger.info(
            "Loading Auto-EQ model path=%s device=%s channels=(%d,%d,%d)",
            MODEL_PATH,
            device,
            ch1,
            ch2,
            ch3,
        )

        model = AutoEQCNN(ch1=ch1, ch2=ch2, ch3=ch3)
        model.load_state_dict(state, strict=True)
        model.to(device)
        model.eval()
        # cache โมเดลไว้หนึ่งชุดเพื่อลดเวลาโหลดซ้ำทุกครั้งที่เรียก endpoint
        return model
    except AutoEQModelLoadError:
        raise
    except Exception as exc:
        raise AutoEQModelLoadError(f"Failed to load Auto-EQ model from {MODEL_PATH}: {exc}") from exc


def apply_auto_eq_file(input_path: str, output_path: str) -> str:
    y, _ = librosa.load(input_path, sr=SR, mono=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_auto_eq_model(device)

    segment_samples = SEGMENT_SECONDS * SR
    overlap_samples = min(int(OVERLAP_SECONDS * SR), max(segment_samples // 2, 1))
    step_samples = max(segment_samples - overlap_samples, 1)

    mixed = np.zeros(len(y), dtype=np.float32)
    weight = np.zeros(len(y), dtype=np.float32)

    # ประมวลผลทีละช่วงเพื่อคุม memory และให้ไฟล์ยาว ๆ ใช้โมเดลได้
    for start in range(0, len(y), step_samples):
        end = min(start + segment_samples, len(y))
        chunk = y[start:end]
        if chunk.size == 0:
            continue
        mel_db = waveform_to_mel_db(chunk)

        mel_tensor = torch.from_numpy(mel_db).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_mel = model(mel_tensor).squeeze(0).squeeze(0).cpu().numpy()

        # Keep model correction bounded, matching training-side inference helper.
        delta = np.clip(pred_mel - mel_db, -DELTA_CLAMP_DB, DELTA_CLAMP_DB)
        mel_out = np.clip(mel_db + delta, MEL_DB_MIN, MEL_DB_MAX)

        # ใช้ phase จากสัญญาณต้นฉบับเพื่อให้ reconstruction เสถียรกว่าเดาสัญญาณใหม่ทั้งหมด
        enhanced_chunk = mel_db_to_waveform_with_input_phase(mel_out, chunk)
        if enhanced_chunk.shape[0] != chunk.shape[0]:
            if enhanced_chunk.shape[0] > chunk.shape[0]:
                enhanced_chunk = enhanced_chunk[: chunk.shape[0]]
            else:
                pad = chunk.shape[0] - enhanced_chunk.shape[0]
                enhanced_chunk = np.pad(enhanced_chunk, (0, pad))

        # match RMS กลับให้ความดังโดยรวมไม่กระโดดเมื่อเทียบกับต้นฉบับ
        enhanced_chunk = match_loudness_rms(chunk, enhanced_chunk)
        enhanced_chunk = np.clip(enhanced_chunk, -1.0, 1.0).astype(np.float32)

        chunk_weight = np.ones(chunk.shape[0], dtype=np.float32)
        fade = min(overlap_samples, chunk.shape[0] // 2)
        if fade > 0 and start > 0:
            chunk_weight[:fade] = np.linspace(0.0, 1.0, num=fade, dtype=np.float32)
        if fade > 0 and end < len(y):
            chunk_weight[-fade:] = np.minimum(
                chunk_weight[-fade:],
                np.linspace(1.0, 0.0, num=fade, dtype=np.float32),
            )

        mixed[start:end] += enhanced_chunk * chunk_weight
        weight[start:end] += chunk_weight

    # หารด้วยน้ำหนักรวมเพื่อ blend ช่วงที่ซ้อนกันกลับเป็นสัญญาณเดียว
    enhanced = np.divide(mixed, np.maximum(weight, 1e-8), out=np.zeros_like(mixed), where=weight > 0)
    sf.write(output_path, enhanced, SR)
    return output_path
