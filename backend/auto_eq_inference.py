import os
from functools import lru_cache

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from functools import lru_cache

# พารามิเตอร์เหมือนตอนเทรน
SR = 44100
N_FFT = 2048
HOP = 512
N_MELS = 128
SEGMENT_SECONDS = 5  # โมเดลเทรนด้วย segment 5 วินาที
N_ITER = 16  # รอบ griffin-lim ตามสคริปต์เทรน (ลดได้ถ้าต้องการเร็วกว่า)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "autoeq_cnn_v1.pt")


class AutoEQCNN(nn.Module):
    """
    สถาปัตยกรรมเดียวกับตอนเทรน (CNN residual บน Mel-spectrogram)
    """

    def __init__(self):
        super().__init__()
        # โครงสร้างตาม auto_eq.py ล่าสุด (channels = 16 ตลอด)
        self.body = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    return mel_db.astype(np.float32)


@lru_cache(maxsize=1)
def _mel_pinv():
    mel_basis = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS).astype(np.float32)
    inv_mel = np.linalg.pinv(mel_basis).astype(np.float32)
    return inv_mel


def mel_db_to_waveform(mel_db: np.ndarray) -> np.ndarray:
    """
    แปลง mel (dB) -> power -> inverse mel (pseudoinverse) -> magnitude -> griffin-lim
    ใช้ pseudoinverse แทน nnls เพื่อหลีกเลี่ยงปัญหา memory จาก scipy.optimize
    """
    mel_power = librosa.db_to_power(mel_db).astype(np.float32)  # (n_mels, T)
    inv_mel = _mel_pinv()  # (n_fft//2+1, n_mels)
    linear_power = np.dot(inv_mel, mel_power)  # (n_fft//2+1, T)
    linear_power = np.maximum(linear_power, 0.0)
    linear_mag = np.sqrt(linear_power, dtype=np.float32)
    audio = librosa.griffinlim(linear_mag, hop_length=HOP, n_iter=N_ITER)
    return audio.astype(np.float32)


def match_loudness_rms(y_ref: np.ndarray, y_out: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    ref_rms = np.sqrt(np.mean(y_ref**2)) + eps
    out_rms = np.sqrt(np.mean(y_out**2)) + eps
    gain = ref_rms / out_rms
    return y_out * gain


@lru_cache(maxsize=1)
def load_auto_eq_model(device: str = "cpu") -> AutoEQCNN:
    model = AutoEQCNN()
    state = torch.load(MODEL_PATH, map_location=device)
    # รองรับ state_dict ทั้งที่ใช้ body.* (ตามไฟล์ใหม่) หรือ net.* (ไฟล์เก่า)
    new_state = {}
    for k, v in state.items():
        if k.startswith("net."):
            new_key = k.replace("net.", "body.")
        else:
            new_key = k
        new_state[new_key] = v
    model.load_state_dict(new_state, strict=False)
    model.to(device)
    model.eval()
    return model


def apply_auto_eq_file(input_path: str, output_path: str) -> str:
    """
    โหลดไฟล์ -> แปลงเป็น mel dB -> โมเดลปรับ EQ -> กลับเป็น waveform -> บันทึก
    ทำงานแบบ chunk 5 วินาทีเพื่อไม่กินหน่วยความจำ
    """
    y, _ = librosa.load(input_path, sr=SR, mono=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_auto_eq_model(device)

    segment_samples = SEGMENT_SECONDS * SR
    outputs: list[np.ndarray] = []

    for start in range(0, len(y), segment_samples):
        end = min(start + segment_samples, len(y))
        chunk = y[start:end]
        mel_db = waveform_to_mel_db(chunk)

        mel_tensor = torch.from_numpy(mel_db).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_mel = model(mel_tensor).squeeze(0).squeeze(0).cpu().numpy()

        enhanced_chunk = mel_db_to_waveform(pred_mel)
        # match loudness ให้ใกล้ต้นฉบับ
        enhanced_chunk = match_loudness_rms(chunk, enhanced_chunk)
        outputs.append(enhanced_chunk)

    enhanced = np.concatenate(outputs) if outputs else np.array([], dtype=np.float32)
    sf.write(output_path, enhanced, SR)
    return output_path
