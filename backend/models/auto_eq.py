# -*- coding: utf-8 -*-
"""
Auto-EQ (Colab Free Safe, keep max_segments=20)

Fixes vs original:
- No more stacking waveform dataset in RAM
- Convert to mel on-the-fly during dataset build
- Store mel as float16 + np.savez_compressed
- Training uses batch_size=2 + AMP to reduce VRAM
"""

from google.colab import drive
drive.mount('/content/drive')

import os
import numpy as np
import librosa

# -----------------------------
# 0) Quick sanity check folders
# -----------------------------
BASE_DATA_DIR = "/content/drive/MyDrive/project-music/data"
print("BASE_DATA_DIR:", BASE_DATA_DIR)
print("ภายใน data มีอะไรบ้าง:")
print(os.listdir(BASE_DATA_DIR))

print("\nภายใน mix_tracks:")
print(os.listdir(os.path.join(BASE_DATA_DIR, "mix_tracks")))

# -----------------------------
# 1) Paths & global settings
# -----------------------------
BASE_DIR    = "/content/drive/MyDrive/project-music"
RAW_ROOT    = os.path.join(BASE_DIR, "data/raw_tracks")
MIX_ROOT    = os.path.join(BASE_DIR, "data/mix_tracks")
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

GENRES = ["trap", "pop", "rock", "soul", "country"]

SAMPLE_RATE      = 44100
SEGMENT_SECONDS  = 5
SEGMENT_SAMPLES  = SAMPLE_RATE * SEGMENT_SECONDS
MAX_SEGMENTS     = 20  

# Mel settings
sr     = SAMPLE_RATE
n_fft  = 2048
hop    = 512
n_mels = 128

# -----------------------------
# 2) Helpers
# -----------------------------
def load_segments_pair(raw_path, mix_path, max_segments=MAX_SEGMENTS):
    """
    Load both files fully (as original), then slice into equal-length segments.
    Note: This loads whole track into RAM per song. It's OK if songs are not huge.
    If songs are very long, switch to offset/duration streaming later.
    """
    raw_y, sr1 = librosa.load(raw_path, sr=SAMPLE_RATE, mono=True)
    mix_y, sr2 = librosa.load(mix_path, sr=SAMPLE_RATE, mono=True)
    assert sr1 == sr2 == SAMPLE_RATE

    min_len = min(len(raw_y), len(mix_y))
    raw_y = raw_y[:min_len]
    mix_y = mix_y[:min_len]

    if min_len < SEGMENT_SAMPLES:
        return [], []

    segments_raw, segments_mix = [], []
    num_segments = min(max_segments, min_len // SEGMENT_SAMPLES)

    for i in range(num_segments):
        start = i * SEGMENT_SAMPLES
        end   = start + SEGMENT_SAMPLES
        segments_raw.append(raw_y[start:end])
        segments_mix.append(mix_y[start:end])

    return segments_raw, segments_mix

def to_mel_db(wave: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=wave, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels, power=2.0
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.astype(np.float32)

# -----------------------------
# 3) Build MEL dataset (Colab Free safe)
#    - Convert mel immediately
#    - Save float16 + compressed
# -----------------------------
OUT_PATH = os.path.join(DATASET_DIR, "autoeq_mel_rawmix_v1_colabfree_fp16.npz")

X_mel_list = []
Y_mel_list = []
genre_labels = []

total_pairs = 0
total_segs  = 0

for genre in GENRES:
    raw_folder = os.path.join(RAW_ROOT, genre)
    mix_folder = os.path.join(MIX_ROOT, genre)

    raw_files = [f for f in os.listdir(raw_folder) if f.endswith(".wav")]
    mix_files = [f for f in os.listdir(mix_folder) if f.endswith(".wav")]

    common = sorted(list(set(raw_files) & set(mix_files)))
    print(f"\nGenre {genre}: มีคู่ไฟล์ {len(common)} เพลง")
    total_pairs += len(common)

    for fname in common:
        raw_path = os.path.join(raw_folder, fname)
        mix_path = os.path.join(mix_folder, fname)

        raw_segs, mix_segs = load_segments_pair(raw_path, mix_path, max_segments=MAX_SEGMENTS)

        for r_seg, m_seg in zip(raw_segs, mix_segs):
            # Convert mel NOW (do not store waveform dataset)
            r_mel = to_mel_db(r_seg)
            m_mel = to_mel_db(m_seg)

            # Store as float16 to halve RAM + disk
            X_mel_list.append(r_mel.astype(np.float16))
            Y_mel_list.append(m_mel.astype(np.float16))
            genre_labels.append(genre)
            total_segs += 1

print("\nTotal pairs:", total_pairs)
print("Total segments:", total_segs)

# Stack once
X_mel = np.stack(X_mel_list, axis=0)  # (N, 128, T)
Y_mel = np.stack(Y_mel_list, axis=0)
genre_labels = np.array(genre_labels)

print("X_mel shape:", X_mel.shape, "dtype:", X_mel.dtype)
print("Y_mel shape:", Y_mel.shape, "dtype:", Y_mel.dtype)
print("labels sample:", genre_labels[:10])

# Save compressed
np.savez_compressed(OUT_PATH, X=X_mel, Y=Y_mel, genre=genre_labels)
print("Saved mel dataset to:", OUT_PATH)

# Free big lists ASAP (reduce RAM)
del X_mel_list, Y_mel_list

# -----------------------------
# 4) Dataset & DataLoader
# -----------------------------
import torch
from torch.utils.data import Dataset, DataLoader

class AutoEQDataset(Dataset):
    """
    Loads fp16 mel from NPZ, converts to float32 tensors per item.
    """
    def __init__(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)
        self.X = data["X"]  # fp16
        self.Y = data["Y"]  # fp16

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx].astype(np.float32)).unsqueeze(0)  # (1,128,T)
        y = torch.from_numpy(self.Y[idx].astype(np.float32)).unsqueeze(0)
        return x, y

npz_path = OUT_PATH
ds = AutoEQDataset(npz_path)

# Colab Free: safer VRAM
dl = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0, pin_memory=True)

x_batch, y_batch = next(iter(dl))
print("x_batch:", x_batch.shape)
print("y_batch:", y_batch.shape)

# -----------------------------
# 5) Model
# -----------------------------
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class AutoEQCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 1, kernel_size=1),
        )

    def forward(self, x):
        return x + self.body(x)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

model = AutoEQCNN().to(device)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4)

# AMP helps VRAM a lot on Colab
use_amp = (device == "cuda")
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
eps = 1e-8

EPOCHS = 60

baseline_loss = None
baseline_sdr  = None

torch.cuda.empty_cache() if device == "cuda" else None

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    running_sdr  = 0.0
    n_batches    = 0

    loop = tqdm(dl, leave=False)
    for x, y in loop:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            pred = model(x)
            loss = loss_fn(pred, y)

        # backward
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # SDR (avoid -inf)
        num = (y ** 2).mean(dim=(1, 2, 3))
        den = ((y - pred) ** 2).mean(dim=(1, 2, 3)) + eps
        ratio = torch.clamp(num / den, min=1e-8)
        sdr_batch = 10.0 * torch.log10(ratio)
        sdr_value = sdr_batch.mean().item()

        running_loss += loss.item()
        running_sdr  += sdr_value
        n_batches += 1

        loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
        loop.set_postfix(loss=loss.item(), sdr=sdr_value)

    if n_batches == 0:
        print(f"Epoch {epoch+1}: no batches, skip")
        continue

    avg_loss = running_loss / n_batches
    avg_sdr  = running_sdr  / n_batches

    if epoch == 0:
        baseline_loss = avg_loss
        baseline_sdr  = avg_sdr
        loss_improve_pct = 0.0
        sdr_improve_db   = 0.0
        print(f"\n[Baseline set at epoch 1] loss={baseline_loss:.4f}, SDR={baseline_sdr:.2f} dB\n")
    else:
        loss_improve_pct = (baseline_loss - avg_loss) / baseline_loss * 100.0
        sdr_improve_db   = avg_sdr - baseline_sdr

    print(
        f"Epoch {epoch+1:02d}/{EPOCHS} | "
        f"Loss: {avg_loss:.4f} | "
        f"Improvement: {loss_improve_pct:+.2f}% | "
        f"SDR: {avg_sdr:.2f} dB | "
        f"ΔSDR: {sdr_improve_db:+.2f} dB"
    )

# -----------------------------
# 6) Save model
# -----------------------------
MODEL_PATH = os.path.join(MODEL_DIR, "autoeq_cnn_v1_colabfree.pt")

model_cpu = AutoEQCNN()
model_cpu.load_state_dict(model.state_dict())
torch.save(model_cpu.state_dict(), MODEL_PATH)

print("Saved model to:", MODEL_PATH)

# -----------------------------
# 7) Inference helpers (same idea as original)
# -----------------------------
import soundfile as sf

def waveform_to_mel_db(y: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels, power=2.0
    )
    mel_db = librosa.power_to_db(mel)
    mel_db = np.clip(mel_db, -80.0, 0.0)
    return mel_db.astype(np.float32)

def mel_db_to_waveform_with_input_phase(mel_db: np.ndarray, y_ref: np.ndarray) -> np.ndarray:
    mel = librosa.db_to_power(mel_db)

    mag_pred = librosa.feature.inverse.mel_to_stft(
        mel, sr=sr, n_fft=n_fft
    )

    stft_ref = librosa.stft(y_ref, n_fft=n_fft, hop_length=hop)

    T = min(mag_pred.shape[1], stft_ref.shape[1])
    mag_pred = mag_pred[:, :T]
    stft_ref = stft_ref[:, :T]

    phase = np.angle(stft_ref)
    stft_new = mag_pred * np.exp(1j * phase)

    y = librosa.istft(stft_new, hop_length=hop)
    return y.astype(np.float32)

def match_loudness_rms(y_ref: np.ndarray, y_out: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    ref_rms = np.sqrt(np.mean(y_ref**2)) + eps
    out_rms = np.sqrt(np.mean(y_out**2)) + eps
    return y_out * (ref_rms / out_rms)

def match_loudness_peak(y_ref: np.ndarray, y_out: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    ref_peak = np.max(np.abs(y_ref)) + eps
    out_peak = np.max(np.abs(y_out)) + eps
    return y_out * (ref_peak / out_peak)

def enhance_waveform(
    y_raw: np.ndarray,
    model: torch.nn.Module,
    device: str = "cuda",
    loudness_mode: str = "rms",
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        mel_raw = waveform_to_mel_db(y_raw)  # (128, T)
        mel_tensor = torch.from_numpy(mel_raw).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,128,T)

        mel_pred = model(mel_tensor).detach().cpu().squeeze(0).squeeze(0).numpy()

    y_enh = mel_db_to_waveform_with_input_phase(mel_pred, y_ref=y_raw)

    if loudness_mode == "peak":
        y_enh = match_loudness_peak(y_raw, y_enh)
    else:
        y_enh = match_loudness_rms(y_raw, y_enh)

    y_enh = np.clip(y_enh, -1.0, 1.0)
    return y_enh

def split_into_blocks(y: np.ndarray, block_sec: float = 5.0, sr: int = 44100):
    block_len = int(block_sec * sr)
    blocks = []
    for start in range(0, len(y), block_len):
        end = start + block_len
        b = y[start:end]
        if len(b) > 0:
            blocks.append(b)
    return blocks

def enhance_long_audio(
    y_raw: np.ndarray,
    model: torch.nn.Module,
    device: str = "cuda",
    block_sec: float = 5.0,
    loudness_mode: str = "rms",
) -> np.ndarray:
    blocks = split_into_blocks(y_raw, block_sec=block_sec, sr=sr)
    out_blocks = []
    for b in blocks:
        out_blocks.append(enhance_waveform(b, model=model, device=device, loudness_mode=loudness_mode))
    return np.concatenate(out_blocks) if len(out_blocks) else y_raw