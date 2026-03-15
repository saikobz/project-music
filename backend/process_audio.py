"""

- รวมงานประมวลผลเสียงพื้นฐานของระบบ เช่น แยก stem, วิเคราะห์ tempo/key/pitch และ pitch shift
- เชื่อมกับไลบรารีด้าน audio อย่าง Open-Unmix, torchaudio และ librosa
- ส่งผลลัพธ์กลับเป็นไฟล์หรือข้อมูลวิเคราะห์ให้ backend main เรียกใช้ต่อ
"""

import os
import torch
import warnings
import torchaudio
import numpy as np
import librosa
import soundfile as sf

# เลือกใช้ GPU อัตโนมัติถ้ามี 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def separate_audio(input_path: str, output_dir: str = "separated") -> str:
    # import แบบ lazy เพื่อให้ backend ยังรันส่วนอื่นได้ แม้เครื่องยังไม่ได้ลง openunmix
    try:
        from openunmix.predict import separate
        from openunmix import utils as openunmix_utils
    except ImportError as exc:
        raise RuntimeError("ต้องติดตั้ง openunmix ก่อน: pip install openunmix") from exc

    os.makedirs(output_dir, exist_ok=True)

    ext = os.path.splitext(input_path)[-1].lower()
    if ext != ".wav":
        raise ValueError("รองรับเฉพาะไฟล์ WAV (.wav)")

    try:
        # โหลดไฟล์เสียงและเตรียม separator ที่ pretrained มาแล้ว
        audio_tensor, rate = torchaudio.load(input_path)
        input_frames = int(audio_tensor.shape[-1])
        # โหลด separator ของ Open-Unmix แค่ target ที่ต้องใช้ในระบบนี้
        separator = openunmix_utils.load_separator(
            model_str_or_path="umxl",
            targets=["vocals", "drums", "bass", "other"],
            niter=1,
            residual=False,
            wiener_win_len=300,
            device=str(DEVICE),
            pretrained=True,
            filterbank="torch",
        )
        separator.freeze()
        separator = separator.to(DEVICE)
        # บาง checkpoint เก็บ sample_rate เป็น tensor จึง normalize ให้เป็น int ก่อนใช้เทียบขนาด
        separator_rate = int(
            separator.sample_rate.item()
            if isinstance(separator.sample_rate, torch.Tensor)
            else separator.sample_rate
        )
        expected_model_frames = int(round(input_frames * (separator_rate / float(rate))))

        estimates = separate(
            audio=audio_tensor.to(DEVICE),
            rate=rate,
            targets=["vocals", "drums", "bass", "other"],
            separator=separator,
            device=str(DEVICE),
        )

        # บันทึกผลลัพธ์แต่ละ target เป็นไฟล์ WAV แยกกันเพื่อใช้ดาวน์โหลดและเล่นบนหน้าเว็บ
        for target, waveform in estimates.items():
            if waveform.ndim == 3:
                waveform = waveform.squeeze(0)

            estimate_frames = int(waveform.shape[-1])
            # เดาว่า output ของโมเดลอยู่ใน sample rate ไหน เพื่อ resample กลับให้ตรงกับไฟล์ต้นฉบับ
            if abs(estimate_frames - expected_model_frames) <= 8:
                estimate_rate = separator_rate
            elif abs(estimate_frames - input_frames) <= 8:
                estimate_rate = rate
            else:
                estimate_rate = separator_rate

            if estimate_rate != rate:
                waveform = torchaudio.functional.resample(
                    waveform,
                    orig_freq=estimate_rate,
                    new_freq=rate,
                )

            torchaudio.save(
                os.path.join(output_dir, f"{target}.wav"),
                waveform.cpu(),
                sample_rate=rate,
            )

        print("แยกเสียงเสร็จแล้ว:", output_dir)
        return output_dir

    except Exception as e:
        print(f"[ERROR] separate_audio: {e}")
        raise


def analyze_audio(input_path: str) -> dict:
    """วิเคราะห์ไฟล์เสียงแล้วคืนค่า tempo, pitch และ key"""
    try:
        # แปลงเป็น mono ก่อน เพื่อให้การวิเคราะห์ภาพรวมของเพลงง่ายและสม่ำเสมอ
        y, sr = librosa.load(input_path, sr=None, mono=True)

        # ใช้ beat.tempo (ใน librosa 0.11 ยังมี alias) และละเว้น FutureWarning
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            tempo = float(librosa.beat.tempo(y=y, sr=sr)[0])

        f0 = librosa.yin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"))
        f0 = f0[~np.isnan(f0)]
        pitch_note = None
        if f0.size:
            # ใช้ค่ากลางของ pitch เพื่อลดผลกระทบจากโน้ตสั้นหรือเสียงรบกวนบางช่วง
            pitch_hz = float(np.median(f0))
            pitch_note = librosa.hz_to_note(pitch_hz)

        # เปรียบเทียบ chroma ของเพลงกับ profile major/minor เพื่อเดาว่าเพลงอยู่คีย์ไหน
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = chroma.mean(axis=1)

        maj_profile = np.array(
            [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
        )
        min_profile = np.array(
            [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
        )
        keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

        maj_scores = [np.correlate(chroma_mean, np.roll(maj_profile, i))[0] for i in range(12)]
        min_scores = [np.correlate(chroma_mean, np.roll(min_profile, i))[0] for i in range(12)]

        maj_idx = int(np.argmax(maj_scores))
        min_idx = int(np.argmax(min_scores))

        # เลือก key จาก profile ที่ได้คะแนน correlation สูงกว่า
        if maj_scores[maj_idx] >= min_scores[min_idx]:
            key = f"{keys[maj_idx]} major"
        else:
            key = f"{keys[min_idx]} minor"

        return {"tempo": tempo, "pitch": pitch_note, "key": key}
    except Exception as e:
        print(f"[ERROR] analyze_audio: {e}")
        raise


def pitch_shift_audio(input_path: str, steps: float, output_path: str) -> str:
    """เลื่อน pitch ของไฟล์เสียงตามจำนวน half-steps ที่ระบุ"""
    try:
        # ใช้ librosa ทำ pitch shifting แล้วเขียนผลลัพธ์กลับเป็นไฟล์ใหม่
        y, sr = librosa.load(input_path, sr=None)
        shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
        sf.write(output_path, shifted, sr)
        return output_path
    except Exception as e:
        print(f"[ERROR] pitch_shift_audio: {e}")
        raise
