# Refactoring Plan: UploadBox & Audio Workspace Architecture

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the monolithic `UploadBox.tsx` (986 lines) into modular frontend components with a dedicated state hook, and encapsulate backend disk/job operations into a deep `AudioJobWorkspace` module.

**Architecture:** 
- Frontend: Extract processing state & API calls into `useAudioProcessor.ts` hook. Split `UploadBox.tsx` into `AudioIngestionBox`, `AudioToolSelector`, and `AudioResultView`.
- Backend: Unify `storage.py`, `job_manager.py`, and manual file operations in routers into a deep `AudioJobWorkspace` service in `backend/services/audio_workspace.py`.

**Tech Stack:** React 19, TypeScript, Next.js App Router, FastAPI, Python 3.10, PyTorch, Librosa.

## Global Constraints

- Preserve all existing API endpoints and responses without breaking frontend contract.
- Keep Python 3.10 compatibility (no Python 3.11+ features).
- Keep WaveSurfer.js instances cleaned up to prevent memory leaks.
- Ensure automated tests in `tests/` pass after each refactoring task.

---

### Task 1: Create Backend `AudioJobWorkspace` Deep Module

**Files:**
- Create: `backend/services/audio_workspace.py`
- Modify: `backend/routers/stems.py`
- Modify: `backend/routers/audio_ops.py`
- Modify: `backend/main.py`
- Test: `tests/test_audio_workspace.py`

**Interfaces:**
- Produces: `AudioJobWorkspace` class with methods:
  - `save_upload(file: UploadFile, trim_start: float | None, trim_end: float | None) -> tuple[str, str]`
  - `get_job_output_dir(job_id: str) -> str`
  - `create_stem_zip(job_id: str, format_mp3: bool) -> str`
  - `cleanup_expired(ttl_seconds: int)`

- [ ] **Step 1: Write test for `AudioJobWorkspace`**

```python
# tests/test_audio_workspace.py
import os
import pytest
from backend.services.audio_workspace import AudioJobWorkspace

def test_audio_workspace_paths():
    workspace = AudioJobWorkspace()
    out_dir = workspace.get_job_output_dir("test_job_123")
    assert "separated" in out_dir
    assert "test_job_123" in out_dir
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_audio_workspace.py -v`
Expected: FAIL with ModuleNotFoundError or AttributeError.

- [ ] **Step 3: Implement `AudioJobWorkspace` module**

```python
# backend/services/audio_workspace.py
import os
import shutil
import zipfile
import asyncio
import logging
import soundfile as sf
import numpy as np
from fastapi import UploadFile
from backend.services.storage import save_upload, convert_to_mp3, UPLOAD_DIR
from backend.services.job_manager import job_manager

logger = logging.getLogger(__name__)

class AudioJobWorkspace:
    def __init__(self, upload_dir: str = UPLOAD_DIR):
        self.upload_dir = upload_dir

    async def save_upload_file(self, file: UploadFile, trim_start: float | None = None, trim_end: float | None = None) -> tuple[str, str]:
        return await save_upload(file, trim_start=trim_start, trim_end=trim_end)

    def get_job_output_dir(self, file_id: str, category: str = "separated") -> str:
        out_dir = os.path.join(category, os.path.basename(file_id))
        os.makedirs(out_dir, exist_ok=True)
        job_manager.register_job(file_id, out_dir)
        return out_dir

    async def prepare_zip(self, file_id: str, output_dir: str, export_format: str = "wav") -> str:
        job_manager.complete_job(file_id)
        if export_format == "mp3":
            for root, _, files in os.walk(output_dir):
                for name in files:
                    if name.lower().endswith(".wav"):
                        wav_path = os.path.join(root, name)
                        await asyncio.to_thread(convert_to_mp3, wav_path)

        zip_filename = f"{os.path.basename(file_id)}_separated.zip"
        zip_path = os.path.join(self.upload_dir, zip_filename)
        with zipfile.ZipFile(zip_path, "w") as zipf:
            for root, _, files in os.walk(output_dir):
                for name in files:
                    file_path = os.path.join(root, name)
                    arcname = os.path.relpath(file_path, output_dir)
                    zipf.write(file_path, arcname)
        return zip_path

    def cleanup_expired_files(self, ttl_seconds: int):
        import time
        now = time.time()
        dirs_to_clean = [self.upload_dir, "separated", "eq_applied", "compressed"]
        for d in dirs_to_clean:
            if not os.path.exists(d):
                continue
            for item in os.listdir(d):
                item_path = os.path.join(d, item)
                try:
                    if os.path.isfile(item_path):
                        if now - os.path.getmtime(item_path) > ttl_seconds:
                            os.remove(item_path)
                    elif os.path.isdir(item_path):
                        if now - os.path.getmtime(item_path) > ttl_seconds:
                            shutil.rmtree(item_path, ignore_errors=True)
                except Exception as e:
                    logger.warning(f"Cleanup error for {item_path}: {e}")

workspace = AudioJobWorkspace()
```

- [ ] **Step 4: Refactor `stems.py` and `audio_ops.py` to use `workspace`**

Replace manual `os.makedirs`, `zipfile`, and cleanup code in `routers/stems.py` and `routers/audio_ops.py` with calls to `workspace`.

- [ ] **Step 5: Run pytest verification**

Run: `pytest tests/ -v`
Expected: PASS

---

### Task 2: Extract Frontend Custom Hook `useAudioProcessor`

**Files:**
- Create: `app/components/studio/useAudioProcessor.ts`
- Modify: `app/components/UploadBox.tsx`

**Interfaces:**
- Produces: `useAudioProcessor()` hook returning:
  - State: `file`, `trimStart`, `trimEnd`, `selectedAction`, `isProcessing`, `progress`, `resultFileId`, `stemUrls`, `analysisData`, `masteringResult`
  - Actions: `setFile`, `setTrimStart`, `setTrimEnd`, `setSelectedAction`, `processAudio`, `resetWorkspace`

- [ ] **Step 1: Create `useAudioProcessor.ts` custom hook**

Move processing state, Axios network calls, progress interval simulation, and toast notifications into `useAudioProcessor.ts`.

- [ ] **Step 2: Verify type check**

Run: `npx tsc --noEmit`
Expected: No compilation errors.

---

### Task 3: Split `UploadBox.tsx` into Modular UI Components

**Files:**
- Create: `app/components/studio/AudioIngestionBox.tsx`
- Create: `app/components/studio/AudioToolSelector.tsx`
- Create: `app/components/studio/AudioResultView.tsx`
- Refactor: `app/components/UploadBox.tsx`

- [ ] **Step 1: Extract `AudioIngestionBox.tsx`**

Move drag & drop zone, file info badge, trim slider controls into `AudioIngestionBox.tsx`.

- [ ] **Step 2: Extract `AudioToolSelector.tsx`**

Move tab selection (Separate, Auto-EQ, Compressor, Pitch Shift) and parameter setting sliders into `AudioToolSelector.tsx`.

- [ ] **Step 3: Extract `AudioResultView.tsx`**

Move skeleton loading indicators, `MultiStemLivePlayer`, `WaveformPlayer`, `AudioAnalysis`, and Export modal triggers into `AudioResultView.tsx`.

- [ ] **Step 4: Update `UploadBox.tsx` to orchestrate sub-components**

Re-wire `UploadBox.tsx` to bring together `useAudioProcessor`, `AudioIngestionBox`, `AudioToolSelector`, and `AudioResultView`.

- [ ] **Step 5: Type-check and Lint verification**

Run: `npx tsc --noEmit`
Expected: Clean pass with 0 errors.

---

### Task 4: End-to-End Verification & Walkthrough

- Run full test suite: `pytest tests/` and `npm run type-check`
- Verify studio page loads cleanly and audio workflow operations function as expected.
