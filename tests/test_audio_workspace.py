import os
import pytest
from backend.services.audio_workspace import AudioJobWorkspace, audio_workspace

def test_audio_workspace_paths():
    workspace = AudioJobWorkspace()
    out_dir = workspace.prepare_output_dir("test_job_123", category="separated")
    assert "separated" in out_dir
    assert "test_job_123" in out_dir
    assert os.path.exists(out_dir)

def test_audio_workspace_singleton():
    assert audio_workspace is not None
