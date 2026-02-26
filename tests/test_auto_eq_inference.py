import os
import tempfile
import unittest

import numpy as np
import torch

import backend.auto_eq_inference as auto_eq_inference


class TestAutoEqInference(unittest.TestCase):
    def tearDown(self) -> None:
        auto_eq_inference.load_auto_eq_model.cache_clear()

    def test_load_current_checkpoint_infers_expected_channels(self) -> None:
        auto_eq_inference.load_auto_eq_model.cache_clear()
        model = auto_eq_inference.load_auto_eq_model("cpu")
        self.assertEqual(model.body[0].out_channels, 32)
        self.assertEqual(model.body[3].out_channels, 64)
        self.assertEqual(model.body[6].out_channels, 128)

    def test_legacy_net_prefix_checkpoint_loads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "legacy.pt")
            source_model = auto_eq_inference.AutoEQCNN(ch1=4, ch2=8, ch3=16)
            legacy_state = {
                key.replace("body.", "net."): value.clone()
                for key, value in source_model.state_dict().items()
            }
            torch.save(legacy_state, checkpoint_path)

            original_path = auto_eq_inference.MODEL_PATH
            auto_eq_inference.MODEL_PATH = checkpoint_path
            try:
                auto_eq_inference.load_auto_eq_model.cache_clear()
                loaded_model = auto_eq_inference.load_auto_eq_model("cpu")
                self.assertEqual(loaded_model.body[0].out_channels, 4)
                self.assertEqual(loaded_model.body[3].out_channels, 8)
                self.assertEqual(loaded_model.body[6].out_channels, 16)
            finally:
                auto_eq_inference.MODEL_PATH = original_path
                auto_eq_inference.load_auto_eq_model.cache_clear()

    def test_malformed_checkpoint_raises_specific_error(self) -> None:
        bad_state = {"body.0.weight": torch.randn(4, 1, 3, 3)}
        with self.assertRaises(auto_eq_inference.AutoEQModelLoadError):
            auto_eq_inference.infer_arch_from_state_dict(bad_state)

    def test_phase_reconstruction_preserves_length_and_finite_values(self) -> None:
        seconds = 1
        t = np.arange(auto_eq_inference.SR * seconds, dtype=np.float32) / auto_eq_inference.SR
        y = (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
        mel = auto_eq_inference.waveform_to_mel_db(y)
        reconstructed = auto_eq_inference.mel_db_to_waveform_with_input_phase(mel, y)
        self.assertEqual(len(reconstructed), len(y))
        self.assertTrue(np.isfinite(reconstructed).all())


if __name__ == "__main__":
    unittest.main()
