import os
import tempfile
import unittest

import numpy as np
import soundfile as sf
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

    def test_load_lstm_checkpoint(self) -> None:
        auto_eq_inference.load_auto_eq_model.cache_clear()
        model = auto_eq_inference.load_auto_eq_model(
            "cpu",
            model_id=auto_eq_inference.AUTO_EQ_MODEL_LSTM_LAST,
        )
        self.assertTrue(hasattr(model, "lstm"))
        self.assertEqual(model.auto_eq_kind, "lstm")
        self.assertEqual(model.auto_eq_model_id, auto_eq_inference.AUTO_EQ_MODEL_LSTM_LAST)
        self.assertIn("pop", model.auto_eq_genre2id)

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

    def test_apply_lstm_auto_eq_file_preserves_shape_and_finite_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, "input.wav")
            output_path = os.path.join(temp_dir, "output.wav")

            seconds = 1
            t = np.arange(auto_eq_inference.SR * seconds, dtype=np.float32) / auto_eq_inference.SR
            stereo = np.stack(
                (
                    0.2 * np.sin(2 * np.pi * 220.0 * t),
                    0.2 * np.sin(2 * np.pi * 330.0 * t),
                ),
                axis=1,
            ).astype(np.float32)
            sf.write(input_path, stereo, auto_eq_inference.SR)

            result_path = auto_eq_inference.apply_auto_eq_file(
                input_path,
                output_path,
                genre="pop",
                delta_clamp_db=2.0,
                model_id=auto_eq_inference.AUTO_EQ_MODEL_LSTM_LAST,
            )

            processed, sr = sf.read(result_path, always_2d=True, dtype="float32")
            self.assertEqual(sr, auto_eq_inference.SR)
            self.assertEqual(processed.shape, stereo.shape)
            self.assertTrue(np.isfinite(processed).all())


if __name__ == "__main__":
    unittest.main()
