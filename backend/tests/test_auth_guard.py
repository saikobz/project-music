import unittest
from fastapi import HTTPException
from backend.utils.auth_guard import validate_tier_and_quota

class TestAuthGuard(unittest.TestCase):
    def test_free_tier_cannot_use_cnn_model(self):
        with self.assertRaises(HTTPException) as cm:
            validate_tier_and_quota(user_tier="FREE", used_quota=0, model_type="CNN")
        self.assertEqual(cm.exception.status_code, 403)
        self.assertIn("CNN model requires Basic or Pro", cm.exception.detail)

    def test_free_tier_exceeded_quota(self):
        with self.assertRaises(HTTPException) as cm:
            validate_tier_and_quota(user_tier="FREE", used_quota=3, model_type="LSTM")
        self.assertEqual(cm.exception.status_code, 403)
        self.assertIn("Monthly quota reached", cm.exception.detail)

    def test_free_tier_within_quota(self):
        # Should not raise exception when used_quota is 2 (< 3)
        validate_tier_and_quota(user_tier="FREE", used_quota=2, model_type="LSTM")

    def test_pitch_shift_range_limit_free_tier(self):
        with self.assertRaises(HTTPException) as cm:
            validate_tier_and_quota(user_tier="FREE", used_quota=0, model_type="LSTM", pitch_shift_semitones=5)
        self.assertEqual(cm.exception.status_code, 403)
        self.assertIn("Pitch shift of 5 semitones exceeds allowed limit", cm.exception.detail)

    def test_basic_tier_can_use_cnn_and_higher_pitch_shift(self):
        # Should not raise exception
        validate_tier_and_quota(user_tier="BASIC", used_quota=0, model_type="CNN", pitch_shift_semitones=5)

    def test_pro_tier_unlimited_quota(self):
        # Should not raise exception even with high used_quota and full octave pitch shift
        validate_tier_and_quota(user_tier="PRO", used_quota=999, model_type="CNN", pitch_shift_semitones=12)

if __name__ == "__main__":
    unittest.main()
