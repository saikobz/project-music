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
            validate_tier_and_quota(user_tier="FREE", used_quota=1, model_type="LSTM")
        self.assertEqual(cm.exception.status_code, 403)
        self.assertIn("Monthly quota reached", cm.exception.detail)

    def test_basic_tier_can_use_cnn_model(self):
        # Should not raise exception
        validate_tier_and_quota(user_tier="BASIC", used_quota=0, model_type="CNN")

    def test_pro_tier_unlimited_quota(self):
        # Should not raise exception even with high used_quota
        validate_tier_and_quota(user_tier="PRO", used_quota=999, model_type="CNN")

if __name__ == "__main__":
    unittest.main()
