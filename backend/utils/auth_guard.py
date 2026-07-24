from fastapi import HTTPException, status

def validate_tier_and_quota(user_tier: str, used_quota: int, model_type: str = "LSTM", pitch_shift_semitones: int = 0):
    """
    Validates user tier permissions and usage quotas for music separation, AutoEQ, Compressor, and Pitch Shifting.
    Python 3.10 compatible.
    """
    tier_upper = (user_tier or "FREE").upper()
    model_upper = (model_type or "LSTM").upper()

    # 1. Model Lock Check: CNN Model is locked for Free Tier users
    if model_upper == "CNN" and tier_upper == "FREE":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="AutoEQ CNN model requires Basic or Pro subscription. Please upgrade to unlock."
        )

    # 2. Pitch Shift Range Check
    max_pitch_shifts = {
        "FREE": 2,
        "BASIC": 6,
        "PRO": 12
    }
    max_allowed = max_pitch_shifts.get(tier_upper, 2)
    if abs(pitch_shift_semitones) > max_allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Pitch shift of {pitch_shift_semitones} semitones exceeds allowed limit for {tier_upper} tier (Max ±{max_allowed})."
        )

    # 3. Quota Check (Free=3, Basic=15, Pro=-1 Unlimited)
    tier_limits = {
        "FREE": 3,
        "BASIC": 15,
        "PRO": -1
    }

    limit = tier_limits.get(tier_upper, 3)
    if limit != -1 and used_quota >= limit:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Monthly quota reached for {tier_upper} tier ({used_quota}/{limit}). Please upgrade for more processing."
        )
