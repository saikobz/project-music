from fastapi import HTTPException, status

def validate_tier_and_quota(user_tier: str, used_quota: int, model_type: str = "LSTM"):
    """
    Validates user tier permissions and usage quotas for music separation and AutoEQ.
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

    # 2. Quota Check (Free=1, Basic=15, Pro=-1 Unlimited)
    tier_limits = {
        "FREE": 1,
        "BASIC": 15,
        "PRO": -1
    }

    limit = tier_limits.get(tier_upper, 1)
    if limit != -1 and used_quota >= limit:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Monthly quota reached for {tier_upper} tier ({used_quota}/{limit}). Please upgrade for more processing."
        )
