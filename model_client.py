import httpx
from datetime import datetime, timedelta, timezone
from typing import List
from app_config import settings

TARGETS = ["temperature", "humidity", "wind_speed", "radiation", "precipitation"]

async def predict_8h(device_id: str, recent_window: List[List[float]]):
    if len(recent_window) != 24:
        raise ValueError("recent_window must contain exactly 24 rows")

    url = f"{settings.MODEL_SERVER_URL}/model/predict"
    payload = {
        "device_id": device_id,
        "recent_window": recent_window,
    }

    async with httpx.AsyncClient(timeout=15) as client:
        try:
            r = await client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()

            preds = data["predictions_8h"]
            model_version = data["model_version"]

        except Exception as e:
            # HARD fallback (model server unreachable)
            last = recent_window[-1]
            preds = {
                t: [float(last[i])] * 8
                for i, t in enumerate(TARGETS)
            }
            model_version = "fallback"

    now = datetime.now(timezone.utc)
    for_ts = [
        (now + timedelta(hours=i + 1)).isoformat()
        for i in range(8)
    ]

    return for_ts, preds, model_version
