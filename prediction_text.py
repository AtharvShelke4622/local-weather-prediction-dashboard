"""
prediction_text.py
Human-readable weather insight generator
All outputs are lists of strings (frontend-safe)
"""

from typing import Dict, List
import numpy as np


# =====================================================
# HELPERS (UNCHANGED SIGNATURES)
# =====================================================
def mean(v: List[float]) -> float:
    return float(np.mean(v))


def trend(v: List[float]) -> str:
    if len(v) < 2:
        return "stable"
    d = v[-1] - v[0]
    if d > 1.0:
        return "rising"
    if d < -1.0:
        return "falling"
    return "stable"


def hour(i: int) -> str:
    return f"Hour {i + 1}"


# =====================================================
# TEMPERATURE (THEORETICAL)
# =====================================================
def describe_temperature(values: List[float]) -> List[str]:
    msgs = []

    t = trend(values)
    if t == "rising":
        msgs.append("Temperature is expected to gradually increase, indicating warming conditions.")
    elif t == "falling":
        msgs.append("Temperature is expected to decrease slightly, bringing cooler conditions.")
    else:
        msgs.append("Temperature is expected to remain fairly stable throughout the period.")

    msgs.append("Overall conditions suggest a comfortable to warm thermal environment.")
    return msgs


# =====================================================
# HUMIDITY (THEORETICAL)
# =====================================================
def describe_humidity(values: List[float]) -> List[str]:
    msgs = []

    t = trend(values)
    if t == "rising":
        msgs.append("Humidity levels are increasing, which may lead to a more humid and sticky atmosphere.")
    elif t == "falling":
        msgs.append("Humidity levels are decreasing, indicating drier and more comfortable air.")
    else:
        msgs.append("Humidity levels are expected to stay relatively constant.")

    msgs.append("Moisture conditions remain within a manageable range.")
    return msgs


# =====================================================
# WIND (THEORETICAL)
# =====================================================
def describe_wind(values: List[float]) -> List[str]:
    msgs = []

    avg = mean(values)
    if avg < 2:
        msgs.append("Wind conditions are expected to be calm with minimal air movement.")
    elif avg < 5:
        msgs.append("Moderate winds are expected, providing gentle ventilation.")
    else:
        msgs.append("Stronger winds are expected, which may affect outdoor activities.")

    msgs.append("Overall wind behavior supports stable atmospheric conditions.")
    return msgs


# =====================================================
# RADIATION (THEORETICAL)
# =====================================================
def describe_radiation(values: List[float]) -> List[str]:
    msgs = []

    peak = max(values)
    if peak < 200:
        msgs.append("Solar radiation levels remain low, suggesting cloudy or shaded conditions.")
    elif peak < 600:
        msgs.append("Moderate solar radiation is expected, indicating partial sunlight.")
    else:
        msgs.append("High solar radiation levels suggest strong sunlight exposure.")

    msgs.append("Radiation trends may influence surface temperature and comfort levels.")
    return msgs


# =====================================================
# PRECIPITATION (THEORETICAL)
# =====================================================
def describe_precipitation(values: List[float]) -> List[str]:
    msgs = []

    total = sum(values)
    if total == 0:
        msgs.append("No rainfall is expected, indicating dry weather conditions.")
    elif total < 2:
        msgs.append("Light rainfall is possible, with minimal impact on daily activities.")
    else:
        msgs.append("Noticeable rainfall is expected, which may affect outdoor plans.")

    msgs.append("Precipitation levels should be monitored for potential weather changes.")
    return msgs


# =====================================================
# PUBLIC API (UNCHANGED)
# =====================================================
def generate_prediction_text(
    preds: Dict[str, List[float]]
) -> Dict[str, List[str]]:
    return {
        "temperature": describe_temperature(preds["temperature"]),
        "humidity": describe_humidity(preds["humidity"]),
        "wind": describe_wind(preds["wind_speed"]),
        "radiation": describe_radiation(preds["radiation"]),
        "precipitation": describe_precipitation(preds["precipitation"]),
    }
