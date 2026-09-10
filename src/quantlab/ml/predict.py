"""ML predictions -> Signal (identical downstream path to rule-based signals)."""

from __future__ import annotations

import pandas as pd


def predictions_to_signal(pred_wide: pd.DataFrame) -> pd.DataFrame:
    return pred_wide
