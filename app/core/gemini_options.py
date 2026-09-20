"""Gemini 呼叫的共用成本控制選項。

output token 的單價遠高於 input（`gemini-3.5-flash-lite` 為 8.3 倍），且 thinking token
併入 output 計價，因此這兩項是主要的成本槓桿。本模組集中處理選用的 thinking 設定，
避免 llm_service 與 market_report_service 各寫一套。
"""

from typing import Any, Dict

from google.genai import types

from app.core.config import settings
from app.core.logging import logger


def thinking_kwargs() -> Dict[str, Any]:
    """依設定回傳可展開至 `GenerateContentConfig` 的 thinking 參數。

    回傳空字典代表不傳送此參數、沿用模型自身預設（`gemini-3.5-flash-lite` 預設為 Off）。
    刻意採「預設不傳送」策略：並非所有模型世代都接受 `thinking_level`，
    無條件傳送會讓不支援的模型整批呼叫失敗。
    """
    level = (settings.GEMINI_THINKING_LEVEL or "").strip().upper()
    if not level:
        return {}

    valid = {e.value for e in types.ThinkingLevel}
    if level not in valid:
        logger.warning(
            f"[GeminiOptions] GEMINI_THINKING_LEVEL='{level}' 不是合法值 {sorted(valid)}，已忽略。"
        )
        return {}

    return {"thinking_config": types.ThinkingConfig(thinking_level=level)}
