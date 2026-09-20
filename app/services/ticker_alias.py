"""股票代號別名對照表（單一資料來源）。

`nlu_rules`（確定性解析）與 `slm_service`（選用的 SLM fallback）共用此表，
避免兩邊各維護一份而逐漸失去同步。
"""

from typing import Dict, Optional

TICKER_ALIASES: Dict[str, str] = {
    # 台股
    "台積電": "2330",
    "台積": "2330",
    "tsmc": "2330",
    "聯發科": "2454",
    "發哥": "2454",
    "廣達": "2382",
    "光聖": "6442",
    "華通": "2313",
    "鴻海": "2317",
    "大立光": "3008",
    "富邦金": "2881",
    "國泰金": "2882",
    # 美股
    "輝達": "NVDA",
    "英偉達": "NVDA",
    "nvidia": "NVDA",
    "特斯拉": "TSLA",
    "tesla": "TSLA",
    "蘋果": "AAPL",
    "apple": "AAPL",
    "微軟": "MSFT",
    "谷哥": "GOOGL",
    "谷歌": "GOOGL",
    "google": "GOOGL",
}


def resolve_alias(text: Optional[str]) -> Optional[str]:
    """將公司名稱或代號正規化為股票代號。

    比對順序：完全相符 → 子字串相符（長別名優先，避免「台積」先吃掉「台積電」）。
    都沒命中時回傳原字串的大寫形式（可能本身就是代號）。
    """
    if not text:
        return None
    cleaned = text.strip()
    if not cleaned:
        return None

    lowered = cleaned.lower()
    if lowered in TICKER_ALIASES:
        return TICKER_ALIASES[lowered]
    if cleaned in TICKER_ALIASES:
        return TICKER_ALIASES[cleaned]

    for alias in sorted(TICKER_ALIASES, key=len, reverse=True):
        if alias in cleaned or alias in lowered:
            return TICKER_ALIASES[alias]

    return cleaned.upper()
