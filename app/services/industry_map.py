"""持股代號 → 產業分類對照表。

產業報告以「使用者持股所屬產業」為單位，因此需要把代號歸到產業。
優先查本表（中文、與新聞用語一致）；查不到才由 `IndustryReportService`
以 yfinance 的 sector 補位，仍無則歸入 `UNKNOWN_INDUSTRY`。
新增持股若產業分類不準，直接在此補一行即可。
"""

from typing import Dict, Optional

UNKNOWN_INDUSTRY = "其他"

TICKER_INDUSTRY: Dict[str, str] = {
    # 台股
    "2330": "半導體",
    "2454": "半導體",
    "2382": "AI 伺服器",
    "2317": "電子代工",
    "6442": "光通訊",
    "2313": "PCB 電路板",
    "3008": "光學鏡頭",
    "2881": "金融",
    "2882": "金融",
    # 美股
    "NVDA": "AI 晶片",
    "TSLA": "電動車",
    "AAPL": "消費電子",
    "MSFT": "雲端軟體",
    "GOOGL": "網路與雲端",
    "SPCX": "航太與國防",
}

# yfinance sector (英文) → 中文產業名；僅用於本表查不到時
SECTOR_ZH: Dict[str, str] = {
    "Technology": "科技",
    "Financial Services": "金融",
    "Healthcare": "醫療保健",
    "Consumer Cyclical": "非必需消費",
    "Consumer Defensive": "必需消費",
    "Industrials": "工業",
    "Energy": "能源",
    "Utilities": "公用事業",
    "Real Estate": "不動產",
    "Basic Materials": "原物料",
    "Communication Services": "通訊服務",
}


def lookup_industry(ticker: str) -> Optional[str]:
    """查本表；以大寫代號並去除 .TW/.TWO 後綴比對。"""
    key = ticker.strip().upper().split(".")[0]
    return TICKER_INDUSTRY.get(key)
