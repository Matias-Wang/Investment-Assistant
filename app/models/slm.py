from enum import Enum
from typing import Optional, List, Union, Any
from pydantic import BaseModel, Field


class NLUIntent(str, Enum):
    RECORD_BUY = "RECORD_BUY"            # 記帳買入，如「剛用1050買入台積電8股」
    RECORD_SELL = "RECORD_SELL"          # 記帳賣出，如「賣出台積電4股」
    QUERY_PORTFOLIO = "QUERY_PORTFOLIO"  # 查詢庫存與損益，如「看一下我賺多少」、「查詢庫存」
    QUERY_CONFIG = "QUERY_CONFIG"        # 查詢策略參數，如「目前設定為何」
    UPDATE_CONFIG = "UPDATE_CONFIG"      # 修改策略參數，如「把預算改成12萬」、「停損改-30%」
    MARKET_REPORT = "MARKET_REPORT"      # 今日市場快報，如「今天大盤如何」、「市場報告」
    BUY_SCAN = "BUY_SCAN"                # 觸發買入掃描，如「有沒有推薦買進的股票」
    SELL_CHECK = "SELL_CHECK"            # 觸發賣出風控檢查，如「檢查哪些該賣了」
    MENU = "MENU"                        # 開啟選單
    HELP = "HELP"                        # 查詢說明
    UNKNOWN = "UNKNOWN"                  # 無法識別之閒聊或無關意圖


class NLUIntentResult(BaseModel):
    """
    SLM 意圖識別與槽位提取結果
    """
    intent: NLUIntent = Field(default=NLUIntent.UNKNOWN, description="識別出的核心使用者意圖")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="信心分數")
    
    # 交易相關槽位
    ticker: Optional[str] = Field(default=None, description="股票代號或名稱，如 2330 或 台積電")
    price: Optional[float] = Field(default=None, description="成交價格或參考價格")
    shares: Optional[float] = Field(default=None, description="股數")
    
    # 系統參數調整槽位
    config_field: Optional[str] = Field(
        default=None,
        description="修改之參數名稱: 'budget' (預算), 'take_profit' (停利), 'stop_loss' (停損), 'profit_alert' (漲幅通知)"
    )
    config_value: Optional[Union[float, List[float]]] = Field(
        default=None,
        description="參數新數值，停利可為單一數字或區間列表 [min, max]"
    )
    
    # 原始文本與模型附註
    raw_text: str = Field(default="", description="使用者的原始輸入訊息")
    explanation: Optional[str] = Field(default=None, description="SLM 簡短判定理由")
