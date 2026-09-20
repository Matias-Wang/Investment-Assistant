from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class UserSession(BaseModel):
    """
    LINE 用戶對話狀態 (用於狀態更新對話流)
    """
    user_id: str
    awaiting_action: Optional[str] = None  # 例如: "CONFIRM_BUY", "CONFIRM_SELL"
    ticker: Optional[str] = None
    default_price: Optional[float] = None
    default_shares: Optional[float] = None
    extra_data: Dict[str, Any] = Field(default_factory=dict)
    updated_at: float = Field(default=0.0)


class LinePostbackPayload(BaseModel):
    """
    按鈕回傳動作資料載荷
    """
    action: str  # e.g., "buy_click", "sell_click", "quick_action"
    ticker: str
    shares: Optional[float] = None
    price: Optional[float] = None
