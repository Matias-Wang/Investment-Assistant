from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional, List
from pydantic import BaseModel, Field


class SystemConfig(BaseModel):
    """
    動態系統設定
    包含單次投資預算、持股漲幅通知門檻、停利區間 (min ~ max)、停損點
    """
    trade_budget_per_order: Decimal = Field(default=Decimal("100000.0"), description="單次買入預算 (預設 100,000)")
    profit_alert_threshold: float = Field(default=0.05, description="持股漲幅通知門檻 (預設 5%)")
    take_profit_min: float = Field(default=0.10, description="停利點下限 (預設 10%)")
    take_profit_max: float = Field(default=0.20, description="停利點上限 (預設 20%)")
    stop_loss: float = Field(default=-0.50, description="停損點 (預設 -50%)")
    currency: str = Field(default="TWD", description="基準計價幣別")


class PortfolioItem(BaseModel):
    """
    持股庫存項目
    """
    ticker: str = Field(..., description="股票代號，例如 '2330'")
    name: str = Field(..., description="標的名稱，例如 '台積電'")
    asset_type: str = Field(default="tw_stock", description="資產類型 ('tw_stock' 或 'us_stock')")
    shares: Decimal = Field(..., description="持股股數")
    average_cost: Decimal = Field(..., description="平均建倉成本")
    currency: str = Field(default="TWD", description="幣別")
    purchase_date: Optional[str] = Field(default=None, description="建倉日期")
    last_updated: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="最後更新時間"
    )
    status: str = Field(default="holding", description="狀態 ('holding' 或 'sold')")
    last_alert_type: Optional[str] = Field(default=None, description="前次觸發之警示類型 ('TAKE_PROFIT', 'STOP_LOSS' 或 'PROFIT_ALERT')")
    last_alert_at: Optional[str] = Field(default=None, description="前次觸發警示之 ISO 8601 時間戳記")
    version: int = Field(default=1, description="樂觀鎖版本號")


class PositionValuation(BaseModel):
    """
    單一持股評價
    """
    ticker: str
    name: str
    asset_type: str
    shares: Decimal
    average_cost: Decimal
    current_price: Decimal
    currency: str
    market_value: Decimal
    cost_basis: Decimal
    pnl_rate: float
    pnl_amount: Decimal


class PortfolioValuation(BaseModel):
    """
    投資組合整體評價
    """
    items: List[PositionValuation] = Field(default_factory=list)
    total_cost: Decimal = Decimal("0.0")
    total_market_value: Decimal = Decimal("0.0")
    total_unrealized_pnl: Decimal = Decimal("0.0")
    total_unrealized_pnl_rate: float = 0.0


class SellAlert(BaseModel):
    """
    賣出警示物件 (由規則引擎產出)
    """
    ticker: str
    name: str
    shares: Decimal
    current_price: Decimal
    average_cost: Decimal
    pnl_rate: float
    pnl_amount: Decimal
    currency: str
    alert_type: str  # "TAKE_PROFIT", "STOP_LOSS" 或 "PROFIT_ALERT"
    trigger_reason: str


class BuyCandidate(BaseModel):
    """
    通過量化過濾與預算計算之候選標的
    """
    ticker: str
    name: str
    asset_type: str
    current_price: float
    sma_20: float
    target_shares: int
    estimated_amount: float
    currency: str = "TWD"


class NewsItem(BaseModel):
    """
    新聞資料結構 (供 RAG 文本檢索使用)
    """
    title: str
    publisher: str
    published_at: Optional[str] = None
    link: Optional[str] = None
    snippet: str


class MarketIndexItem(BaseModel):
    """
    市場大盤主要指數即時數據
    """
    symbol: str
    name: str
    price: float
    change: float
    change_percent: float

