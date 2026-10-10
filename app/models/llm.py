from typing import List, Literal
from pydantic import BaseModel, Field


class BuyRecommendation(BaseModel):
    """
    Gemini LLM 買入建議結構化輸出模式
    """
    ticker: str = Field(..., description="股票代號，例如 '2330'")
    company_name: str = Field(..., description="公司名稱，例如 '台積電'")
    decision: Literal["BUY", "HOLD", "AVOID"] = Field(
        ...,
        description="操作決策：僅能為 'BUY' (買入)、'HOLD' (觀望) 或 'AVOID' (避開)"
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="LLM 評估信心度分數 (0.0 到 1.0)"
    )
    target_shares: int = Field(
        ...,
        ge=0,
        description="建議買入零股股數"
    )
    estimated_amount: float = Field(
        ...,
        ge=0.0,
        description="預估花費總金額"
    )
    sentiment_summary: str = Field(
        ...,
        description="基於提供之近期新聞文本所歸納的情緒與基本面摘要"
    )
    reasons: List[str] = Field(
        default_factory=list,
        description="僅依據新聞文本所提取的支持推論理由列表"
    )
    risk_factors: List[str] = Field(
        default_factory=list,
        description="僅依據新聞文本所提及的風險因素列表"
    )


class MarketReport(BaseModel):
    """
    Gemini LLM 市場分析報告結構化輸出
    """
    title: str = Field(..., description="市場快訊精簡標題，例如 '台美股市高檔震盪，AI供應鏈動能續強'")
    date_str: str = Field(..., description="報告生成時間或日期，例如 '2026-09-07'")
    market_sentiment: Literal["樂觀偏多", "中立震盪", "保守觀望", "空方承壓"] = Field(
        ...,
        description="整體市場氛圍評估"
    )
    index_summary: str = Field(..., description="大盤指數與總經走勢精簡解析 (1-2 句話)")
    key_headlines: List[str] = Field(
        default_factory=list,
        description="3~4 則今日關鍵財經焦點與產業大事"
    )
    portfolio_insight: str = Field(
        ...,
        description="針對使用者持股 (如台積電、光聖、華通、SPCX 等) 之實質影響或受惠分析"
    )
    actionable_takeaway: str = Field(
        ...,
        description="給使用者的具體短線操作策略或風控叮嚀 (1 句話)"
    )


class IndustrySection(BaseModel):
    """單一產業的趨勢解析"""
    industry: str = Field(..., description="產業名稱，必須與提供之【持股產業分組】名稱完全一致")
    outlook: Literal["偏多", "中性", "偏空"] = Field(..., description="依據提供之行情與新聞評估的產業短線展望")
    summary: str = Field(..., description="該產業近況與驅動因素 (1-2 句話，僅依據提供之新聞與行情)")


class IndustryReportDraft(BaseModel):
    """
    交給 Gemini 填寫的欄位：只含需要「語意判斷」的部分。
    氛圍、強弱對比、日期都能由代碼從行情算出，不讓 LLM 接手 (見 IndustryReportService)。
    """
    title: str = Field(..., description="產業快訊精簡標題，例如 'AI 伺服器與 PCB 供應鏈動能延續'")
    industries: List[IndustrySection] = Field(
        default_factory=list,
        description="每個持股產業一則趨勢解析，順序同【持股產業分組】"
    )
    key_headlines: List[str] = Field(
        default_factory=list,
        description="3~4 則今日關鍵產業大事 (每條 20 字以內)"
    )
    portfolio_insight: str = Field(
        ...,
        description="針對使用者持股所屬產業的實質影響或受惠/受害分析"
    )
    actionable_takeaway: str = Field(
        ...,
        description="給使用者的具體短線操作策略或風控叮嚀 (1 句話)"
    )


class IndustryReport(BaseModel):
    """
    每日產業報告完整結構 (格式比照 MarketReport)。
    title / industries / key_headlines / portfolio_insight / actionable_takeaway 來自 LLM；
    date_str / market_sentiment / index_summary 由代碼產生。
    """
    title: str = Field(..., description="產業快訊精簡標題，例如 'AI 伺服器與 PCB 供應鏈動能延續'")
    date_str: str = Field(..., description="報告生成時間或日期，例如 '2026-09-07 08:30'")
    market_sentiment: Literal["樂觀偏多", "中立震盪", "保守觀望", "空方承壓"] = Field(
        ...,
        description="持股所屬產業的整體氛圍評估"
    )
    index_summary: str = Field(..., description="各產業今日強弱對比精簡解析 (1-2 句話)")
    industries: List[IndustrySection] = Field(
        default_factory=list,
        description="每個持股產業一則趨勢解析，順序同【持股產業分組】"
    )
    key_headlines: List[str] = Field(
        default_factory=list,
        description="3~4 則今日關鍵產業大事 (每條 20 字以內)"
    )
    portfolio_insight: str = Field(
        ...,
        description="針對使用者持股所屬產業的實質影響或受惠/受害分析"
    )
    actionable_takeaway: str = Field(
        ...,
        description="給使用者的具體短線操作策略或風控叮嚀 (1 句話)"
    )

