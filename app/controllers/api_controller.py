import hmac
from typing import List, Optional
from fastapi import APIRouter, Request, HTTPException, Header, Depends, status, Query
from pydantic import BaseModel, Field
from app.core.config import settings
from app.models.domain import SystemConfig, PortfolioItem, SellAlert
from app.models.llm import BuyRecommendation
from app.services.sell_engine import build_no_alert_notice

router = APIRouter(prefix="/api", tags=["System Management & Triggers"])


async def verify_admin(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    """驗證管理員 API 金鑰，防止未授權操作 (Fail-closed)"""
    if not settings.ADMIN_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin API key is not configured on server"
        )
    if not x_api_key or not hmac.compare_digest(x_api_key, settings.ADMIN_API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header"
        )


class ManualTradeRequest(BaseModel):
    ticker: str = Field(..., pattern=r"^[A-Za-z0-9.]{1,10}$")
    shares: float = Field(..., gt=0, le=1_000_000)
    price: float = Field(..., gt=0, le=1_000_000)
    name: Optional[str] = None
    asset_type: str = "tw_stock"
    currency: str = "TWD"


class ScanResponse(BaseModel):
    status: str
    count: int
    items: list


@router.get("/health")
async def health_check(request: Request):
    """雲端健康檢查端點 (無需驗證)"""
    return {
        "status": "healthy",
        "app": request.app.title,
        "backend": request.app.state.config_repo.__class__.__name__,
    }


@router.get("/config", response_model=SystemConfig, dependencies=[Depends(verify_admin)])
async def get_system_config(request: Request):
    """查詢當前動態參數設定"""
    config_repo = request.app.state.config_repo
    return await config_repo.get_config()


@router.post("/config", response_model=SystemConfig, dependencies=[Depends(verify_admin)])
async def update_system_config(config: SystemConfig, request: Request):
    """更新系統動態參數設定"""
    config_repo = request.app.state.config_repo
    return await config_repo.update_config(config)


@router.get("/portfolio", dependencies=[Depends(verify_admin)])
async def get_portfolio(request: Request):
    """查詢當前庫存與評價"""
    sell_engine = request.app.state.sell_engine
    valuation, config = await sell_engine.get_portfolio_valuation()
    return {
        "valuation": valuation,
        "config": config,
    }


@router.post("/portfolio/buy", response_model=PortfolioItem, dependencies=[Depends(verify_admin)])
async def manual_record_buy(trade: ManualTradeRequest, request: Request):
    """手動記錄買入交易"""
    portfolio_repo = request.app.state.portfolio_repo
    item = await portfolio_repo.record_buy(
        ticker=trade.ticker,
        shares=trade.shares,
        price=trade.price,
        name=trade.name,
        asset_type=trade.asset_type,
        currency=trade.currency,
    )
    return item


@router.post("/portfolio/sell", dependencies=[Depends(verify_admin)])
async def manual_record_sell(trade: ManualTradeRequest, request: Request):
    """手動記錄賣出交易"""
    portfolio_repo = request.app.state.portfolio_repo
    item = await portfolio_repo.record_sell(ticker=trade.ticker, shares=trade.shares)
    if not item:
        raise HTTPException(status_code=404, detail=f"Ticker {trade.ticker} not found in portfolio")
    return item


@router.post("/trigger/sell-check", response_model=List[SellAlert], dependencies=[Depends(verify_admin)])
async def trigger_sell_check(
    request: Request,
    # 刻意不用 Query(default=...)：那會讓直接呼叫函式的測試拿到 Query 物件
    # （且它是 truthy），FastAPI 對非路徑的純量參數本來就會視為查詢參數。
    push: bool = False,
    user_id: Optional[str] = None,
    notify_empty: bool = True,
):
    """觸發賣出監控掃描 (Rule Engine)。

    同步完成後才回應，故可安全地由 Cloud Scheduler 或 Cloud Tasks 呼叫——
    Cloud Run 在請求處理期間會持續分配 CPU。

    `notify_empty`：無警示時是否也推播一則「未觸發」通知。
    使用者在 LINE 手動輸入「賣出檢查」時需要（預設 True），否則只看到「檢查中」
    後一片安靜，與故障無法區分；但每 30 分鐘一次的排程應帶 `notify_empty=false`,
    只在真的該賣時才出聲，不然一天會收到十幾則「沒事」。
    """
    sell_engine = request.app.state.sell_engine
    result = await sell_engine.scan_portfolio_detailed()
    alerts = result.alerts
    line_service = getattr(request.app.state, "line_service", None)
    if push and line_service:
        if not alerts and notify_empty:
            # 無警示時也要回報，否則使用者只看到「檢查中」後一片安靜，
            # 與 Cloud Tasks／SERVICE_BASE_URL 沒生效的故障狀況無法區分。
            # 冷卻中的部位須一併說明，不可說成「未觸發」。
            config = await request.app.state.config_repo.get_config()
            await line_service.push_text(
                build_no_alert_notice(result, config),
                target_user_id=user_id,
            )
        for alert in alerts:
            await line_service.push_sell_alert(alert, target_user_id=user_id)
    return alerts


@router.post("/trigger/buy-scan", dependencies=[Depends(verify_admin)])
async def trigger_buy_scan(
    request: Request,
    push: bool = False,
    user_id: Optional[str] = None,
):
    """觸發買入策略掃描 (Quant + RAG + LLM)。

    同步完成後才回應（約 90 秒），故可安全地由 Cloud Scheduler 或 Cloud Tasks 呼叫。
    幻覺驗證層的模型會在此路徑首次使用時載入。
    """
    buy_engine = request.app.state.buy_engine
    results = await buy_engine.scan_candidates()
    line_service = getattr(request.app.state, "line_service", None)
    if push and line_service:
        if not results:
            await line_service.push_text(
                "ℹ️ 今日掃描完成：目前觀察名單中尚無同時符合 20MA 與 AI 買入評級之標的。",
                target_user_id=user_id,
            )
        for rec, candidate in results:
            await line_service.push_buy_recommendation(rec, candidate, target_user_id=user_id)
    return [
        {
            "recommendation": rec.model_dump(),
            "candidate": candidate.model_dump(),
        }
        for rec, candidate in results
    ]


@router.post("/trigger/market-report", dependencies=[Depends(verify_admin)])
async def trigger_market_report(
    request: Request,
    push: bool = False,
    user_id: Optional[str] = None,
):
    """手動或外部排程觸發市場快訊報告 (AI 綜合分析)，可選是否推播至 LINE"""
    market_report_service = getattr(request.app.state, "market_report_service", None)
    line_service = getattr(request.app.state, "line_service", None)
    if not market_report_service:
        raise HTTPException(status_code=503, detail="Market report service not initialized")
    report, indices = await market_report_service.generate_market_report()
    pushed = False
    if push and line_service:
        await line_service.push_market_report(report, indices, target_user_id=user_id)
        pushed = True
    return {
        "report": report.model_dump(),
        "indices": [idx.model_dump() for idx in indices],
        "pushed": pushed,
    }


@router.post("/trigger/industry-report", dependencies=[Depends(verify_admin)])
async def trigger_industry_report(
    request: Request,
    push: bool = False,
    user_id: Optional[str] = None,
):
    """手動或外部排程觸發每日產業報告 (依持股所屬產業)，可選是否推播至 LINE"""
    industry_report_service = getattr(request.app.state, "industry_report_service", None)
    line_service = getattr(request.app.state, "line_service", None)
    if not industry_report_service:
        raise HTTPException(status_code=503, detail="Industry report service not initialized")
    try:
        report, performance = await industry_report_service.generate_industry_report()
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    pushed = False
    if push and line_service:
        await line_service.push_industry_report(report, performance, target_user_id=user_id)
        pushed = True
    return {
        "report": report.model_dump(),
        "performance": [p.model_dump() for p in performance],
        "pushed": pushed,
    }

