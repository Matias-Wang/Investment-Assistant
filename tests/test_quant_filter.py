import pytest
from unittest.mock import MagicMock, AsyncMock
from app.models.domain import SystemConfig
from app.models.llm import BuyRecommendation
from app.services.buy_engine import BuyStrategyEngine


@pytest.mark.asyncio
async def test_buy_quant_filter_logic():
    config_repo = MagicMock()
    portfolio_repo = MagicMock()
    market_service = MagicMock()
    news_service = MagicMock()
    llm_service = MagicMock()

    # 預算 100,000
    config_repo.get_config = AsyncMock(
        return_value=SystemConfig(trade_budget_per_order=100000.0)
    )
    portfolio_repo.get_all = AsyncMock(return_value=[])
    portfolio_repo.get_by_ticker = AsyncMock(return_value=None)

    # 模擬 2330: 現價 1000 > 20MA 950 (站上均線，零股 = 100000 // 1000 = 100 股)
    # 模擬 2313: 現價 200 < 20MA 220 (跌破均線，應被過濾)
    def mock_get_sma(ticker, asset_type):
        if ticker == "2330":
            return 1000.0, 950.0
        elif ticker == "2313":
            return 200.0, 220.0
        return None, None

    market_service.get_price_and_sma20 = MagicMock(side_effect=mock_get_sma)
    news_service.get_recent_news_text = AsyncMock(return_value="近期營收年增 30%，AI 晶片訂單滿載。")

    # 模擬 LLM 給出 BUY 決策
    llm_service.analyze_buy_candidate = MagicMock(
        return_value=BuyRecommendation(
            ticker="2330",
            company_name="台積電",
            decision="BUY",
            confidence_score=0.9,
            target_shares=100,
            estimated_amount=100000.0,
            sentiment_summary="市場需求強勁，營收成長亮眼",
            reasons=["營收大幅成長", "訂單能見度高"],
            risk_factors=["地緣政治風險"]
        )
    )

    engine = BuyStrategyEngine(
        config_repo, portfolio_repo, market_service, news_service, llm_service
    )

    results = await engine.scan_candidates(candidate_tickers=["2330", "2313"])

    # 2313 因未站上 20MA 被排除，僅 2330 進入 LLM 並被推薦買入
    assert len(results) == 1
    rec, cand = results[0]
    assert rec.ticker == "2330"
    assert rec.decision == "BUY"
    assert cand.target_shares == 100
    assert cand.estimated_amount == 100000.0
