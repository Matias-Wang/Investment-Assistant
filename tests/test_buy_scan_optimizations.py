"""P0-3 / P0-4 / P0-5 的買入掃描最佳化測試。

覆蓋：
* 新聞快取（同標的不重抓、TTL 到期重抓、查無新聞不快取）
* 掃描並行化（牆鐘時間不隨標的數線性增長、並行度受 semaphore 限制、順序仍為決定性）
* 預篩三種模式（off / shadow / enforce）與 fail-open
* Gemini 呼叫確實帶上 output token 上限
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.config import settings
from app.models.domain import SystemConfig
from app.models.llm import BuyRecommendation
from app.services.buy_engine import BuyStrategyEngine
from app.services.news_filter import NewsRelevanceFilter
from app.services.news_service import NewsService

GOOD_NEWS = "營收年增 30%，AI 訂單滿載，擴產計畫啟動。"
NOISE_NEWS = "董事長出席公益路跑活動，與員工完成五公里。"


def _recommendation(ticker: str, decision: str = "BUY") -> BuyRecommendation:
    return BuyRecommendation(
        ticker=ticker,
        company_name=ticker,
        decision=decision,
        confidence_score=0.9,
        target_shares=10,
        estimated_amount=10000.0,
        sentiment_summary="測試用摘要",
        reasons=["營收成長"],
        risk_factors=["匯率"],
    )


def _build_engine(
    tickers_prices: dict,
    news_text: str = GOOD_NEWS,
    decision: str = "BUY",
    news_filter: NewsRelevanceFilter | None = None,
    news_delay: float = 0.0,
):
    """組裝一個全 mock 的 BuyStrategyEngine。"""
    config_repo, portfolio_repo = MagicMock(), MagicMock()
    market_service, news_service, llm_service = MagicMock(), MagicMock(), MagicMock()

    config_repo.get_config = AsyncMock(return_value=SystemConfig(trade_budget_per_order=100000.0))
    portfolio_repo.get_all = AsyncMock(return_value=[])
    portfolio_repo.get_by_ticker = AsyncMock(return_value=None)

    market_service.get_price_and_sma20 = MagicMock(
        side_effect=lambda t, a: tickers_prices.get(t, (None, None))
    )

    async def _fetch_news(ticker, name, asset_type):
        if news_delay:
            await asyncio.sleep(news_delay)
        return news_text

    news_service.get_recent_news_text = AsyncMock(side_effect=_fetch_news)
    llm_service.analyze_buy_candidate = MagicMock(
        side_effect=lambda cand, txt: _recommendation(cand.ticker, decision)
    )

    engine = BuyStrategyEngine(
        config_repo, portfolio_repo, market_service, news_service, llm_service,
        news_filter=news_filter,
    )
    return engine, llm_service, news_service


# ---------------------------------------------------------------- 新聞快取

@pytest.mark.asyncio
async def test_news_cache_avoids_refetch(monkeypatch):
    monkeypatch.setattr(settings, "NEWS_CACHE_TTL_SECONDS", 3600)
    svc = NewsService()
    svc.fetch_google_news_rss = AsyncMock(return_value=[
        MagicMock(title="營收創高", publisher="A", published_at="2026-09-20", snippet="營收年增 30%")
    ])
    svc.fetch_yfinance_news = MagicMock(return_value=[])

    first = await svc.get_recent_news_text("2330", "台積電")
    second = await svc.get_recent_news_text("2330", "台積電")

    assert first == second
    # 第二次應完全走快取，不再打 RSS
    assert svc.fetch_google_news_rss.await_count == 1


@pytest.mark.asyncio
async def test_news_cache_expires(monkeypatch):
    monkeypatch.setattr(settings, "NEWS_CACHE_TTL_SECONDS", 1)
    svc = NewsService()
    svc.fetch_google_news_rss = AsyncMock(return_value=[
        MagicMock(title="營收創高", publisher="A", published_at="", snippet="營收年增")
    ])
    svc.fetch_yfinance_news = MagicMock(return_value=[])

    await svc.get_recent_news_text("2330", "台積電")
    # 直接把寫入時間往前推，模擬 TTL 過期（避免測試真的 sleep）
    key = "2330|tw_stock"
    cached_at, text = svc._news_cache[key]
    svc._news_cache[key] = (cached_at - 10, text)

    await svc.get_recent_news_text("2330", "台積電")
    assert svc.fetch_google_news_rss.await_count == 2


@pytest.mark.asyncio
async def test_news_cache_disabled_when_ttl_zero(monkeypatch):
    monkeypatch.setattr(settings, "NEWS_CACHE_TTL_SECONDS", 0)
    svc = NewsService()
    svc.fetch_google_news_rss = AsyncMock(return_value=[
        MagicMock(title="營收創高", publisher="A", published_at="", snippet="營收年增")
    ])
    svc.fetch_yfinance_news = MagicMock(return_value=[])

    await svc.get_recent_news_text("2330", "台積電")
    await svc.get_recent_news_text("2330", "台積電")
    assert svc.fetch_google_news_rss.await_count == 2


@pytest.mark.asyncio
async def test_empty_news_is_not_cached(monkeypatch):
    """抓取失敗與真的沒新聞無法區分，快取空結果會遮蔽後續真實新聞。"""
    monkeypatch.setattr(settings, "NEWS_CACHE_TTL_SECONDS", 3600)
    svc = NewsService()
    svc.fetch_google_news_rss = AsyncMock(return_value=[])
    svc.fetch_yfinance_news = MagicMock(return_value=[])

    text = await svc.get_recent_news_text("2330", "台積電")
    assert "查無" in text
    assert svc._news_cache == {}


# ---------------------------------------------------------------- 並行化

@pytest.mark.asyncio
async def test_scan_runs_concurrently(monkeypatch):
    """6 檔各延遲 0.1s，並行度 3 時牆鐘時間應遠低於序列的 0.6s。"""
    monkeypatch.setattr(settings, "BUY_SCAN_CONCURRENCY", 3)
    monkeypatch.setattr(settings, "NEWS_FILTER_MODE", "off")

    prices = {t: (100.0, 90.0) for t in ["A", "B", "C", "D", "E", "F"]}
    engine, _, _ = _build_engine(prices, news_delay=0.1)

    start = time.perf_counter()
    results = await engine.scan_candidates(candidate_tickers=list(prices))
    elapsed = time.perf_counter() - start

    assert len(results) == 6
    assert elapsed < 0.45, f"看起來仍是序列執行（耗時 {elapsed:.2f}s）"


@pytest.mark.asyncio
async def test_concurrency_limit_is_respected(monkeypatch):
    """semaphore 必須真的限制同時在飛的請求數，以免觸發上游 429。"""
    monkeypatch.setattr(settings, "BUY_SCAN_CONCURRENCY", 2)
    monkeypatch.setattr(settings, "NEWS_FILTER_MODE", "off")

    in_flight = 0
    peak = 0

    async def _tracked_news(ticker, name, asset_type):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.05)
        in_flight -= 1
        return GOOD_NEWS

    prices = {t: (100.0, 90.0) for t in ["A", "B", "C", "D", "E"]}
    engine, _, news_service = _build_engine(prices)
    news_service.get_recent_news_text = AsyncMock(side_effect=_tracked_news)

    await engine.scan_candidates(candidate_tickers=list(prices))
    assert peak <= 2, f"並行度超出上限：peak={peak}"


@pytest.mark.asyncio
async def test_result_order_is_deterministic(monkeypatch):
    """並行化後輸出順序仍須與輸入一致，否則推播順序會隨機跳動。"""
    monkeypatch.setattr(settings, "BUY_SCAN_CONCURRENCY", 5)
    monkeypatch.setattr(settings, "NEWS_FILTER_MODE", "off")

    prices = {t: (100.0, 90.0) for t in ["A", "B", "C", "D"]}
    engine, _, _ = _build_engine(prices)
    results = await engine.scan_candidates(candidate_tickers=["A", "B", "C", "D"])
    assert [cand.ticker for _, cand in results] == ["A", "B", "C", "D"]


@pytest.mark.asyncio
async def test_one_candidate_failure_does_not_abort_scan(monkeypatch):
    monkeypatch.setattr(settings, "NEWS_FILTER_MODE", "off")
    prices = {"A": (100.0, 90.0), "B": (100.0, 90.0)}
    engine, llm_service, _ = _build_engine(prices)

    def _flaky(cand, txt):
        if cand.ticker == "A":
            raise RuntimeError("Gemini 500")
        return _recommendation(cand.ticker)

    llm_service.analyze_buy_candidate = MagicMock(side_effect=_flaky)
    results = await engine.scan_candidates(candidate_tickers=["A", "B"])
    assert [c.ticker for _, c in results] == ["B"]


# ---------------------------------------------------------------- 預篩模式

@pytest.mark.asyncio
async def test_shadow_mode_does_not_drop_candidates(monkeypatch):
    """shadow 為預設值：即使預篩判為不通過，也必須照樣送 Gemini。"""
    monkeypatch.setattr(settings, "NEWS_FILTER_MODE", "shadow")
    prices = {"A": (100.0, 90.0)}
    engine, llm_service, _ = _build_engine(prices, news_text=NOISE_NEWS)

    results = await engine.scan_candidates(candidate_tickers=["A"])
    assert llm_service.analyze_buy_candidate.call_count == 1
    assert len(results) == 1


@pytest.mark.asyncio
async def test_enforce_mode_drops_noise(monkeypatch):
    monkeypatch.setattr(settings, "NEWS_FILTER_MODE", "enforce")
    prices = {"A": (100.0, 90.0)}
    engine, llm_service, _ = _build_engine(prices, news_text=NOISE_NEWS)

    results = await engine.scan_candidates(candidate_tickers=["A"])
    assert llm_service.analyze_buy_candidate.call_count == 0
    assert results == []


@pytest.mark.asyncio
async def test_enforce_mode_keeps_substantive_news(monkeypatch):
    monkeypatch.setattr(settings, "NEWS_FILTER_MODE", "enforce")
    prices = {"A": (100.0, 90.0)}
    engine, llm_service, _ = _build_engine(prices, news_text=GOOD_NEWS)

    results = await engine.scan_candidates(candidate_tickers=["A"])
    assert llm_service.analyze_buy_candidate.call_count == 1
    assert len(results) == 1


@pytest.mark.asyncio
async def test_prefilter_error_fails_open(monkeypatch):
    """預篩自身爆掉時必須放行給 Gemini，絕不可 fail-closed 吞掉標的。"""
    monkeypatch.setattr(settings, "NEWS_FILTER_MODE", "enforce")
    broken = MagicMock()
    broken.evaluate = MagicMock(side_effect=RuntimeError("regex boom"))

    prices = {"A": (100.0, 90.0)}
    engine, llm_service, _ = _build_engine(prices, news_filter=broken)

    results = await engine.scan_candidates(candidate_tickers=["A"])
    assert llm_service.analyze_buy_candidate.call_count == 1
    assert len(results) == 1


# ---------------------------------------------------------------- output 上限

def test_buy_analysis_sends_max_output_tokens():
    """output 單價是 input 的 8.3 倍，上限必須真的送進 API。"""
    from app.services.llm_service import LLMService
    from app.models.domain import BuyCandidate

    svc = LLMService()
    svc.api_key = "test-key"
    candidate = BuyCandidate(
        ticker="2330", name="台積電", asset_type="tw_stock",
        current_price=1000.0, sma_20=950.0, target_shares=10,
        estimated_amount=10000.0, currency="TWD",
    )

    fake_client = MagicMock()
    fake_client.models.generate_content = MagicMock(
        return_value=MagicMock(parsed=_recommendation("2330"), text=None)
    )
    svc._client = fake_client

    svc.analyze_buy_candidate(candidate, GOOD_NEWS)

    config = fake_client.models.generate_content.call_args.kwargs["config"]
    assert config.max_output_tokens == settings.BUY_ANALYSIS_MAX_OUTPUT_TOKENS
    assert config.max_output_tokens > 0


def test_thinking_config_omitted_by_default(monkeypatch):
    """預設不送 thinking 參數，避免不支援該參數的模型整批失敗。"""
    from app.core.gemini_options import thinking_kwargs

    monkeypatch.setattr(settings, "GEMINI_THINKING_LEVEL", "")
    assert thinking_kwargs() == {}


def test_thinking_config_applied_when_set(monkeypatch):
    from app.core.gemini_options import thinking_kwargs

    monkeypatch.setattr(settings, "GEMINI_THINKING_LEVEL", "minimal")
    kwargs = thinking_kwargs()
    assert "thinking_config" in kwargs
    assert kwargs["thinking_config"].thinking_level == "MINIMAL"


def test_invalid_thinking_level_is_ignored(monkeypatch):
    from app.core.gemini_options import thinking_kwargs

    monkeypatch.setattr(settings, "GEMINI_THINKING_LEVEL", "ludicrous")
    assert thinking_kwargs() == {}
