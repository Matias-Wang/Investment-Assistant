import pytest
from unittest.mock import MagicMock, AsyncMock
from app.models.domain import SystemConfig, PortfolioItem, BuyCandidate
from app.services.sell_engine import SellMonitoringEngine
from app.services.buy_engine import BuyStrategyEngine
from app.services.line_bot_service import LineBotService
from app.repositories.json_repo import JsonPortfolioRepository


@pytest.mark.asyncio
async def test_regression_take_profit_above_max():
    """pnl_rate=0.25 (超過上限 0.20) 仍需正確產生 TAKE_PROFIT alert"""
    portfolio_repo = MagicMock()
    config_repo = MagicMock()
    market_service = MagicMock()

    config = SystemConfig(
        take_profit_min=0.10,
        take_profit_max=0.20,
        stop_loss=-0.50
    )
    config_repo.get_config = AsyncMock(return_value=config)

    # 獲利率 +25%
    portfolio_repo.get_all = AsyncMock(return_value=[
        PortfolioItem(
            ticker="2330",
            name="台積電",
            shares=10,
            average_cost=1000.0,
            status="holding"
        )
    ])
    market_service.get_current_price = MagicMock(return_value=1250.0)

    engine = SellMonitoringEngine(portfolio_repo, config_repo, market_service)
    alerts = await engine.scan_portfolio()

    assert len(alerts) == 1
    assert alerts[0].ticker == "2330"
    assert alerts[0].alert_type == "TAKE_PROFIT"
    assert "已超過停利上限" in alerts[0].trigger_reason


@pytest.mark.asyncio
async def test_regression_sell_alert_cooldown():
    """6 小時內相同型別之警示不應重複觸發"""
    from datetime import datetime, timezone, timedelta

    portfolio_repo = MagicMock()
    config_repo = MagicMock()
    market_service = MagicMock()

    config = SystemConfig(take_profit_min=0.10, take_profit_max=0.20, stop_loss=-0.50)
    config_repo.get_config = AsyncMock(return_value=config)

    # 模擬 2 小時前已發送過 TAKE_PROFIT
    two_hours_ago = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    portfolio_repo.get_all = AsyncMock(return_value=[
        PortfolioItem(
            ticker="2330",
            name="台積電",
            shares=10,
            average_cost=1000.0,
            status="holding",
            last_alert_type="TAKE_PROFIT",
            last_alert_at=two_hours_ago,
        )
    ])
    market_service.get_current_price = MagicMock(return_value=1250.0)

    engine = SellMonitoringEngine(portfolio_repo, config_repo, market_service)
    alerts = await engine.scan_portfolio()

    # 冷卻中，不應產生重複警示
    assert len(alerts) == 0



@pytest.mark.asyncio
async def test_regression_invalid_trade_input_does_not_crash(tmp_path):
    """'買入 2330 1.2.3 5' 不得擲例外拋給上層"""
    config_repo = AsyncMock()
    portfolio_repo = AsyncMock()
    session_repo = AsyncMock()
    session_repo.get_session.return_value = None

    service = LineBotService(config_repo, portfolio_repo, session_repo)
    service.reply_message = MagicMock()

    event = MagicMock()
    event.source.user_id = "test_user"
    event.message.text = "買入 2330 1.2.3 5"
    event.reply_token = "dummy_token"

    # 模擬未設定白名單時允許測試，或將 test_user 加入白名單
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(service, "_is_user_authorized", lambda uid: True)
        await service._handle_text_message(event, {})

    # 應該有回應，且不擲出 ValueError
    assert service.reply_message.called


@pytest.mark.asyncio
async def test_regression_negative_or_zero_shares_rejected(tmp_path):
    """shares <= 0 需被 repository 拒絕"""
    repo_file = tmp_path / "assets.json"
    repo = JsonPortfolioRepository(file_path=str(repo_file))

    with pytest.raises(ValueError):
        await repo.record_buy(ticker="2330", shares=0, price=1000)

    with pytest.raises(ValueError):
        await repo.record_buy(ticker="2330", shares=-5, price=1000)

    with pytest.raises(ValueError):
        await repo.record_sell(ticker="2330", shares=0)

    with pytest.raises(ValueError):
        await repo.record_sell(ticker="2330", shares=-10)


@pytest.mark.asyncio
async def test_regression_postback_param_injection():
    """postback data 含有非法 ticker 格式時應被拒絕"""
    config_repo = AsyncMock()
    portfolio_repo = AsyncMock()
    session_repo = AsyncMock()

    service = LineBotService(config_repo, portfolio_repo, session_repo)
    service.reply_message = MagicMock()

    event = MagicMock()
    event.source.user_id = "test_user"
    event.postback.data = "action=buy_click&ticker=2330&action=sell_click&shares=10&price=100"
    event.reply_token = "dummy_token"

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(service, "_is_user_authorized", lambda uid: True)
        await service._handle_postback_event(event, {})


@pytest.mark.asyncio
async def test_regression_buy_candidate_zero_price():
    """current_close <= 0 不得造成除零錯誤"""
    config_repo = AsyncMock()
    portfolio_repo = AsyncMock()
    market_service = MagicMock()
    news_service = AsyncMock()
    llm_service = MagicMock()

    config = SystemConfig(trade_budget_per_order=100000.0)
    config_repo.get_config = AsyncMock(return_value=config)

    # 價格回傳 0
    market_service.get_price_and_sma20 = MagicMock(return_value=(0.0, -10.0))

    engine = BuyStrategyEngine(config_repo, portfolio_repo, market_service, news_service, llm_service)
    results = await engine.scan_candidates(candidate_tickers=["2330"])
    assert len(results) == 0


@pytest.mark.asyncio
async def test_regression_market_report_generation_structure():
    """generate_market_report 必須回傳 (report, indices) tuple，不得為 None"""
    from app.services.market_report_service import MarketReportService
    from app.models.llm import MarketReport
    from app.models.domain import MarketIndexItem

    portfolio_repo = AsyncMock()
    portfolio_repo.get_all = AsyncMock(return_value=[])
    news_service = AsyncMock()
    news_service.fetch_google_news_rss = AsyncMock(return_value=[])

    service = MarketReportService(portfolio_repo, news_service)
    service.fetch_market_indices = MagicMock(return_value=[
        MarketIndexItem(symbol="^TWII", name="台股加權", price=22000.0, change=100.0, change_percent=0.45)
    ])

    mock_report = MarketReport(
        title="市場快報測試",
        date_str="2026-09-09",
        market_sentiment="中立震盪",
        index_summary="指數持穩",
        key_headlines=["焦點事件一", "焦點事件二"],
        portfolio_insight="部位安全",
        actionable_takeaway="按兵不動"
    )
    service._call_gemini_report = MagicMock(return_value=mock_report)

    report, indices = await service.generate_market_report()
    assert report is not None
    assert report.title == "市場快報測試"
    assert len(indices) == 1


@pytest.mark.asyncio
async def test_regression_optimistic_lock_version_conflict(tmp_path):
    """樂觀鎖版本不符合時應擲出 ConflictError"""
    from app.repositories.base import ConflictError

    repo_file = tmp_path / "assets.json"
    repo = JsonPortfolioRepository(file_path=str(repo_file))

    # 首次建倉 version=1
    item = await repo.record_buy(ticker="2330", shares=10, price=1000)
    assert item.version == 1

    # 再次買入 version 遞增為 2
    item2 = await repo.record_buy(ticker="2330", shares=5, price=1050)
    assert item2.version == 2

    # 使用過期版本 (v1) upsert 應拋出 ConflictError
    stale_item = PortfolioItem(
        ticker="2330",
        name="台積電",
        shares=10,
        average_cost=1000.0,
        version=1,
    )
    with pytest.raises(ConflictError):
        await repo.upsert(stale_item)


def test_regression_production_startup_validation():
    """APP_ENV == 'production' 時缺少金鑰必須擲例外中止啟動"""
    from app.core.config import Settings

    with pytest.raises(ValueError) as excinfo:
        Settings(
            APP_ENV="production",
            LINE_CHANNEL_SECRET="",
            LINE_CHANNEL_ACCESS_TOKEN="",
            GEMINI_API_KEY="",
            ADMIN_API_KEY="",
        )
    assert "Production startup failed" in str(excinfo.value)


@pytest.mark.asyncio
async def test_regression_decimal_precision_and_quantization(tmp_path):
    """買入成本計算需使用 Decimal 並以 quantize(0.0001) 收斂，避免浮點誤差"""
    from decimal import Decimal
    repo_file = tmp_path / "assets.json"
    repo = JsonPortfolioRepository(file_path=str(repo_file))

    # 買入 100 股 @ 10.333333
    item1 = await repo.record_buy(ticker="2330", shares=100.0, price=10.333333)
    assert isinstance(item1.shares, Decimal)
    assert isinstance(item1.average_cost, Decimal)
    assert item1.average_cost == Decimal("10.3333")

    # 再買入 50 股 @ 20.666666
    item2 = await repo.record_buy(ticker="2330", shares=50.0, price=20.666666)
    assert isinstance(item2.average_cost, Decimal)
    assert item2.average_cost == Decimal("13.7778")


@pytest.mark.asyncio
async def test_regression_profit_alert_cooldown():
    """PROFIT_ALERT 於 6 小時內不應重複觸發"""
    from datetime import datetime, timezone, timedelta

    portfolio_repo = MagicMock()
    config_repo = MagicMock()
    market_service = MagicMock()

    config = SystemConfig(profit_alert_threshold=0.05, take_profit_min=0.10, stop_loss=-0.50)
    config_repo.get_config = AsyncMock(return_value=config)

    two_hours_ago = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    portfolio_repo.get_all = AsyncMock(return_value=[
        PortfolioItem(
            ticker="2330",
            name="台積電",
            shares=10,
            average_cost=1000.0,
            status="holding",
            last_alert_type="PROFIT_ALERT",
            last_alert_at=two_hours_ago,
        )
    ])
    market_service.get_current_price = MagicMock(return_value=1060.0)

    engine = SellMonitoringEngine(portfolio_repo, config_repo, market_service)
    alerts = await engine.scan_portfolio()

    # 冷卻中，不應產生重複警示
    assert len(alerts) == 0


@pytest.mark.asyncio
async def test_regression_line_bot_profit_alert_command():
    """LINE 指令『設定漲幅通知 8%』應正確更新 profit_alert_threshold"""
    config = SystemConfig(profit_alert_threshold=0.05)
    config_repo = AsyncMock()
    config_repo.get_config = AsyncMock(return_value=config)
    config_repo.update_config = AsyncMock()

    portfolio_repo = AsyncMock()
    session_repo = AsyncMock()
    session_repo.get_session = AsyncMock(return_value=None)

    service = LineBotService(config_repo, portfolio_repo, session_repo)
    service.reply_message = MagicMock()

    event = MagicMock()
    event.source.user_id = "test_user"
    event.message.text = "設定漲幅通知 8%"
    event.reply_token = "dummy_token"

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(service, "_is_user_authorized", lambda uid: True)
        await service._handle_text_message(event, {})

    assert config.profit_alert_threshold == 0.08
    assert config_repo.update_config.called
    assert service.reply_message.called


def test_regression_profit_alert_flex_message():
    """PROFIT_ALERT 產生的 Flex Message 樣式與標題需正確"""
    from decimal import Decimal
    from app.models.domain import SellAlert
    from app.utils.flex_templates import build_sell_flex_message

    alert = SellAlert(
        ticker="2330",
        name="台積電",
        shares=10,
        current_price=Decimal("1060.00"),
        average_cost=Decimal("1000.00"),
        pnl_rate=0.06,
        pnl_amount=Decimal("600.00"),
        currency="TWD",
        alert_type="PROFIT_ALERT",
        trigger_reason="未實現損益率 +6.00%，已突破漲幅通知門檻 (+5.0%)",
    )
    flex_dict = build_sell_flex_message(alert)
    header = flex_dict["header"]
    assert header["backgroundColor"] == "#2E7D32"
    assert "持股漲幅通知" in header["contents"][0]["text"]




