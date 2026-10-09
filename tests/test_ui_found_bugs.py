"""以 UI 測試資料 (tests/fixtures/ui_test_assets.json) 實測時發現的三個問題之回歸測試。

1. 庫存總額把 USD 部位直接當 TWD 相加。
2. 已賣光（status=sold）的標的再「賣出」仍回報成功。
3. 部位在警示冷卻中時，賣出檢查回報「皆在安全範圍內」。
"""

import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.config import settings
from app.models.domain import PortfolioItem, SystemConfig
from app.models.line import UserSession
from app.repositories.json_repo import JsonPortfolioRepository
from app.services.sell_engine import SellMonitoringEngine, build_no_alert_notice, fx_rate_to_base
from app.utils.flex_templates import build_portfolio_flex_message


def _engine(holdings, prices, config=None):
    portfolio_repo = MagicMock()
    portfolio_repo.get_all = AsyncMock(return_value=holdings)
    portfolio_repo.upsert = AsyncMock()
    config_repo = MagicMock()
    config_repo.get_config = AsyncMock(return_value=config or SystemConfig())
    market_service = MagicMock()
    market_service.get_current_price = MagicMock(side_effect=lambda ticker, asset_type: prices[ticker])
    return SellMonitoringEngine(portfolio_repo, config_repo, market_service)


# ----------------------------------------------------------------------
# 1. 幣別換算
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_portfolio_totals_convert_usd_to_base_currency(monkeypatch):
    monkeypatch.setattr(settings, "USD_TWD_RATE", 32.0)
    engine = _engine(
        [
            PortfolioItem(ticker="2330", name="台積電", shares=10, average_cost=1000),
            PortfolioItem(ticker="AAPL", name="Apple", asset_type="us_stock", shares=5,
                          average_cost=300, currency="USD"),
        ],
        {"2330": 1100.0, "AAPL": 330.0},
    )

    val, cfg = await engine.get_portfolio_valuation()

    # 台股 10,000 + 美股 1,500 USD × 32 = 58,000（修正前會算成 11,500）
    assert val.total_cost == Decimal("58000.00")
    assert val.total_market_value == Decimal("63800.00")  # 11,000 + 1,650 × 32
    assert val.total_unrealized_pnl == Decimal("5800.00")
    assert val.total_unrealized_pnl_rate == 0.1
    # 單一部位維持原幣別
    aapl = next(i for i in val.items if i.ticker == "AAPL")
    assert aapl.currency == "USD" and aapl.cost_basis == Decimal("1500.00")


def test_fx_rate_to_base(monkeypatch):
    monkeypatch.setattr(settings, "USD_TWD_RATE", 32.0)
    assert fx_rate_to_base("TWD", "TWD") == 1
    assert fx_rate_to_base("USD", "TWD") == 32
    assert fx_rate_to_base("TWD", "USD") == Decimal("1") / 32
    assert fx_rate_to_base("JPY", "TWD") is None
    monkeypatch.setattr(settings, "USD_TWD_RATE", 0.0)
    assert fx_rate_to_base("USD", "TWD") is None


@pytest.mark.asyncio
async def test_unsupported_currency_is_excluded_from_totals_not_added_raw():
    engine = _engine(
        [
            PortfolioItem(ticker="2330", name="台積電", shares=10, average_cost=1000),
            PortfolioItem(ticker="7203", name="Toyota", asset_type="jp_stock", shares=100,
                          average_cost=3000, currency="JPY"),
        ],
        {"2330": 1000.0, "7203": 3000.0},
    )
    val, _ = await engine.get_portfolio_valuation()
    assert val.total_cost == Decimal("10000.00")
    assert len(val.items) == 2, "無法換算的部位仍須列出"


def test_portfolio_card_labels_currencies():
    from app.models.domain import PortfolioValuation, PositionValuation
    val = PortfolioValuation(items=[PositionValuation(
        ticker="AAPL", name="Apple", asset_type="us_stock", shares=5, average_cost=300,
        current_price=330, currency="USD", market_value=1650, cost_basis=1500,
        pnl_rate=0.1, pnl_amount=150,
    )])
    card = json.dumps(build_portfolio_flex_message(val, SystemConfig()), ensure_ascii=False)
    assert "成本: US$300.00" in card and "現價: US$330.00" in card
    assert "總建倉成本 (TWD)" in card


# ----------------------------------------------------------------------
# 2. 已賣光的標的
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_selling_already_sold_ticker_returns_none_without_writing(tmp_path):
    path = tmp_path / "assets.json"
    repo = JsonPortfolioRepository(file_path=str(path))
    await repo.record_buy(ticker="2882", shares=10, price=60)
    assert (await repo.record_sell(ticker="2882", shares=10)).status == "sold"
    before = path.read_bytes()

    assert await repo.record_sell(ticker="2882", shares=1) is None
    assert path.read_bytes() == before, "失敗的賣出不可寫入（version 也不可遞增）"


@pytest.mark.asyncio
async def test_line_sell_of_sold_ticker_is_reported_as_not_found():
    from app.services.line_bot_service import LineBotService

    portfolio_repo = MagicMock()
    portfolio_repo.record_sell = AsyncMock(return_value=None)
    session_repo = MagicMock()
    session_repo.get_session = AsyncMock(return_value=None)
    service = LineBotService(MagicMock(), portfolio_repo, session_repo)
    service._is_user_authorized = MagicMock(return_value=True)
    service.reply_message = MagicMock()

    event = MagicMock()
    event.source.user_id = "U1"
    event.message.text = "賣出 2882 60 1"
    await service._handle_text_message(event, {})

    text = service.reply_message.call_args.args[1][0].text
    assert "成功" not in text and "找不到持有中的庫存標的 2882" in text


@pytest.mark.asyncio
async def test_confirm_sell_on_sold_ticker_does_not_report_success():
    """按舊卡片的「手動已賣出」再回覆「確認」，但標的早已賣光。"""
    from app.services.line_bot_service import LineBotService

    portfolio_repo = MagicMock()
    portfolio_repo.record_sell = AsyncMock(return_value=None)
    session_repo = MagicMock()
    session_repo.clear_session = AsyncMock()
    service = LineBotService(MagicMock(), portfolio_repo, session_repo)
    service.reply_message = MagicMock()

    event = MagicMock()
    event.source.user_id = "U1"
    session = UserSession(user_id="U1", awaiting_action="CONFIRM_SELL", ticker="2603",
                          default_price=243, default_shares=100)
    assert await service._handle_session_reply(event, session, "確認") is True

    text = service.reply_message.call_args.args[1][0].text
    assert "成功" not in text and "找不到持有中的庫存標的 2603" in text
    session_repo.clear_session.assert_awaited_once_with("U1")


# ----------------------------------------------------------------------
# 3. 警示冷卻
# ----------------------------------------------------------------------
def _cooldown_engine():
    recent = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    return _engine(
        [
            PortfolioItem(ticker="2603", name="長榮", shares=100, average_cost=800,
                          last_alert_type="STOP_LOSS", last_alert_at=recent),
            PortfolioItem(ticker="2454", name="聯發科", shares=10, average_cost=2600),
            PortfolioItem(ticker="2330", name="台積電", shares=10, average_cost=1000),
        ],
        {"2603": 243.0, "2454": 5285.0, "2330": 1010.0},
    )


@pytest.mark.asyncio
async def test_cooldown_positions_are_reported_separately():
    result = await _cooldown_engine().scan_portfolio_detailed()

    assert [a.ticker for a in result.alerts] == ["2454"]
    assert [(c.ticker, c.alert_type) for c in result.cooling_down] == [("2603", "STOP_LOSS")]
    assert result.cooling_down[0].pnl_rate == pytest.approx(-0.6963, abs=1e-4)


@pytest.mark.asyncio
async def test_scan_portfolio_still_returns_only_alerts():
    """排程等既有呼叫端依賴的回傳型別不變。"""
    alerts = await _cooldown_engine().scan_portfolio()
    assert [a.ticker for a in alerts] == ["2454"]


def test_no_alert_notice_without_cooldown_uses_configured_thresholds():
    from app.models.domain import SellScanResult
    cfg = SystemConfig(take_profit_min=0.12, profit_alert_threshold=0.06, stop_loss=-0.3)
    text = build_no_alert_notice(SellScanResult(), cfg)
    assert "≥12%" in text and "≥6%" in text and "≤-30%" in text
