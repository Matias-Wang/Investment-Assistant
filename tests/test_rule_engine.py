import pytest
from unittest.mock import MagicMock, AsyncMock
from app.models.domain import PortfolioItem, SystemConfig
from app.services.sell_engine import SellMonitoringEngine


@pytest.mark.asyncio
async def test_sell_rule_engine_take_profit():
    # 建立假 Repo 與 MarketService
    portfolio_repo = MagicMock()
    config_repo = MagicMock()
    market_service = MagicMock()

    # 系統設定：停利 10% ~ 20%，停損 -50%
    config = SystemConfig(
        trade_budget_per_order=100000.0,
        take_profit_min=0.10,
        take_profit_max=0.20,
        stop_loss=-0.50
    )
    config_repo.get_config = AsyncMock(return_value=config)

    # 模擬持股：成本 1000，現價 1150 (獲利率 +15%，介於 [10%, 20%])
    portfolio_repo.get_all = AsyncMock(return_value=[
        PortfolioItem(
            ticker="2330",
            name="台積電",
            shares=10,
            average_cost=1000.0,
            status="holding"
        )
    ])
    market_service.get_current_price = MagicMock(return_value=1150.0)

    engine = SellMonitoringEngine(portfolio_repo, config_repo, market_service)
    alerts = await engine.scan_portfolio()

    assert len(alerts) == 1
    assert alerts[0].ticker == "2330"
    assert alerts[0].alert_type == "TAKE_PROFIT"
    assert alerts[0].pnl_rate == 0.15
    assert alerts[0].pnl_amount == 1500.0


@pytest.mark.asyncio
async def test_sell_rule_engine_stop_loss():
    portfolio_repo = MagicMock()
    config_repo = MagicMock()
    market_service = MagicMock()

    config = SystemConfig(
        trade_budget_per_order=100000.0,
        take_profit_min=0.10,
        take_profit_max=0.20,
        stop_loss=-0.50
    )
    config_repo.get_config = AsyncMock(return_value=config)

    # 模擬持股：成本 1000，現價 450 (跌幅 -55%，低於 -50%)
    portfolio_repo.get_all = AsyncMock(return_value=[
        PortfolioItem(
            ticker="6442",
            name="光聖",
            shares=50,
            average_cost=1000.0,
            status="holding"
        )
    ])
    market_service.get_current_price = MagicMock(return_value=450.0)

    engine = SellMonitoringEngine(portfolio_repo, config_repo, market_service)
    alerts = await engine.scan_portfolio()

    assert len(alerts) == 1
    assert alerts[0].ticker == "6442"
    assert alerts[0].alert_type == "STOP_LOSS"
    assert alerts[0].pnl_rate == -0.55
    assert alerts[0].pnl_amount == -27500.0


@pytest.mark.asyncio
async def test_sell_rule_engine_normal_holding():
    portfolio_repo = MagicMock()
    config_repo = MagicMock()
    market_service = MagicMock()

    config = SystemConfig(
        trade_budget_per_order=100000.0,
        take_profit_min=0.10,
        take_profit_max=0.20,
        stop_loss=-0.50
    )
    config_repo.get_config = AsyncMock(return_value=config)

    # 模擬持股：成本 1000，現價 1020 (獲利率 +2%，未達 5% 漲幅門檻，未達停利門檻，未跌破停損)
    portfolio_repo.get_all = AsyncMock(return_value=[
        PortfolioItem(
            ticker="2313",
            name="華通",
            shares=200,
            average_cost=1000.0,
            status="holding"
        )
    ])
    market_service.get_current_price = MagicMock(return_value=1020.0)

    engine = SellMonitoringEngine(portfolio_repo, config_repo, market_service)
    alerts = await engine.scan_portfolio()

    assert len(alerts) == 0


@pytest.mark.asyncio
async def test_sell_rule_engine_profit_alert():
    """持有股票漲幅超過 5% 時生成 PROFIT_ALERT 警示"""
    portfolio_repo = MagicMock()
    config_repo = MagicMock()
    market_service = MagicMock()

    config = SystemConfig(
        trade_budget_per_order=100000.0,
        profit_alert_threshold=0.05,
        take_profit_min=0.10,
        take_profit_max=0.20,
        stop_loss=-0.50
    )
    config_repo.get_config = AsyncMock(return_value=config)

    # 模擬持股：成本 1000，現價 1060 (獲利率 +6%，超過 5% 漲幅門檻，且未達 10% 停利點)
    portfolio_repo.get_all = AsyncMock(return_value=[
        PortfolioItem(
            ticker="2330",
            name="台積電",
            shares=10,
            average_cost=1000.0,
            status="holding"
        )
    ])
    market_service.get_current_price = MagicMock(return_value=1060.0)

    engine = SellMonitoringEngine(portfolio_repo, config_repo, market_service)
    alerts = await engine.scan_portfolio()

    assert len(alerts) == 1
    assert alerts[0].ticker == "2330"
    assert alerts[0].alert_type == "PROFIT_ALERT"
    assert alerts[0].pnl_rate == 0.06
    assert alerts[0].pnl_amount == 600.0
    assert "已突破漲幅通知門檻" in alerts[0].trigger_reason

