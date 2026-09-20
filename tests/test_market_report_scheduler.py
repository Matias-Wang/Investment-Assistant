import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from apscheduler.triggers.cron import CronTrigger
from app.models.domain import MarketIndexItem
from app.models.llm import MarketReport
from app.services.scheduler_service import SchedulerService
from app.services.line_bot_service import LineBotService


@pytest.fixture
def mock_market_report():
    return MarketReport(
        title="美科技股反彈引領台股高檔震盪",
        date_str="2026-09-15 08:00",
        market_sentiment="樂觀偏多",
        index_summary="加權指數穩守月線上方，成交量維持溫和放大。",
        key_headlines=[
            "台積電美廠試產良率超預期",
            "輝達 Blackwell 晶片放量出貨",
            "Fed 降息預期激勵科技成長股",
        ],
        portfolio_insight="手中持股台積電受惠先進製程動能，短線展望穩健偏多。",
        actionable_takeaway="短線沿短期均線操作，強勢標的續抱，勿盲目追高。",
    )


@pytest.fixture
def mock_indices():
    return [
        MarketIndexItem(
            symbol="^TWII",
            name="台股加權",
            price=22500.0,
            change=150.0,
            change_percent=0.67,
        ),
        MarketIndexItem(
            symbol="2330.TW",
            name="台積電",
            price=1050.0,
            change=20.0,
            change_percent=1.94,
        ),
    ]


@pytest.mark.asyncio
async def test_scheduled_market_report_calls_generate_and_push(mock_market_report, mock_indices):
    """測試 SchedulerService 的 scheduled_market_report 能正確觸發報告生成並呼叫 LINE 推播"""
    sell_engine = MagicMock()
    buy_engine = MagicMock()
    line_service = MagicMock()
    line_service.push_market_report = AsyncMock()
    market_report_service = MagicMock()
    market_report_service.generate_market_report = AsyncMock(
        return_value=(mock_market_report, mock_indices)
    )

    scheduler = SchedulerService(
        sell_engine=sell_engine,
        buy_engine=buy_engine,
        line_service=line_service,
        market_report_service=market_report_service,
    )

    await scheduler.scheduled_market_report()

    market_report_service.generate_market_report.assert_awaited_once()
    line_service.push_market_report.assert_awaited_once_with(mock_market_report, mock_indices)


@pytest.mark.asyncio
async def test_scheduled_market_report_handles_missing_service():
    """測試當 MarketReportService 未注入時，不會拋出未捕捉例外"""
    sell_engine = MagicMock()
    buy_engine = MagicMock()
    line_service = MagicMock()
    line_service.push_market_report = AsyncMock()

    scheduler = SchedulerService(
        sell_engine=sell_engine,
        buy_engine=buy_engine,
        line_service=line_service,
        market_report_service=None,
    )

    await scheduler.scheduled_market_report()
    line_service.push_market_report.assert_not_awaited()


def test_scheduler_registers_morning_market_report_job():
    """測試 SchedulerService.start 正確註冊 08:00 Asia/Taipei 的排程工作"""
    sell_engine = MagicMock()
    buy_engine = MagicMock()
    line_service = MagicMock()
    market_report_service = MagicMock()

    scheduler = SchedulerService(
        sell_engine=sell_engine,
        buy_engine=buy_engine,
        line_service=line_service,
        market_report_service=market_report_service,
    )

    with patch.object(scheduler.scheduler, "start"):
        scheduler.start()

    job = scheduler.scheduler.get_job("morning_market_report_job")
    assert job is not None
    assert isinstance(job.trigger, CronTrigger)
    scheduler.shutdown()


@pytest.mark.asyncio
async def test_line_bot_service_push_market_report(mock_market_report, mock_indices):
    """測試 LineBotService.push_market_report 正確組裝 FlexMessage 並呼叫 push_message"""
    config_repo = MagicMock()
    portfolio_repo = MagicMock()
    session_repo = MagicMock()

    service = LineBotService(config_repo, portfolio_repo, session_repo)
    service.push_message = MagicMock()

    await service.push_market_report(
        report=mock_market_report,
        indices=mock_indices,
        target_user_id="U1234567890",
    )

    service.push_message.assert_called_once()
    args, _ = service.push_message.call_args
    assert args[0] == "U1234567890"
    messages = args[1]
    assert len(messages) == 1
    assert "早安市場快報" in messages[0].alt_text


@pytest.mark.asyncio
async def test_trigger_market_report_endpoint_push(mock_market_report, mock_indices):
    """測試 trigger_market_report API 端點在 push=True 時正確調用 push_market_report"""
    from app.controllers.api_controller import trigger_market_report

    mock_request = MagicMock()
    market_report_service = MagicMock()
    market_report_service.generate_market_report = AsyncMock(
        return_value=(mock_market_report, mock_indices)
    )
    line_service = MagicMock()
    line_service.push_market_report = AsyncMock()

    mock_request.app.state.market_report_service = market_report_service
    mock_request.app.state.line_service = line_service

    result = await trigger_market_report(mock_request, push=True)
    assert result["pushed"] is True
    line_service.push_market_report.assert_awaited_once_with(mock_market_report, mock_indices)
