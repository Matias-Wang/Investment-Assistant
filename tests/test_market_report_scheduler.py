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

    market_report_service.generate_market_report.assert_awaited_once_with(edition="pre")
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
    assert "盤前快報" in messages[0].alt_text


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
    # 端點新增 user_id 參數以支援 Cloud Tasks 指定推播對象
    line_service.push_market_report.assert_awaited_once_with(
        mock_market_report, mock_indices, target_user_id=None
    )


# ---------------------------------------------------------------------------
# 盤前／盤後版別與資料時點標示
# ---------------------------------------------------------------------------
from datetime import datetime  # noqa: E402

from app.services.market_report_service import (  # noqa: E402
    TAIPEI_TZ,
    MarketDataUnavailable,
    MarketNotTradedToday,
    MarketReportService,
    build_data_basis,
    describe_data_time,
    resolve_edition,
)
from app.models.llm import MarketReportDraft  # noqa: E402
from app.utils.flex_templates import build_market_report_flex_message  # noqa: E402


def _tpe(y: int, m: int, d: int, hh: int, mm: int = 0) -> datetime:
    return datetime(y, m, d, hh, mm, tzinfo=TAIPEI_TZ)


def _idx(symbol: str, data_date: str) -> MarketIndexItem:
    return MarketIndexItem(
        symbol=symbol, name=symbol, price=100.0, change=1.0, change_percent=1.0,
        data_date=data_date,
    )


# 2026-11-17 (二) 08:00：台股最新為 11/16 收盤，美股最新為 11/16 (美東) 收盤
PRE_INDICES = [_idx("^TWII", "2026-11-16"), _idx("^GSPC", "2026-11-16")]
# 2026-11-17 (二) 14:00：台股已有 11/17 收盤
POST_INDICES = [_idx("^TWII", "2026-11-17"), _idx("^GSPC", "2026-11-16")]


def test_describe_data_time_marks_close_and_intraday():
    """收盤後標「收盤」；同日未收盤標「盤中」，避免盤中價被當成收盤價"""
    assert describe_data_time(_idx("^TWII", "2026-11-16"), _tpe(2026, 11, 17, 8)) == "11/16 收盤"
    assert describe_data_time(_idx("^TWII", "2026-11-17"), _tpe(2026, 11, 17, 10)) == "11/17 盤中"
    assert describe_data_time(_idx("^TWII", "2026-11-17"), _tpe(2026, 11, 17, 14)) == "11/17 收盤"
    # 台北 11/17 23:00 = 美東 11/17 10:00，美股盤中
    assert describe_data_time(_idx("^GSPC", "2026-11-17"), _tpe(2026, 11, 17, 23)) == "11/17 盤中"
    assert describe_data_time(_idx("^GSPC", "2026-11-16"), _tpe(2026, 11, 17, 8)) == "11/16 收盤"


def test_build_data_basis():
    assert build_data_basis(PRE_INDICES, _tpe(2026, 11, 17, 8)) == "台股 11/16 收盤・美股 11/16 收盤"
    assert build_data_basis([], _tpe(2026, 11, 17, 8)) == "行情資料取得失敗"


def test_resolve_edition():
    # 指定版別
    assert resolve_edition("pre", PRE_INDICES, _tpe(2026, 11, 17, 8)) == "pre"
    assert resolve_edition("post", POST_INDICES, _tpe(2026, 11, 17, 14)) == "post"
    # 尚未收盤時指定盤後 → 略過
    with pytest.raises(MarketNotTradedToday):
        resolve_edition("post", POST_INDICES, _tpe(2026, 11, 17, 10))
    # 交易日 (11/17) 14:00 行情仍停在前一日 → 資料延遲，須告警重試而非略過
    with pytest.raises(MarketDataUnavailable):
        resolve_edition("post", PRE_INDICES, _tpe(2026, 11, 17, 14))
    # 真正休市 (2026-10-09 國慶補假，週五) → 略過
    holiday = [_idx("^TWII", "2026-10-08"), _idx("^GSPC", "2026-10-08")]
    with pytest.raises(MarketNotTradedToday):
        resolve_edition("post", holiday, _tpe(2026, 10, 9, 14))
    # 抓不到台股資料不可誤判為休市
    with pytest.raises(MarketDataUnavailable):
        resolve_edition("post", [_idx("^GSPC", "2026-11-16")], _tpe(2026, 11, 17, 14))
    with pytest.raises(MarketDataUnavailable):
        resolve_edition("post", [_idx("^TWII", None)], _tpe(2026, 11, 17, 14))
    # 自動：收盤前或今日無資料皆為盤前，今日已收盤才是盤後
    assert resolve_edition(None, PRE_INDICES, _tpe(2026, 11, 17, 8)) == "pre"
    assert resolve_edition(None, POST_INDICES, _tpe(2026, 11, 17, 10)) == "pre"
    assert resolve_edition(None, POST_INDICES, _tpe(2026, 11, 17, 14)) == "post"
    assert resolve_edition(None, PRE_INDICES, _tpe(2026, 11, 22, 15)) == "pre"


def _draft() -> MarketReportDraft:
    return MarketReportDraft(
        title="t", market_sentiment="中立震盪", index_summary="s",
        key_headlines=["h"], portfolio_insight="p", actionable_takeaway="a",
    )


def _service(indices):
    portfolio_repo = AsyncMock()
    portfolio_repo.get_all = AsyncMock(return_value=[])
    news_service = AsyncMock()
    news_service.fetch_google_news_rss = AsyncMock(return_value=[])
    svc = MarketReportService(portfolio_repo, news_service)
    svc.fetch_market_indices = MagicMock(return_value=indices)
    svc._call_gemini_report = MagicMock(return_value=_draft())
    svc._get_client = MagicMock()
    return svc


@pytest.mark.asyncio
async def test_generate_pre_market_report_labels_past_data():
    """盤前版：版別與資料基準由代碼填入，prompt 明示台股為前一交易日資料"""
    svc = _service(PRE_INDICES)
    with patch("app.services.market_report_service._taipei_now", return_value=_tpe(2026, 11, 17, 8)):
        report, _ = await svc.generate_market_report(edition="pre")

    assert report.edition == "盤前快報"
    assert report.data_basis == "台股 11/16 收盤・美股 11/16 收盤"
    assert report.date_str == "2026-11-17 08:00"
    prompt = svc._call_gemini_report.call_args[0][1]
    assert "盤前快報" in prompt
    assert "日期不是今天 (11/17) 就不得說成今日走勢" in prompt
    assert "資料時點：11/16 收盤" in prompt


@pytest.mark.asyncio
async def test_prompt_has_no_hardcoded_time_claims_contradicting_labels():
    """盤中 (10:00) 與美股盤中 (23:00) 查詢時，prompt 不得寫死與標示矛盾的時點"""
    intraday = [_idx("^TWII", "2026-11-17"), _idx("^GSPC", "2026-11-16")]
    svc = _service(intraday)
    with patch("app.services.market_report_service._taipei_now", return_value=_tpe(2026, 11, 17, 10)):
        report, _ = await svc.generate_market_report()
    prompt = svc._call_gemini_report.call_args[0][1]
    assert report.edition == "盤前快報"
    assert "資料時點：11/17 盤中" in prompt
    assert "前一交易日收盤" not in prompt

    us_open = [_idx("^TWII", "2026-11-17"), _idx("^GSPC", "2026-11-17")]
    svc = _service(us_open)
    with patch("app.services.market_report_service._taipei_now", return_value=_tpe(2026, 11, 17, 23)):
        report, _ = await svc.generate_market_report()
    prompt = svc._call_gemini_report.call_args[0][1]
    assert report.edition == "盤後快報"
    assert "^GSPC (^GSPC): 100.0 (+1.00%)｜資料時點：11/17 盤中" in prompt
    assert "前一晚收盤" not in prompt


def test_fetch_market_indices_takes_exchange_local_date_of_last_valid_close():
    """data_date 取 dropna 後最後一列的交易所當地日期；最後一列 Close 為 NaN 時取前一列"""
    import pandas as pd

    def frame(tz: str, nan_last: bool) -> pd.DataFrame:
        idx = pd.DatetimeIndex(
            ["2026-11-13", "2026-11-16", "2026-11-17"]
        ).tz_localize(tz)
        closes = [100.0, 101.0, float("nan") if nan_last else 102.0]
        return pd.DataFrame({"Open": closes, "Close": closes}, index=idx)

    frames = {
        "^TWII": frame("Asia/Taipei", nan_last=True),
        "^GSPC": frame("America/New_York", nan_last=False),
        "^SOX": frame("America/New_York", nan_last=False),
        "2330.TW": frame("Asia/Taipei", nan_last=False),
    }

    def fake_ticker(symbol: str) -> MagicMock:
        t = MagicMock()
        t.history.return_value = frames[symbol]
        return t

    svc = MarketReportService(MagicMock(), MagicMock())
    with patch("app.services.market_report_service.yf.Ticker", side_effect=fake_ticker):
        items = {i.symbol: i for i in svc.fetch_market_indices()}

    assert items["^TWII"].data_date == "2026-11-16"
    assert items["^TWII"].price == 101.0
    assert items["^GSPC"].data_date == "2026-11-17"


@pytest.mark.asyncio
async def test_generate_post_market_report():
    svc = _service(POST_INDICES)
    with patch("app.services.market_report_service._taipei_now", return_value=_tpe(2026, 11, 17, 14)):
        report, _ = await svc.generate_market_report(edition="post")

    assert report.edition == "盤後快報"
    assert report.data_basis == "台股 11/17 收盤・美股 11/16 收盤"
    assert "以今日台股盤勢為分析主軸" in svc._call_gemini_report.call_args[0][1]


@pytest.mark.asyncio
async def test_generate_post_market_report_skips_holiday_without_llm():
    """休市日不呼叫 LLM，直接擲出 MarketNotTradedToday"""
    svc = _service([_idx("^TWII", "2026-10-08"), _idx("^GSPC", "2026-10-08")])
    with patch("app.services.market_report_service._taipei_now", return_value=_tpe(2026, 10, 9, 14)):
        with pytest.raises(MarketNotTradedToday):
            await svc.generate_market_report(edition="post")
    svc._call_gemini_report.assert_not_called()


def test_tw_trading_calendar():
    from datetime import date
    from app.services.market_report_service import is_tw_trading_day

    assert is_tw_trading_day(date(2026, 11, 17)) is True   # 一般交易日
    assert is_tw_trading_day(date(2026, 10, 9)) is False   # 國慶補假
    assert is_tw_trading_day(date(2026, 11, 21)) is False  # 週六


def test_calendar_unavailable_falls_back_to_skip():
    """日曆不可用時退回舊行為 (視為休市略過)，不讓整個盤後流程崩潰"""
    with patch(
        "app.services.market_report_service._tw_calendar",
        side_effect=ImportError("no exchange_calendars"),
    ):
        with pytest.raises(MarketNotTradedToday):
            resolve_edition("post", PRE_INDICES, _tpe(2026, 11, 17, 14))


@pytest.mark.asyncio
async def test_scheduled_post_market_report_skips_on_holiday():
    line_service = MagicMock()
    line_service.push_market_report = AsyncMock()
    market_report_service = MagicMock()
    market_report_service.generate_market_report = AsyncMock(side_effect=MarketNotTradedToday("休市"))
    scheduler = SchedulerService(
        sell_engine=MagicMock(), buy_engine=MagicMock(), line_service=line_service,
        market_report_service=market_report_service,
    )

    await scheduler.scheduled_market_report(edition="post")

    market_report_service.generate_market_report.assert_awaited_once_with(edition="post")
    line_service.push_market_report.assert_not_awaited()


@pytest.mark.asyncio
async def test_scheduled_post_market_report_logs_error_when_data_unavailable():
    """抓不到台股資料須記 error，不可當成休市只記 info"""
    line_service = MagicMock()
    line_service.push_market_report = AsyncMock()
    market_report_service = MagicMock()
    market_report_service.generate_market_report = AsyncMock(
        side_effect=MarketDataUnavailable("無資料")
    )
    scheduler = SchedulerService(
        sell_engine=MagicMock(), buy_engine=MagicMock(), line_service=line_service,
        market_report_service=market_report_service,
    )
    with patch("app.services.scheduler_service.logger") as log:
        await scheduler.scheduled_market_report(edition="post")
    log.error.assert_called_once()
    line_service.push_market_report.assert_not_awaited()


def _api_client(market_report_service: MagicMock):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.controllers.api_controller import router, verify_admin

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[verify_admin] = lambda: None
    app.state.market_report_service = market_report_service
    app.state.line_service = MagicMock()
    return TestClient(app)


def test_trigger_endpoint_post_edition_on_holiday_returns_skipped():
    """休市回 200 + skipped，避免 Cloud Scheduler 記為失敗而重試"""
    svc = MagicMock()
    svc.generate_market_report = AsyncMock(side_effect=MarketNotTradedToday("休市"))
    resp = _api_client(svc).post("/api/trigger/market-report?push=true&edition=post")
    assert resp.status_code == 200
    assert resp.json()["skipped"] is True
    assert resp.json()["pushed"] is False
    svc.generate_market_report.assert_awaited_once_with(edition="post")


def test_trigger_endpoint_data_unavailable_returns_503():
    svc = MagicMock()
    svc.generate_market_report = AsyncMock(side_effect=MarketDataUnavailable("無資料"))
    resp = _api_client(svc).post("/api/trigger/market-report?edition=post")
    assert resp.status_code == 503


def test_trigger_endpoint_rejects_invalid_edition():
    svc = MagicMock()
    svc.generate_market_report = AsyncMock()
    resp = _api_client(svc).post("/api/trigger/market-report?edition=noon")
    assert resp.status_code == 422
    svc.generate_market_report.assert_not_called()


def test_scheduler_registers_post_market_report_job():
    """14:00 週一至週五註冊盤後快報，並以 edition=post 呼叫"""
    scheduler = SchedulerService(
        sell_engine=MagicMock(), buy_engine=MagicMock(), line_service=MagicMock(),
        market_report_service=MagicMock(),
    )
    with patch.object(scheduler.scheduler, "start"):
        scheduler.start()

    job = scheduler.scheduler.get_job("afternoon_market_report_job")
    assert job is not None
    assert job.kwargs == {"edition": "post"}
    fields = {f.name: str(f) for f in job.trigger.fields}
    assert fields["hour"] == "14"
    assert fields["minute"] == "0"
    assert fields["day_of_week"] == "mon-fri"
    scheduler.shutdown()


def test_flex_message_shows_edition_and_data_basis(mock_indices):
    report = MarketReport(
        **_draft().model_dump(), date_str="2026-11-17 08:00",
        edition="盤前快報", data_basis="台股 11/16 收盤・美股 11/16 收盤",
    )
    flex = build_market_report_flex_message(report, mock_indices)
    texts = str(flex["header"])
    assert "📑 AI 盤前快報" in texts
    assert "資料基準：台股 11/16 收盤・美股 11/16 收盤" in texts
