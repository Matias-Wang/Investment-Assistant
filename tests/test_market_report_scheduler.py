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

    market_report_service.generate_market_report.assert_awaited_once_with("us", strict=True)
    line_service.push_market_report.assert_awaited_once_with(mock_market_report, mock_indices, market="us")


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
    assert job.kwargs == {"market": "us"}
    assert str(job.trigger.fields[5]) == "8"  # hour

    tw_job = scheduler.scheduler.get_job("tw_market_report_job")
    assert tw_job is not None
    assert tw_job.kwargs == {"market": "tw"}
    assert str(tw_job.trigger.fields[5]) == "14"  # hour
    assert str(tw_job.trigger.fields[4]) == "mon-fri"  # day_of_week
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
    assert "市場快報" in messages[0].alt_text


@pytest.mark.asyncio
@pytest.mark.parametrize("market,label", [("us", "美股晨報"), ("tw", "台股盤後快報")])
async def test_push_market_report_uses_market_label(mock_market_report, mock_indices, market, label):
    service = LineBotService(MagicMock(), MagicMock(), MagicMock())
    service.push_message = MagicMock()
    await service.push_market_report(mock_market_report, mock_indices, target_user_id="U1", market=market)
    msg = service.push_message.call_args[0][1][0]
    assert label in msg.alt_text


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

    result = await trigger_market_report(mock_request, push=True, market="tw")
    assert result["pushed"] is True
    assert result["market"] == "tw"
    market_report_service.generate_market_report.assert_awaited_once_with("tw", strict=True)
    # 端點新增 user_id 參數以支援 Cloud Tasks 指定推播對象
    line_service.push_market_report.assert_awaited_once_with(
        mock_market_report, mock_indices, target_user_id=None, market="tw"
    )


@pytest.mark.asyncio
async def test_trigger_market_report_rejects_unknown_market():
    from fastapi import HTTPException
    from app.controllers.api_controller import trigger_market_report

    mock_request = MagicMock()
    with pytest.raises(HTTPException) as exc:
        await trigger_market_report(mock_request, market="jp")
    assert exc.value.status_code == 422


def test_resolve_market_by_taipei_time():
    from datetime import datetime, timezone
    import app.services.market_report_service as mrs

    class FakeDT(datetime):
        fixed = None

        @classmethod
        def now(cls, tz=None):
            return cls.fixed

    with patch.object(mrs, "datetime", FakeDT):
        FakeDT.fixed = datetime(2026, 11, 18, 0, 0, tzinfo=timezone.utc)  # 台北 08:00
        assert mrs.resolve_market() == "us"
        FakeDT.fixed = datetime(2026, 11, 18, 6, 0, tzinfo=timezone.utc)  # 台北 14:00
        assert mrs.resolve_market() == "tw"
    assert mrs.resolve_market("TW") == "tw"


def test_market_profiles_are_separated():
    """美股晨報只抓美股指數，台股盤後只抓台股標的"""
    from app.services.market_report_service import MARKET_PROFILES

    us_symbols = [s for s, _ in MARKET_PROFILES["us"].index_targets]
    tw_symbols = [s for s, _ in MARKET_PROFILES["tw"].index_targets]
    assert "^TWII" not in us_symbols and not any(s.endswith(".TW") for s in us_symbols)
    assert all(s == "^TWII" or s.endswith(".TW") for s in tw_symbols)


# ---------------------------------------------------------------------------
# 資料時點標示與台股休市／資料延遲判斷
# ---------------------------------------------------------------------------
from datetime import datetime  # noqa: E402

from app.services.market_report_service import (  # noqa: E402
    TAIPEI_TZ,
    MarketDataUnavailable,
    MarketNotTradedToday,
    MarketReportService,
    build_data_basis,
    describe_data_time,
    ensure_tw_closed_today,
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


# 2026-11-17 (二) 08:00：美股最新為 11/16 (美東) 收盤
US_INDICES = [_idx("^GSPC", "2026-11-16")]
# 2026-11-17 (二) 14:00：台股已有 11/17 收盤
TW_TODAY = [_idx("^TWII", "2026-11-17")]
# 台股行情仍停在前一交易日
TW_YESTERDAY = [_idx("^TWII", "2026-11-16")]


def test_describe_data_time_marks_close_and_intraday():
    """收盤後標「收盤」；同日未收盤標「盤中」，避免盤中價被當成收盤價"""
    assert describe_data_time(_idx("^TWII", "2026-11-16"), _tpe(2026, 11, 17, 8)) == "11/16 收盤"
    assert describe_data_time(_idx("^TWII", "2026-11-17"), _tpe(2026, 11, 17, 10)) == "11/17 盤中"
    assert describe_data_time(_idx("^TWII", "2026-11-17"), _tpe(2026, 11, 17, 14)) == "11/17 收盤"
    # 台北 11/17 23:00 = 美東 11/17 10:00，美股盤中
    assert describe_data_time(_idx("^GSPC", "2026-11-17"), _tpe(2026, 11, 17, 23)) == "11/17 盤中"
    assert describe_data_time(_idx("^GSPC", "2026-11-16"), _tpe(2026, 11, 17, 8)) == "11/16 收盤"


def test_build_data_basis():
    both = [_idx("^TWII", "2026-11-16"), _idx("^GSPC", "2026-11-16")]
    assert build_data_basis(both, _tpe(2026, 11, 17, 8)) == "台股 11/16 收盤・美股 11/16 收盤"
    assert build_data_basis(US_INDICES, _tpe(2026, 11, 17, 8)) == "美股 11/16 收盤"
    assert build_data_basis(TW_TODAY, _tpe(2026, 11, 17, 14)) == "台股 11/17 收盤"
    assert build_data_basis([], _tpe(2026, 11, 17, 8)) == "行情資料取得失敗"


def test_ensure_tw_closed_today():
    # 今日已收盤 → 放行
    ensure_tw_closed_today(TW_TODAY, _tpe(2026, 11, 17, 14))
    # 尚未收盤 → 略過
    with pytest.raises(MarketNotTradedToday):
        ensure_tw_closed_today(TW_TODAY, _tpe(2026, 11, 17, 10))
    # 交易日 (11/17) 14:00 行情仍停在前一日 → 資料延遲，須告警重試而非略過
    with pytest.raises(MarketDataUnavailable):
        ensure_tw_closed_today(TW_YESTERDAY, _tpe(2026, 11, 17, 14))
    # 真正休市 (2026-10-09 國慶補假，週五) → 略過
    with pytest.raises(MarketNotTradedToday):
        ensure_tw_closed_today([_idx("^TWII", "2026-10-08")], _tpe(2026, 10, 9, 14))
    # 抓不到台股資料不可誤判為休市
    with pytest.raises(MarketDataUnavailable):
        ensure_tw_closed_today([], _tpe(2026, 11, 17, 14))
    with pytest.raises(MarketDataUnavailable):
        ensure_tw_closed_today([_idx("^TWII", None)], _tpe(2026, 11, 17, 14))


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
async def test_generate_us_report_labels_past_data():
    """美股晨報：資料基準與產出時間由代碼填入，prompt 明示資料時點"""
    svc = _service(US_INDICES)
    with patch("app.services.market_report_service._taipei_now", return_value=_tpe(2026, 11, 17, 8)):
        report, _ = await svc.generate_market_report("us", strict=True)

    assert report.data_basis == "美股 11/16 收盤"
    assert report.date_str == "2026-11-17 08:00"
    prompt = svc._call_gemini_report.call_args[0][1]
    assert "美股晨報" in prompt
    assert "日期不是今天 (11/17) 就不得說成今日走勢" in prompt
    assert "資料時點：11/16 收盤" in prompt


@pytest.mark.asyncio
async def test_prompt_has_no_hardcoded_time_claims_contradicting_labels():
    """台股盤中 (10:00) 與美股盤中 (23:00) 查詢時，prompt 標示盤中而非寫死今日收盤"""
    svc = _service(TW_TODAY)
    with patch("app.services.market_report_service._taipei_now", return_value=_tpe(2026, 11, 17, 10)):
        await svc.generate_market_report("tw")
    prompt = svc._call_gemini_report.call_args[0][1]
    assert "資料時點：11/17 盤中" in prompt
    assert "今日台灣股市" not in prompt

    svc = _service([_idx("^GSPC", "2026-11-17")])
    with patch("app.services.market_report_service._taipei_now", return_value=_tpe(2026, 11, 17, 23)):
        await svc.generate_market_report("us")
    prompt = svc._call_gemini_report.call_args[0][1]
    assert "^GSPC (^GSPC): 100.0 (+1.00%)｜資料時點：11/17 盤中" in prompt


def test_fetch_market_indices_takes_exchange_local_date_of_last_valid_close():
    """data_date 取 dropna 後最後一列的交易所當地日期；最後一列 Close 為 NaN 時取前一列"""
    import pandas as pd

    def frame(tz: str, nan_last: bool) -> pd.DataFrame:
        idx = pd.DatetimeIndex(
            ["2026-11-13", "2026-11-16", "2026-11-17"]
        ).tz_localize(tz)
        closes = [100.0, 101.0, float("nan") if nan_last else 102.0]
        return pd.DataFrame({"Open": closes, "Close": closes}, index=idx)

    def fake_ticker(symbol: str) -> MagicMock:
        t = MagicMock()
        if symbol == "^TWII":
            t.history.return_value = frame("Asia/Taipei", nan_last=True)
        elif symbol.endswith(".TW"):
            t.history.return_value = frame("Asia/Taipei", nan_last=False)
        else:
            t.history.return_value = frame("America/New_York", nan_last=False)
        return t

    svc = MarketReportService(MagicMock(), MagicMock())
    with patch("app.services.market_report_service.yf.Ticker", side_effect=fake_ticker):
        tw = {i.symbol: i for i in svc.fetch_market_indices("tw")}
        us = {i.symbol: i for i in svc.fetch_market_indices("us")}

    assert tw["^TWII"].data_date == "2026-11-16"
    assert tw["^TWII"].price == 101.0
    assert us["^GSPC"].data_date == "2026-11-17"


@pytest.mark.asyncio
async def test_generate_tw_report_strict_after_close():
    svc = _service(TW_TODAY)
    with patch("app.services.market_report_service._taipei_now", return_value=_tpe(2026, 11, 17, 14)):
        report, _ = await svc.generate_market_report("tw", strict=True)

    assert report.data_basis == "台股 11/17 收盤"
    assert "台股盤後快報" in svc._call_gemini_report.call_args[0][1]


@pytest.mark.asyncio
async def test_strict_tw_report_skips_holiday_without_llm():
    """排程 (strict) 遇休市日不呼叫 LLM，直接擲出 MarketNotTradedToday"""
    svc = _service([_idx("^TWII", "2026-10-08")])
    with patch("app.services.market_report_service._taipei_now", return_value=_tpe(2026, 10, 9, 14)):
        with pytest.raises(MarketNotTradedToday):
            await svc.generate_market_report("tw", strict=True)
    svc._call_gemini_report.assert_not_called()


@pytest.mark.asyncio
async def test_manual_tw_report_on_holiday_still_generates_with_past_date():
    """LINE 手動查詢 (非 strict) 遇休市日照常產出，並標示前一交易日收盤"""
    svc = _service([_idx("^TWII", "2026-10-08")])
    with patch("app.services.market_report_service._taipei_now", return_value=_tpe(2026, 10, 9, 15)):
        report, _ = await svc.generate_market_report("tw")
    assert report.data_basis == "台股 10/08 收盤"


def test_tw_trading_calendar():
    from datetime import date
    from app.services.market_report_service import is_tw_trading_day

    assert is_tw_trading_day(date(2026, 11, 17)) is True   # 一般交易日
    assert is_tw_trading_day(date(2026, 10, 9)) is False   # 國慶補假
    assert is_tw_trading_day(date(2026, 11, 21)) is False  # 週六


def test_calendar_unavailable_falls_back_to_skip():
    """日曆不可用時退回視為休市略過，不讓整個盤後流程崩潰"""
    with patch(
        "app.services.market_report_service._tw_calendar",
        side_effect=ImportError("no exchange_calendars"),
    ):
        with pytest.raises(MarketNotTradedToday):
            ensure_tw_closed_today(TW_YESTERDAY, _tpe(2026, 11, 17, 14))


@pytest.mark.asyncio
async def test_scheduled_tw_report_skips_on_holiday():
    line_service = MagicMock()
    line_service.push_market_report = AsyncMock()
    market_report_service = MagicMock()
    market_report_service.generate_market_report = AsyncMock(side_effect=MarketNotTradedToday("休市"))
    scheduler = SchedulerService(
        sell_engine=MagicMock(), buy_engine=MagicMock(), line_service=line_service,
        market_report_service=market_report_service,
    )

    with patch("app.services.scheduler_service.logger") as log:
        await scheduler.scheduled_market_report(market="tw")

    market_report_service.generate_market_report.assert_awaited_once_with("tw", strict=True)
    line_service.push_market_report.assert_not_awaited()
    log.error.assert_not_called()


@pytest.mark.asyncio
async def test_scheduled_tw_report_logs_error_when_data_unavailable():
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
        await scheduler.scheduled_market_report(market="tw")
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


def test_trigger_endpoint_tw_on_holiday_returns_skipped():
    """休市回 200 + skipped，避免 Cloud Scheduler 記為失敗而重試"""
    svc = MagicMock()
    svc.generate_market_report = AsyncMock(side_effect=MarketNotTradedToday("休市"))
    resp = _api_client(svc).post("/api/trigger/market-report?push=true&market=tw")
    assert resp.status_code == 200
    assert resp.json()["skipped"] is True
    assert resp.json()["pushed"] is False
    svc.generate_market_report.assert_awaited_once_with("tw", strict=True)


def test_trigger_endpoint_data_unavailable_returns_503():
    svc = MagicMock()
    svc.generate_market_report = AsyncMock(side_effect=MarketDataUnavailable("無資料"))
    resp = _api_client(svc).post("/api/trigger/market-report?market=tw")
    assert resp.status_code == 503


def test_trigger_endpoint_without_market_is_not_strict(mock_market_report, mock_indices):
    """未指定 market (手動呼叫) 不做休市檢查"""
    svc = MagicMock()
    svc.generate_market_report = AsyncMock(return_value=(mock_market_report, mock_indices))
    resp = _api_client(svc).post("/api/trigger/market-report")
    assert resp.status_code == 200
    assert svc.generate_market_report.await_args.kwargs == {"strict": False}


def test_flex_message_shows_label_and_data_basis(mock_indices):
    report = MarketReport(
        **_draft().model_dump(), date_str="2026-11-17 08:00",
        data_basis="美股 11/16 收盤",
    )
    flex = build_market_report_flex_message(report, mock_indices, label="美股晨報")
    texts = str(flex["header"])
    assert "📑 AI 美股晨報" in texts
    assert "資料基準：美股 11/16 收盤" in texts
