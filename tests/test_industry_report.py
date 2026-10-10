import pytest
from decimal import Decimal
from unittest.mock import MagicMock, AsyncMock, patch
from apscheduler.triggers.cron import CronTrigger

from app.models.domain import IndustryPerformance, PortfolioItem
from app.models.llm import IndustryReport, IndustrySection
from app.services.industry_report_service import IndustryReportService
from app.services.line_bot_service import LineBotService
from app.services.scheduler_service import SchedulerService
from app.utils.flex_templates import build_industry_report_flex_message


def _item(ticker, name, asset_type="tw_stock"):
    return PortfolioItem(ticker=ticker, name=name, asset_type=asset_type,
                         shares=Decimal("1"), average_cost=Decimal("100"))


@pytest.fixture
def report():
    return IndustryReport(
        title="AI 供應鏈動能延續",
        date_str="2026-10-10 08:30",
        market_sentiment="樂觀偏多",
        index_summary="半導體強於 PCB。",
        industries=[
            IndustrySection(industry="半導體", outlook="偏多", summary="先進製程需求強。"),
            IndustrySection(industry="PCB 電路板", outlook="中性", summary="報價持平。"),
            IndustrySection(industry="光通訊", outlook="偏空", summary="庫存調整。"),
        ],
        key_headlines=["A", "B", "C"],
        portfolio_insight="整體偏多。",
        actionable_takeaway="續抱，勿追高。",
    )


@pytest.fixture
def performance():
    return [
        IndustryPerformance(industry="半導體", tickers=["2330"], avg_change_percent=1.2),
        IndustryPerformance(industry="PCB 電路板", tickers=["2313"], avg_change_percent=-0.5),
        IndustryPerformance(industry="光通訊", tickers=["6442"], avg_change_percent=0.0),
    ]


def test_group_by_industry_uses_static_map():
    svc = IndustryReportService(MagicMock(), MagicMock())
    groups = svc.group_by_industry([_item("2330", "台積電"), _item("2454", "聯發科"), _item("2313", "華通")])
    assert list(groups.keys()) == ["半導體", "PCB 電路板"]
    assert [i.ticker for i in groups["半導體"]] == ["2330", "2454"]


def test_classify_unknown_falls_back_to_yfinance_sector_then_other():
    svc = IndustryReportService(MagicMock(), MagicMock())
    with patch("app.services.industry_report_service.yf.Ticker") as t:
        t.return_value.info = {"sector": "Healthcare"}
        assert svc.classify_industry(_item("XYZ", "某藥廠", "us_stock")) == "醫療保健"
        t.return_value.info = {}
        assert svc.classify_industry(_item("XYZ", "某公司", "us_stock")) == "其他"


@pytest.mark.asyncio
async def test_generate_without_holdings_raises_and_skips_llm():
    repo = AsyncMock()
    repo.get_all = AsyncMock(return_value=[])
    svc = IndustryReportService(repo, MagicMock())
    svc._call_gemini_report = MagicMock()
    with pytest.raises(ValueError):
        await svc.generate_industry_report()
    svc._call_gemini_report.assert_not_called()


@pytest.mark.asyncio
async def test_generate_returns_report_and_performance(report):
    repo = AsyncMock()
    repo.get_all = AsyncMock(return_value=[_item("2330", "台積電")])
    news = MagicMock()
    news.fetch_google_news_rss = AsyncMock(side_effect=RuntimeError("503"))
    news.fetch_yfinance_news = MagicMock(return_value=[])
    svc = IndustryReportService(repo, news)
    svc._fetch_change_percent = MagicMock(return_value=2.0)
    svc._call_gemini_report = MagicMock(return_value=report)
    svc._get_client = MagicMock()

    got, perf = await svc.generate_industry_report()
    assert got.title == report.title
    assert perf[0].industry == "半導體" and perf[0].avg_change_percent == 2.0
    # RSS 失敗不得讓整份報告失敗；prompt 需含產業與持股
    prompt = svc._call_gemini_report.call_args[0][1]
    assert "半導體" in prompt and "2330 台積電" in prompt


def test_flex_handles_odd_number_of_industries(report, performance):
    flex = build_industry_report_flex_message(report, performance)
    assert flex["type"] == "bubble"
    grid = flex["body"]["contents"][0]["contents"]
    assert len(grid) == 2 and len(grid[1]["contents"]) == 2  # 3 個產業 → 2 列，末列補位


@pytest.mark.asyncio
async def test_push_industry_report(report, performance):
    svc = LineBotService(MagicMock(), MagicMock(), MagicMock())
    svc.push_message = MagicMock()
    await svc.push_industry_report(report, performance, target_user_id="U1")
    args, _ = svc.push_message.call_args
    assert args[0] == "U1" and "產業報告" in args[1][0].alt_text


@pytest.mark.asyncio
async def test_scheduler_industry_report_pushes_and_swallows_no_holdings(report, performance):
    line = MagicMock()
    line.push_industry_report = AsyncMock()
    isvc = MagicMock()
    isvc.generate_industry_report = AsyncMock(return_value=(report, performance))
    s = SchedulerService(MagicMock(), MagicMock(), line, industry_report_service=isvc)
    await s.scheduled_industry_report()
    line.push_industry_report.assert_awaited_once_with(report, performance)

    line.push_industry_report.reset_mock()
    isvc.generate_industry_report = AsyncMock(side_effect=ValueError("無持股"))
    await s.scheduled_industry_report()
    line.push_industry_report.assert_not_awaited()


def test_scheduler_registers_industry_job():
    s = SchedulerService(MagicMock(), MagicMock(), MagicMock(), industry_report_service=MagicMock())
    with patch.object(s.scheduler, "start"):
        s.start()
    job = s.scheduler.get_job("daily_industry_report_job")
    assert job is not None and isinstance(job.trigger, CronTrigger)
    s.shutdown()


# ---- 股價方向矛盾查核 (回歸：光聖 -1.65% 卻被寫成「股價強勢上攻」) ----------------

from app.services.industry_report_service import find_price_contradictions, strip_contradictions


def _report_with(summary_by_industry):
    return IndustryReport(
        title="t", date_str="d", market_sentiment="中立震盪", index_summary="s",
        industries=[IndustrySection(industry=k, outlook="偏多", summary=v) for k, v in summary_by_industry.items()],
        key_headlines=["a"], portfolio_insight="p", actionable_takeaway="x",
    )


_PERF = [
    IndustryPerformance(industry="光通訊", tickers=["6442"], avg_change_percent=-1.65),
    IndustryPerformance(industry="PCB 電路板", tickers=["2313"], avg_change_percent=0.81),
]
_NAMES = {"光通訊": ["光聖"], "PCB 電路板": ["華通"]}


def test_detects_up_wording_when_price_fell():
    r = _report_with({"光通訊": "光聖Q3營收創歷史新高。帶動股價強勢上攻", "PCB 電路板": "華通營收創高"})
    issues = find_price_contradictions(r, _PERF, _NAMES)
    assert [(i, "上攻" in c) for i, c in issues] == [(0, True)]


def test_detects_down_wording_when_price_rose():
    r = _report_with({"光通訊": "庫存調整", "PCB 電路板": "華通股價回檔整理"})
    assert [i for i, _ in find_price_contradictions(r, _PERF, _NAMES)] == [1]


def test_no_false_positive_on_revenue_or_consistent_direction():
    r = _report_with({"光通訊": "光聖營收創歷史新高，股價回檔但基本面不變", "PCB 電路板": "華通營收創高，受惠升級潮"})
    assert find_price_contradictions(r, _PERF, _NAMES) == []


def test_flat_band_skips_direction_check():
    perf = [IndustryPerformance(industry="光通訊", tickers=["6442"], avg_change_percent=-0.1)]
    r = _report_with({"光通訊": "光聖股價上攻"})
    assert find_price_contradictions(r, perf, _NAMES) == []


def test_strip_removes_only_bad_clause():
    r = _report_with({"光通訊": "光聖營收創高。帶動股價強勢上攻", "PCB 電路板": "ok"})
    fixed = strip_contradictions(r, find_price_contradictions(r, _PERF, _NAMES))
    assert fixed.industries[0].summary == "光聖營收創高"
    assert "上攻" not in fixed.industries[0].summary


@pytest.mark.asyncio
async def test_generate_retries_then_falls_back_to_strip():
    repo = AsyncMock()
    repo.get_all = AsyncMock(return_value=[_item("6442", "光聖")])
    news = MagicMock()
    news.fetch_google_news_rss = AsyncMock(return_value=[])
    news.fetch_yfinance_news = MagicMock(return_value=[])
    svc = IndustryReportService(repo, news)
    svc._fetch_change_percent = MagicMock(return_value=-1.65)
    svc._get_client = MagicMock()
    bad = _report_with({"光通訊": "光聖營收創高。帶動股價強勢上攻"})
    svc._call_gemini_report = MagicMock(return_value=bad)  # 重試仍犯同樣的錯

    got, _ = await svc.generate_industry_report()
    assert svc._call_gemini_report.call_count == 2
    assert "股價強勢上攻" not in got.industries[0].summary
    assert "【上一版的錯誤" in svc._call_gemini_report.call_args_list[1][0][1]


@pytest.mark.asyncio
async def test_generate_retry_success_keeps_corrected_report():
    repo = AsyncMock()
    repo.get_all = AsyncMock(return_value=[_item("6442", "光聖")])
    news = MagicMock()
    news.fetch_google_news_rss = AsyncMock(return_value=[])
    news.fetch_yfinance_news = MagicMock(return_value=[])
    svc = IndustryReportService(repo, news)
    svc._fetch_change_percent = MagicMock(return_value=-1.65)
    svc._get_client = MagicMock()
    good = _report_with({"光通訊": "光聖營收創高，與今日股價表現相左"})
    bad = _report_with({"光通訊": "光聖股價強勢上攻"})
    svc._call_gemini_report = MagicMock(side_effect=[bad, good])

    got, _ = await svc.generate_industry_report()
    assert got.industries[0].summary == "光聖營收創高，與今日股價表現相左"


# ---- 代碼計算欄位 (氛圍 / 強弱對比 / 日期不由 LLM 決定) --------------------------

from app.services.industry_report_service import compute_sentiment, build_strength_summary


def _p(*pcts):
    return [IndustryPerformance(industry=f"I{i}", tickers=[str(i)], avg_change_percent=v) for i, v in enumerate(pcts)]


@pytest.mark.parametrize("pcts,expected", [
    ((1.0, 0.4), "樂觀偏多"),        # 平均 +0.7
    ((0.4, -0.4), "中立震盪"),       # 平均 0
    ((-0.5, -0.5), "保守觀望"),      # 平均 -0.5 (邊界歸入保守)
    ((-2.0, -1.5), "空方承壓"),      # 平均 -1.75
    ((), "中立震盪"),
])
def test_compute_sentiment_thresholds(pcts, expected):
    assert compute_sentiment(_p(*pcts)) == expected


def test_strength_summary_uses_actual_numbers():
    s = build_strength_summary(_p(2.0, -1.0, 0.5))
    assert "最強：I0 +2.00%" in s and "最弱：I1 -1.00%" in s and "2 漲 1 跌" in s


@pytest.mark.asyncio
async def test_final_report_overrides_llm_computed_fields():
    repo = AsyncMock()
    repo.get_all = AsyncMock(return_value=[_item("6442", "光聖")])
    news = MagicMock()
    news.fetch_google_news_rss = AsyncMock(return_value=[])
    news.fetch_yfinance_news = MagicMock(return_value=[])
    svc = IndustryReportService(repo, news)
    svc._fetch_change_percent = MagicMock(return_value=-2.0)
    svc._get_client = MagicMock()
    # 即使 (舊式) 回傳物件帶有錯誤的氛圍/日期/強弱對比，最終報告也必須以代碼為準
    llm = _report_with({"光通訊": "光聖營收創高"})
    llm.market_sentiment, llm.date_str, llm.index_summary = "樂觀偏多", "1999-01-01", "LLM 亂寫"
    svc._call_gemini_report = MagicMock(return_value=llm)

    got, _ = await svc.generate_industry_report()
    assert got.market_sentiment == "空方承壓"
    assert got.date_str != "1999-01-01"
    assert got.index_summary != "LLM 亂寫" and "-2.00%" in got.index_summary


# ---- LINE 觸發路徑 (回歸：固定指令分支曾遺失，雲端輸入「產業報告」無反應) ----------------

from app.services.nlu_rules import RuleBasedNLU
from app.models.slm import NLUIntent


def _line_event(text, user_id="U123456"):
    ev = MagicMock()
    ev.reply_token = "tok"
    ev.source.user_id = user_id
    ev.message.text = text
    return ev


def _line_service():
    cfg, pf, sess = AsyncMock(), AsyncMock(), AsyncMock()
    sess.get_session.return_value = None
    svc = LineBotService(cfg, pf, sess)
    svc.reply_message = MagicMock()
    svc.push_message = MagicMock()
    svc.task_dispatcher = MagicMock()
    svc.task_dispatcher.dispatch = AsyncMock()
    return svc


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["產業報告", "產業快報", "產業分析", "給我產業快報", "我的產業狀況怎麼樣"])
async def test_line_text_triggers_industry_report_not_market(text, report, performance):
    svc = _line_service()
    market = AsyncMock()
    industry = MagicMock()
    industry.generate_industry_report = AsyncMock(return_value=(report, performance))
    with patch("app.core.config.settings.ALLOWED_LINE_USER_IDS", ["U123456"]):
        await svc._handle_text_message(
            _line_event(text), {"industry_report_service": industry, "market_report_service": market}
        )
    job = svc.task_dispatcher.dispatch.call_args
    assert job[0][0] == "industry-report"          # 派送的是產業報告，不是市場報告
    assert "產業報告生成中" in svc.reply_message.call_args[0][1][0].text
    market.generate_market_report.assert_not_called()


@pytest.mark.asyncio
async def test_line_industry_fallback_pushes_card_and_reports_no_holdings(report, performance):
    svc = _line_service()
    svc.push_industry_report = AsyncMock()
    industry = MagicMock()
    industry.generate_industry_report = AsyncMock(return_value=(report, performance))
    with patch("app.core.config.settings.ALLOWED_LINE_USER_IDS", ["U123456"]):
        await svc._handle_text_message(_line_event("產業報告"), {"industry_report_service": industry})
        await svc.task_dispatcher.dispatch.call_args.kwargs["fallback"]()
    svc.push_industry_report.assert_awaited_once_with(report, performance, target_user_id="U123456")

    industry.generate_industry_report = AsyncMock(side_effect=ValueError("目前沒有持股，無法產生產業報告。"))
    with patch("app.core.config.settings.ALLOWED_LINE_USER_IDS", ["U123456"]):
        await svc._handle_text_message(_line_event("產業報告"), {"industry_report_service": industry})
        await svc.task_dispatcher.dispatch.call_args.kwargs["fallback"]()
    assert "沒有持股" in svc.push_message.call_args[0][1][0].text


def test_market_phrases_still_route_to_market_report():
    nlu = RuleBasedNLU()
    assert nlu.parse("今天大盤怎樣").intent == NLUIntent.MARKET_REPORT
    assert nlu.parse("市場快報").intent == NLUIntent.MARKET_REPORT
    assert nlu.parse("產業快報").intent == NLUIntent.INDUSTRY_REPORT


def test_menu_quickreply_and_help_expose_industry_report():
    import json
    from app.utils.flex_templates import build_menu_flex_message
    assert "產業報告" in json.dumps(build_menu_flex_message(), ensure_ascii=False)
    items = _line_service()._get_default_quick_reply().items
    assert any(i.action.text == "產業報告" for i in items)
