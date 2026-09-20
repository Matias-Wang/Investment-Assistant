import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from app.models.slm import NLUIntent, NLUIntentResult
from app.models.domain import SystemConfig, PortfolioValuation
from app.services.line_bot_service import LineBotService
from app.services.slm_service import SLMService
from app.core.config import settings


@pytest.fixture
def mock_repos():
    config_repo = AsyncMock()
    portfolio_repo = AsyncMock()
    session_repo = AsyncMock()

    config = SystemConfig(trade_budget_per_order=100000, take_profit_min=0.1, take_profit_max=0.2, stop_loss=-0.5)
    config_repo.get_config.return_value = config
    config_repo.update_field.return_value = config
    session_repo.get_session.return_value = None

    return config_repo, portfolio_repo, session_repo


def make_message_event(text: str, user_id: str = "U123456"):
    event = MagicMock()
    event.reply_token = "reply_token_123"
    event.source = MagicMock()
    event.source.user_id = user_id
    event.message = MagicMock()
    event.message.text = text
    return event


@pytest.mark.asyncio
async def test_line_bot_slm_natural_buy(mock_repos):
    config_repo, portfolio_repo, session_repo = mock_repos
    slm_service = AsyncMock(spec=SLMService)

    # 模擬 SLM 辨識出自然語言買入
    slm_service.parse_intent.return_value = NLUIntentResult(
        intent=NLUIntent.RECORD_BUY,
        confidence=0.96,
        ticker="2330",
        price=1050.0,
        shares=8.0,
        raw_text="剛在 1050 買進 8 股台積電",
    )

    service = LineBotService(config_repo, portfolio_repo, session_repo, slm_service=slm_service)
    service.reply_message = MagicMock()

    with patch("app.core.config.settings.SLM_ENABLED", True), \
         patch("app.core.config.settings.ALLOWED_LINE_USER_IDS", ["U123456"]):

        event = make_message_event("剛在 1050 買進 8 股台積電")
        await service._handle_text_message(event, {})

        # 驗證 portfolio_repo 成功調用買入
        portfolio_repo.record_buy.assert_awaited_once_with(ticker="2330", shares=8.0, price=1050.0)
        assert service.reply_message.called
        reply_call_args = service.reply_message.call_args[0]
        assert "AI 語意記帳成功" in reply_call_args[1][0].text


@pytest.mark.asyncio
async def test_line_bot_slm_natural_update_budget(mock_repos):
    config_repo, portfolio_repo, session_repo = mock_repos
    slm_service = AsyncMock(spec=SLMService)

    slm_service.parse_intent.return_value = NLUIntentResult(
        intent=NLUIntent.UPDATE_CONFIG,
        confidence=0.95,
        config_field="budget",
        config_value=120000.0,
        raw_text="幫我把單次預算改為12萬",
    )

    service = LineBotService(config_repo, portfolio_repo, session_repo, slm_service=slm_service)
    service.reply_message = MagicMock()

    with patch("app.core.config.settings.SLM_ENABLED", True), \
         patch("app.core.config.settings.ALLOWED_LINE_USER_IDS", ["U123456"]):

        event = make_message_event("幫我把單次預算改為12萬")
        await service._handle_text_message(event, {})

        config_repo.update_field.assert_awaited_once_with("trade_budget_per_order", 120000.0)
        assert service.reply_message.called
        reply_text = service.reply_message.call_args[0][1][0].text
        assert "AI 設定更新" in reply_text


@pytest.mark.asyncio
async def test_line_bot_slm_natural_query_portfolio(mock_repos):
    config_repo, portfolio_repo, session_repo = mock_repos
    slm_service = AsyncMock(spec=SLMService)

    slm_service.parse_intent.return_value = NLUIntentResult(
        intent=NLUIntent.QUERY_PORTFOLIO,
        confidence=0.92,
        raw_text="看一下現在損益如何",
    )

    sell_engine = AsyncMock()
    valuation = PortfolioValuation(total_cost=10000, total_market_value=11000, total_pnl=1000, total_pnl_rate=0.1, items=[])
    sell_engine.get_portfolio_valuation.return_value = (valuation, SystemConfig())

    service = LineBotService(config_repo, portfolio_repo, session_repo, slm_service=slm_service)
    service.reply_message = MagicMock()

    with patch("app.core.config.settings.SLM_ENABLED", True), \
         patch("app.core.config.settings.ALLOWED_LINE_USER_IDS", ["U123456"]):

        event = make_message_event("看一下現在損益如何")
        await service._handle_text_message(event, {"sell_engine": sell_engine})

        assert sell_engine.get_portfolio_valuation.called
        assert service.reply_message.called


@pytest.mark.asyncio
async def test_line_bot_slm_disabled_fallback(mock_repos):
    config_repo, portfolio_repo, session_repo = mock_repos
    slm_service = AsyncMock(spec=SLMService)

    service = LineBotService(config_repo, portfolio_repo, session_repo, slm_service=slm_service)
    service.reply_message = MagicMock()

    # SLM_ENABLED=False
    with patch("app.core.config.settings.SLM_ENABLED", False), \
         patch("app.core.config.settings.ALLOWED_LINE_USER_IDS", ["U123456"]):

        event = make_message_event("隨便說一句話")
        await service._handle_text_message(event, {})

        # SLM 未啟用時不會調用 parse_intent，且回退至預設說明訊息
        assert not slm_service.parse_intent.called
        assert service.reply_message.called
        assert "收到訊息：「隨便說一句話」" in service.reply_message.call_args[0][1][0].text


@pytest.mark.asyncio
async def test_rule_layer_intercepts_before_slm(mock_repos):
    """P0-6：規則層必須攔在 SLM 之前，否則口語訊息仍會付費走模型。"""
    config_repo, portfolio_repo, session_repo = mock_repos
    slm_service = AsyncMock(spec=SLMService)

    service = LineBotService(config_repo, portfolio_repo, session_repo, slm_service=slm_service)
    service.reply_message = MagicMock()

    # 僅覆寫需要的設定；整個 settings 換成 mock 會讓 fail-closed 白名單誤判為未授權
    with patch.object(settings, "SLM_ENABLED", True),          patch.object(settings, "ALLOWED_LINE_USER_IDS", ["U123456"]):
        await service._handle_text_message(
            make_message_event("剛在 1050 買進 8 股台積電"), {}
        )

    # 規則層已處理完畢，SLM 不應被呼叫
    slm_service.parse_intent.assert_not_awaited()
    portfolio_repo.record_buy.assert_awaited_once()
    kwargs = portfolio_repo.record_buy.await_args.kwargs
    assert kwargs["ticker"] == "2330"
    assert kwargs["price"] == 1050.0
    assert kwargs["shares"] == 8.0


@pytest.mark.asyncio
async def test_slm_still_used_when_rules_cannot_parse(mock_repos):
    """規則層無法辨識時才降級呼叫 SLM，兩層的分工不可顛倒。"""
    config_repo, portfolio_repo, session_repo = mock_repos
    slm_service = AsyncMock(spec=SLMService)
    slm_service.parse_intent.return_value = NLUIntentResult(
        intent=NLUIntent.QUERY_PORTFOLIO, confidence=0.9, raw_text="幫我看看那個東西"
    )

    sell_engine = AsyncMock()
    sell_engine.get_portfolio_valuation.return_value = (
        PortfolioValuation(items=[], total_cost=0.0, total_market_value=0.0,
                           total_unrealized_pnl=0.0, total_pnl_rate=0.0),
        SystemConfig(),
    )

    service = LineBotService(config_repo, portfolio_repo, session_repo, slm_service=slm_service)
    service.reply_message = MagicMock()

    with patch.object(settings, "SLM_ENABLED", True),          patch.object(settings, "ALLOWED_LINE_USER_IDS", ["U123456"]):
        await service._handle_text_message(
            make_message_event("幫我看看那個東西"), {"sell_engine": sell_engine}
        )

    slm_service.parse_intent.assert_awaited_once()
