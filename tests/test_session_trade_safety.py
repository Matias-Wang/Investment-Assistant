from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.models.line import UserSession
from app.repositories.json_repo import JsonConfigRepository, JsonPortfolioRepository, MemorySessionRepository
from app.services.line_bot_service import LineBotService


@pytest.fixture
def service(tmp_path):
    path = str(tmp_path / "assets.json")
    svc = LineBotService(JsonConfigRepository(path), JsonPortfolioRepository(path), MemorySessionRepository())
    svc.reply_message = MagicMock()
    svc._is_user_authorized = MagicMock(return_value=True)
    return svc


def event(text):
    ev = MagicMock()
    ev.source.user_id = "U_test"
    ev.reply_token = "test-reply"
    ev.message.text = text
    return ev


async def session_for(service, action="CONFIRM_BUY", **overrides):
    values = dict(user_id="U_test", awaiting_action=action, ticker="2330", default_price=1050, default_shares=8)
    values.update(overrides)
    session = UserSession(**values)
    await service.session_repo.save_session(session)
    return session


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["CONFIRM_BUY", "CONFIRM_SELL"])
@pytest.mark.parametrize("text", ["1000001 8", "1050 1000000001", "0 8", "1050 0", "9" * 400 + " 8"])
async def test_extreme_values_do_not_write_or_discard_session(service, action, text):
    await session_for(service, action)
    await service._handle_text_message(event(text), {})
    assert await service.portfolio_repo.get_all(status=None) == []
    assert await service.session_repo.get_session("U_test") is not None
    assert "未記錄交易" in service.reply_message.call_args.args[1][0].text


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["10500 8", "105 8"])
async def test_price_difference_requires_explicit_confirmation(service, text):
    await session_for(service)
    await service._handle_text_message(event(text), {})
    assert await service.portfolio_repo.get_all() == []
    assert "尚未記錄交易" in service.reply_message.call_args.args[1][0].text
    # 重複送補值不能繞過二次確認。
    await service._handle_text_message(event(text), {})
    assert await service.portfolio_repo.get_all() == []
    await service._handle_text_message(event("確認"), {})
    item = await service.portfolio_repo.get_by_ticker("2330")
    assert item.shares == 8
    assert item.average_cost == Decimal(text.split()[0])
    assert await service.session_repo.get_session("U_test") is None


@pytest.mark.asyncio
async def test_anomalous_price_can_be_cancelled(service):
    await session_for(service)
    await service._handle_text_message(event("10500 8"), {})
    await service._handle_text_message(event("取消"), {})
    assert await service.portfolio_repo.get_all() == []
    assert await service.session_repo.get_session("U_test") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["1050 8", "確認"])
async def test_missing_ticker_replies_and_does_not_write(service, text):
    await session_for(service, ticker=None)
    await service._handle_text_message(event(text), {})
    assert "缺少有效股票代號" in service.reply_message.call_args.args[1][0].text
    assert await service.portfolio_repo.get_all(status=None) == []
    assert await service.session_repo.get_session("U_test") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["CONFIRM_BUY", "CONFIRM_SELL"])
@pytest.mark.parametrize("text", ["1050 8", "確認"])
async def test_repository_failure_replies_and_preserves_session(service, action, text):
    await session_for(service, action)
    service.portfolio_repo.record_buy = AsyncMock(side_effect=RuntimeError("private database error"))
    service.portfolio_repo.record_sell = AsyncMock(side_effect=RuntimeError("private database error"))
    await service._handle_text_message(event(text), {})
    message = service.reply_message.call_args.args[1][0].text
    assert "處理失敗" in message and "private database error" not in message
    assert await service.session_repo.get_session("U_test") is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["賣出 2330 1050 20", "賣出20股台積電", "1050 20", "確認"])
async def test_oversell_reports_sold_out_without_inventing_deduction(service, text):
    await service.portfolio_repo.record_buy("2330", 8, 1000)
    if text in ["1050 20", "確認"]:
        await session_for(service, "CONFIRM_SELL", default_shares=20)
    await service._handle_text_message(event(text), {})
    item = await service.portfolio_repo.get_by_ticker("2330")
    assert item.shares == 0 and item.status == "sold"
    message = service.reply_message.call_args.args[1][0].text
    assert "已全數賣出" in message and "超過可用庫存" in message
    assert "扣減 20" not in message and "賣出股數：20" not in message


@pytest.mark.asyncio
async def test_buy_with_one_number_requests_missing_fields(service):
    await service._handle_text_message(event("買入 2330 4"), {})
    assert await service.portfolio_repo.get_all() == []
    assert "請補充價格與股數" in service.reply_message.call_args.args[1][0].text
