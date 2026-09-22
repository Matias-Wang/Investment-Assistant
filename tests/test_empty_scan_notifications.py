"""掃描結果為空時仍須主動回報。

若無結果就完全不推播，使用者只會看到「掃描中／檢查中」後一片安靜，
與 Cloud Tasks 或 SERVICE_BASE_URL 沒生效的故障狀況無法區分。
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from app.controllers.api_controller import trigger_sell_check, trigger_buy_scan
from app.models.domain import SellAlert
from app.services.line_bot_service import LineBotService


def _make_request(sell_alerts=None, buy_results=None):
    request = MagicMock()
    request.app.state.sell_engine.scan_portfolio = AsyncMock(return_value=sell_alerts or [])
    request.app.state.buy_engine.scan_candidates = AsyncMock(return_value=buy_results or [])
    line_service = MagicMock()
    line_service.push_text = AsyncMock()
    line_service.push_sell_alert = AsyncMock()
    line_service.push_buy_recommendation = AsyncMock()
    request.app.state.line_service = line_service
    return request, line_service


@pytest.fixture
def sell_alert():
    return SellAlert(
        ticker="2330",
        name="台積電",
        alert_type="TAKE_PROFIT",
        shares="8",
        average_cost="2375.63",
        current_price="2700.00",
        pnl_rate=0.1365,
        pnl_amount="2595.00",
        currency="TWD",
        trigger_reason="獲利率 13.65% 進入停利區間 [10%, 20%]",
    )


@pytest.mark.asyncio
async def test_sell_check_no_alerts_pushes_notice():
    """賣出檢查 0 筆警示時，仍要推播一則「未觸發」通知"""
    request, line_service = _make_request(sell_alerts=[])

    result = await trigger_sell_check(request, push=True, user_id="U_test")

    assert result == []
    line_service.push_text.assert_awaited_once()
    assert line_service.push_text.await_args.kwargs["target_user_id"] == "U_test"
    line_service.push_sell_alert.assert_not_awaited()


@pytest.mark.asyncio
async def test_sell_check_with_alerts_skips_notice(sell_alert):
    """有警示時只推警示，不要多送一則「未觸發」通知"""
    request, line_service = _make_request(sell_alerts=[sell_alert])

    result = await trigger_sell_check(request, push=True, user_id="U_test")

    assert len(result) == 1
    line_service.push_text.assert_not_awaited()
    line_service.push_sell_alert.assert_awaited_once()


@pytest.mark.asyncio
async def test_sell_check_no_alerts_silent_when_notify_empty_false():
    """排程路徑（notify_empty=false）0 筆時完全不出聲。

    每 30 分鐘一次，若每次都回報「沒事」，一天會收到十幾則雜訊。
    """
    request, line_service = _make_request(sell_alerts=[])

    result = await trigger_sell_check(request, push=True, notify_empty=False)

    assert result == []
    line_service.push_text.assert_not_awaited()
    line_service.push_sell_alert.assert_not_awaited()


@pytest.mark.asyncio
async def test_sell_check_alerts_still_push_when_notify_empty_false(sell_alert):
    """notify_empty=false 只關閉「沒事」通知，真的該賣時仍必須推播"""
    request, line_service = _make_request(sell_alerts=[sell_alert])

    result = await trigger_sell_check(request, push=True, notify_empty=False)

    assert len(result) == 1
    line_service.push_text.assert_not_awaited()
    line_service.push_sell_alert.assert_awaited_once()


@pytest.mark.asyncio
async def test_sell_check_without_push_stays_silent():
    """push=False（驗證用的直接呼叫）不得推播任何訊息"""
    request, line_service = _make_request(sell_alerts=[])

    await trigger_sell_check(request, push=False)

    line_service.push_text.assert_not_awaited()
    line_service.push_sell_alert.assert_not_awaited()


@pytest.mark.asyncio
async def test_buy_scan_no_results_pushes_notice():
    """買入掃描 0 筆建議時，仍要推播一則「無符合標的」通知"""
    request, line_service = _make_request(buy_results=[])

    result = await trigger_buy_scan(request, push=True, user_id="U_test")

    assert result == []
    line_service.push_text.assert_awaited_once()
    assert line_service.push_text.await_args.kwargs["target_user_id"] == "U_test"
    line_service.push_buy_recommendation.assert_not_awaited()


@pytest.mark.asyncio
async def test_push_text_completes_before_returning():
    """push_text 必須等推播真正送出才返回。

    射後不理（loop.create_task）在 Cloud Run 上會被 CPU 節流吃掉——回應送出後
    背景工作就沒有 CPU 了，而「無結果」通知後面沒有任何工作能拖住請求。
    """
    service = LineBotService(MagicMock(), MagicMock(), MagicMock())
    service.channel_token = "test_token"

    with patch.object(service, "_send_push_sync") as mock_send:
        await service.push_text("測試訊息", target_user_id="U_test")

        # await 返回時就必須已經送出，而不是留一個待辦的 task
        mock_send.assert_called_once()
        assert mock_send.call_args.args[0] == "U_test"
        assert mock_send.call_args.args[1][0].text == "測試訊息"


@pytest.mark.asyncio
async def test_push_text_without_token_is_noop():
    """沒有 channel token 時安靜跳過，不得拋出例外中斷掃描流程"""
    service = LineBotService(MagicMock(), MagicMock(), MagicMock())
    service.channel_token = ""

    with patch.object(service, "_send_push_sync") as mock_send:
        await service.push_text("測試訊息", target_user_id="U_test")

    mock_send.assert_not_called()
