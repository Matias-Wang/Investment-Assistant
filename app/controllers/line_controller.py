from fastapi import APIRouter, Request, Header, HTTPException, status, BackgroundTasks
from app.core.logging import logger
from linebot.v3.exceptions import InvalidSignatureError

router = APIRouter(tags=["LINE Webhook"])


@router.post("/callback")
async def line_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_line_signature: str = Header(None, alias="X-Line-Signature")
):
    """
    接收 LINE Messaging API Webhook
    驗簽後立即回傳 200，實際事件處理交由 BackgroundTasks，確保 10 秒內回應
    """
    if not x_line_signature:
        logger.warning("[LINE Controller] Missing X-Line-Signature header.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing X-Line-Signature header"
        )

    body_bytes = await request.body()
    body_str = body_bytes.decode("utf-8")

    line_service = request.app.state.line_service
    engines_dict = {
        "sell_engine": request.app.state.sell_engine,
        "buy_engine": request.app.state.buy_engine,
        "market_report_service": getattr(request.app.state, "market_report_service", None),
    }

    if not line_service.parser:
        logger.error("[LINE Controller] Cannot handle webhook: LINE_CHANNEL_SECRET is not configured.")
        return {"status": "error"}

    # 1. 立即驗證簽章
    try:
        events = line_service.parser.parse(body_str, x_line_signature)
    except InvalidSignatureError:
        logger.error("[LINE Controller] Invalid signature.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid signature"
        )
    except Exception as e:
        logger.exception(f"[LINE Controller] Parse error: {e}")
        return {"status": "error"}

    # 2. 驗簽成功後交由 BackgroundTasks 執行，立即回傳 200
    async def _async_dispatch():
        try:
            await line_service.dispatch_events(events, engines_dict)
        except Exception as e:
            logger.exception(f"[LINE Controller] Background dispatch error: {e}")

    background_tasks.add_task(_async_dispatch)
    return {"status": "ok"}
