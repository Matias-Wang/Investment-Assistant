from fastapi import APIRouter, Request, Header, HTTPException, status
from app.core.logging import logger
from linebot.v3.exceptions import InvalidSignatureError

router = APIRouter(tags=["LINE Webhook"])


@router.post("/callback")
async def line_webhook(
    request: Request,
    x_line_signature: str = Header(None, alias="X-Line-Signature")
):
    """接收 LINE Messaging API Webhook。

    事件處理為**同步完成**，不使用 BackgroundTasks。原因：Cloud Run 在
    「僅於處理請求時分配 CPU」模式下，回應送出後背景工作會被節流而可能跑不完，
    導致使用者的指令無聲消失。改為同步後，快指令（記帳、查詢、改設定）都在
    數百毫秒內完成，遠低於 LINE 的 5 秒期限；而耗時的買入掃描／賣出檢查／
    市場快報則由 `TaskDispatcher` 交給 Cloud Tasks，webhook 本身只負責入列。
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

    # 2. 同步處理事件。耗時工作已在 line_bot_service 內改為入列，故此處很快返回。
    #    即使偶爾超過 LINE 的 5 秒期限，reply token 仍有效、使用者仍會收到回覆，
    #    且 webhook 重送預設為關閉，不會造成重複執行。
    try:
        await line_service.dispatch_events(events, engines_dict)
    except Exception as e:
        # 事件處理失敗不可回非 2xx，否則 LINE 會累計 webhook 錯誤統計
        logger.exception(f"[LINE Controller] Dispatch error: {e}")
    return {"status": "ok"}
