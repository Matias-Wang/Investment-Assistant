import re
import urllib.parse
import asyncio
from typing import Optional, Dict, Any, List
from linebot.v3 import WebhookParser
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    ReplyMessageRequest,
    PushMessageRequest,
    TextMessage,
    FlexMessage,
    FlexContainer,
    QuickReply,
    QuickReplyItem,
    MessageAction,
)
from linebot.v3.webhooks import (
    MessageEvent,
    TextMessageContent,
    PostbackEvent,
)
from app.core.config import settings
from app.core.logging import logger
from app.core.retry import with_retry
from app.models.line import UserSession
from app.models.domain import SystemConfig, SellAlert, BuyCandidate, MarketIndexItem
from app.models.llm import BuyRecommendation, MarketReport
from app.models.slm import NLUIntent, NLUIntentResult
from app.services.slm_service import SLMService
from app.repositories.base import ConfigRepository, PortfolioRepository, SessionRepository
from app.utils.flex_templates import (
    build_buy_flex_message,
    build_sell_flex_message,
    build_portfolio_flex_message,
    build_menu_flex_message,
    build_market_report_flex_message,
)


class LineBotService:
    """
    LINE Bot 服務整合：
    - Webhook 事件分派與簽章驗證
    - 動態指令解析 (設定預算/停利/停損、查詢庫存、觸發買賣)
    - 多輪對話流狀態機 (手動買入/賣出確認與庫存更新)
    - Flex Message 推播與回覆
    - 整合 SLM (Small Language Model) 自然語言理解與智慧意圖提取
    """

    def __init__(
        self,
        config_repo: ConfigRepository,
        portfolio_repo: PortfolioRepository,
        session_repo: SessionRepository,
        slm_service: Optional[SLMService] = None,
    ):
        self.config_repo = config_repo
        self.portfolio_repo = portfolio_repo
        self.session_repo = session_repo
        self.slm_service = slm_service

        self.channel_secret = settings.LINE_CHANNEL_SECRET
        self.channel_token = settings.LINE_CHANNEL_ACCESS_TOKEN
        self.parser = WebhookParser(self.channel_secret) if self.channel_secret else None
        self._processed_events: Dict[str, float] = {}  # webhook_event_id -> timestamp (去重)

    def _get_api_client(self) -> ApiClient:
        configuration = Configuration(access_token=self.channel_token)
        return ApiClient(configuration)

    def _get_default_quick_reply(self) -> QuickReply:
        """提供常駐於鍵盤上方的便捷按鈕 (Quick Reply)"""
        return QuickReply(items=[
            QuickReplyItem(action=MessageAction(label="📊 查詢庫存", text="查詢庫存")),
            QuickReplyItem(action=MessageAction(label="📑 市場報告", text="市場報告")),
            QuickReplyItem(action=MessageAction(label="💡 買入建議", text="買入建議")),
            QuickReplyItem(action=MessageAction(label="🚨 賣出檢查", text="賣出檢查")),
            QuickReplyItem(action=MessageAction(label="⚙️ 查詢設定", text="查詢設定")),
            QuickReplyItem(action=MessageAction(label="🎛️ 功能選單", text="選單")),
        ])

    def _attach_quick_reply(self, messages: List[Any]) -> List[Any]:
        """自動為最後一則訊息掛上 Quick Reply 快捷按鈕"""
        if messages:
            last_msg = messages[-1]
            if hasattr(last_msg, "quick_reply") and last_msg.quick_reply is None:
                last_msg.quick_reply = self._get_default_quick_reply()
        return messages

    @with_retry(max_attempts=3, initial_delay=1.0)
    def _send_reply_sync(self, reply_token: str, messages: List[Any]) -> None:
        with self._get_api_client() as api_client:
            api = MessagingApi(api_client)
            req = ReplyMessageRequest(reply_token=reply_token, messages=messages)
            api.reply_message(req)

    def reply_message(self, reply_token: str, messages: List[Any]) -> None:
        """回覆 LINE 訊息 (使用 asyncio.to_thread 非同步執行，避免阻塞事件迴圈)"""
        if not self.channel_token:
            logger.warning("[LineBotService] LINE_CHANNEL_ACCESS_TOKEN not configured. Mock reply.")
            return

        messages = self._attach_quick_reply(messages)
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(asyncio.to_thread(self._send_reply_sync, reply_token, messages))
        except RuntimeError:
            self._send_reply_sync(reply_token, messages)

    @with_retry(max_attempts=3, initial_delay=1.0)
    def _send_push_sync(self, to_user_id: str, messages: List[Any]) -> None:
        with self._get_api_client() as api_client:
            api = MessagingApi(api_client)
            req = PushMessageRequest(to=to_user_id, messages=messages)
            api.push_message(req)

    def push_message(self, to_user_id: str, messages: List[Any]) -> None:
        """主動推播 LINE 訊息 (使用 asyncio.to_thread 非同步執行，避免阻塞事件迴圈)"""
        if not self.channel_token or not to_user_id:
            logger.warning("[LineBotService] Missing token or to_user_id for push message.")
            return

        messages = self._attach_quick_reply(messages)
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(asyncio.to_thread(self._send_push_sync, to_user_id, messages))
        except RuntimeError:
            self._send_push_sync(to_user_id, messages)

    async def push_buy_recommendation(
        self, rec: BuyRecommendation, candidate: BuyCandidate, target_user_id: Optional[str] = None
    ) -> None:
        """推播買入建議 Flex Message"""
        uid = target_user_id or settings.LINE_USER_ID
        if not uid:
            logger.warning("[LineBotService] No LINE user ID configured for buy push.")
            return

        flex_dict = build_buy_flex_message(rec, candidate)
        flex_container = FlexContainer.from_dict(flex_dict)
        flex_msg = FlexMessage(alt_text=f"💡 買入建議: {rec.ticker} {rec.company_name}", contents=flex_container)
        self.push_message(uid, [flex_msg])

    async def push_sell_alert(self, alert: SellAlert, target_user_id: Optional[str] = None) -> None:
        """推播賣出或漲幅警示 Flex Message"""
        uid = target_user_id or settings.LINE_USER_ID
        if not uid:
            logger.warning("[LineBotService] No LINE user ID configured for sell push.")
            return

        flex_dict = build_sell_flex_message(alert)
        flex_container = FlexContainer.from_dict(flex_dict)
        if alert.alert_type == "PROFIT_ALERT":
            alt_text = f"🚀 漲幅通知: {alert.ticker} {alert.name} (+{alert.pnl_rate*100:.2f}%)"
        elif alert.alert_type == "TAKE_PROFIT":
            alt_text = f"💰 停利警示: {alert.ticker} {alert.name} (+{alert.pnl_rate*100:.2f}%)"
        else:
            alt_text = f"🛑 停損警示: {alert.ticker} {alert.name} ({alert.pnl_rate*100:.2f}%)"
        flex_msg = FlexMessage(alt_text=alt_text, contents=flex_container)
        self.push_message(uid, [flex_msg])

    async def push_market_report(
        self,
        report: MarketReport,
        indices: List[MarketIndexItem],
        target_user_id: Optional[str] = None,
    ) -> None:
        """推播市場快報 Flex Message"""
        uid = target_user_id or settings.LINE_USER_ID
        if not uid:
            logger.warning("[LineBotService] No LINE user ID configured for market report push.")
            return

        flex_dict = build_market_report_flex_message(report, indices)
        flex_container = FlexContainer.from_dict(flex_dict)
        flex_msg = FlexMessage(alt_text=f"📑 早安市場快報: {report.title}", contents=flex_container)
        self.push_message(uid, [flex_msg])

    async def dispatch_events(self, events: List[Any], engines_dict: Dict[str, Any]) -> None:
        """分派並處理 LINE 事件清單 (含事件去重與路由)"""
        import time
        now = time.time()
        if len(self._processed_events) > 1000:
            self._processed_events = {k: v for k, v in self._processed_events.items() if now - v < 86400}

        for event in events:
            event_id = getattr(event, "webhook_event_id", None)
            if event_id:
                if event_id in self._processed_events and (now - self._processed_events[event_id] < 86400):
                    logger.info(f"[LineBotService] Duplicate event skipped: {event_id}")
                    continue
                self._processed_events[event_id] = now

            if isinstance(event, MessageEvent) and isinstance(event.message, TextMessageContent):
                await self._handle_text_message(event, engines_dict)
            elif isinstance(event, PostbackEvent):
                await self._handle_postback_event(event, engines_dict)

    async def handle_webhook(self, body_str: str, signature: str, engines_dict: Dict[str, Any]) -> None:
        """
        處理 Webhook 請求本體與事件路由
        """
        if not self.parser:
            logger.error("[LineBotService] Cannot handle webhook: LINE_CHANNEL_SECRET is not configured.")
            return

        try:
            events = self.parser.parse(body_str, signature)
        except InvalidSignatureError:
            logger.error("[LineBotService] Invalid signature received from LINE webhook.")
            raise

        await self.dispatch_events(events, engines_dict)

    def _is_user_authorized(self, user_id: str) -> bool:
        """檢查 LINE 使用者是否在授權白名單內 (Fail-closed)"""
        allowed = set(settings.ALLOWED_LINE_USER_IDS)
        if settings.LINE_USER_ID:
            allowed.add(settings.LINE_USER_ID)
        return bool(allowed and user_id in allowed)

    async def _handle_text_message(self, event: MessageEvent, engines_dict: Dict[str, Any]) -> None:
        user_id = event.source.user_id
        text = event.message.text.strip()
        reply_token = event.reply_token

        logger.info(f"[LineBotService] Received message from {user_id}: '{text}'")

        # 0. 授權白名單防護
        if not self._is_user_authorized(user_id):
            logger.warning(f"[LineBotService] Unauthorized message from {user_id}")
            self.reply_message(reply_token, [TextMessage(text="⛔ 此帳號未獲授權使用本投資助理系統。")])
            return

        # 1. 檢查使用者是否有進行中的多輪對話狀態 (如確認買賣)
        session = await self.session_repo.get_session(user_id)
        if session and session.awaiting_action:
            handled = await self._handle_session_reply(event, session, text)
            if handled:
                return

        # 2. 直接輸入買賣交易指令 (如: `買入 2330 1050 8` 或 `賣出 2330 1080 4`)
        trade_match = re.match(r"^(買入|買進|賣出)\s+([A-Za-z0-9.]+)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)$", text)
        if trade_match:
            action_type, ticker, price_str, shares_str = trade_match.groups()
            try:
                price = float(price_str)
                shares = float(shares_str)
                if price <= 0 or shares <= 0:
                    raise ValueError("價格與股數必須大於 0")
            except ValueError:
                self.reply_message(reply_token, [TextMessage(text="⚠️ 交易價格或股數格式不正確，數值必須大於 0。")])
                return

            ticker = ticker.upper()

            if action_type in ["買入", "買進"]:
                await self.portfolio_repo.record_buy(ticker=ticker, shares=shares, price=price)
                msg = f"✅ 已成功記錄手動買入：\n標的：{ticker}\n成交價：${price:.2f}\n股數：{shares:,.0f} 股"
            else:
                updated = await self.portfolio_repo.record_sell(ticker=ticker, shares=shares)
                if updated:
                    rem_shares = updated.shares if updated.status == "holding" else 0
                    msg = f"✅ 已成功記錄手動賣出：\n標的：{ticker}\n股數：{shares:,.0f} 股\n剩餘持股：{rem_shares:,.0f} 股"
                else:
                    msg = f"⚠️ 找不到庫存標的 {ticker}，無法執行賣出記錄。"

            self.reply_message(reply_token, [TextMessage(text=msg)])
            return

        # 3. 系統設定修改指令解析
        # 指令：設定預算 120000 / 預算 120000
        budget_match = re.match(r"^(?:設定)?預算\s*[:：]?\s*(\d{1,10})$", text)
        if budget_match:
            try:
                new_budget = float(budget_match.group(1))
                if new_budget <= 0:
                    raise ValueError("預算必須大於 0")
                updated_cfg = await self.config_repo.update_field("trade_budget_per_order", new_budget)
                self.reply_message(
                    reply_token,
                    [TextMessage(text=f"✅ 單次買入預算已更新為：NT$ {updated_cfg.trade_budget_per_order:,.0f}")]
                )
            except ValueError:
                self.reply_message(reply_token, [TextMessage(text="⚠️ 預算金額格式不正確，請輸入正整數（如：設定預算 100000）。")])
            return

        # 指令：設定停利 0.1 0.2 或 設定停利 15%
        tp_match = re.match(r"^(?:設定)?停利\s*[:：]?\s*(\d+(?:\.\d+)?%?)(?:\s+(\d+(?:\.\d+)?%?))?$", text)
        if tp_match:
            try:
                val1_str = tp_match.group(1).replace("%", "")
                val1 = float(val1_str)
                val1 = val1 / 100.0 if val1 > 1.0 else val1

                val2_str = tp_match.group(2)
                if val2_str:
                    val2_clean = val2_str.replace("%", "")
                    val2 = float(val2_clean)
                    val2 = val2 / 100.0 if val2 > 1.0 else val2
                    min_tp, max_tp = min(val1, val2), max(val1, val2)
                else:
                    # 單一數字時以 +-5% 為區間
                    min_tp, max_tp = val1 * 0.8, val1

                if min_tp <= 0 or max_tp <= 0:
                    raise ValueError("停利比率必須大於 0")

                cfg = await self.config_repo.get_config()
                cfg.take_profit_min = round(min_tp, 4)
                cfg.take_profit_max = round(max_tp, 4)
                await self.config_repo.update_config(cfg)
                self.reply_message(
                    reply_token,
                    [TextMessage(text=f"✅ 停利區間已更新為：{cfg.take_profit_min*100:.1f}% ~ {cfg.take_profit_max*100:.1f}%")]
                )
            except ValueError:
                self.reply_message(reply_token, [TextMessage(text="⚠️ 停利參數格式不正確，例如：「設定停利 0.1 0.2」或「設定停利 15%」。")])
            return

        # 指令：設定停損 -0.5 或 設定停損 50%
        sl_match = re.match(r"^(?:設定)?停損\s*[:：]?\s*(-?\d+(?:\.\d+)?%?)$", text)
        if sl_match:
            try:
                val_str = sl_match.group(1).replace("%", "")
                val = float(val_str)
                # 轉換為負小數
                if val > 0:
                    val = -val
                val = val / 100.0 if abs(val) > 1.0 else val

                if val >= 0:
                    raise ValueError("停損點必須為負數")

                cfg = await self.config_repo.get_config()
                cfg.stop_loss = round(val, 4)
                await self.config_repo.update_config(cfg)
                self.reply_message(
                    reply_token,
                    [TextMessage(text=f"✅ 停損點已更新為：{cfg.stop_loss*100:.1f}%")]
                )
            except ValueError:
                self.reply_message(reply_token, [TextMessage(text="⚠️ 停損參數格式不正確，例如：「設定停損 -0.5」或「設定停損 -50%」。")])
            return

        # 指令：設定漲幅通知 5% 或 設定漲幅 0.05
        profit_alert_match = re.match(r"^(?:設定)?(?:漲幅通知|漲幅提醒|漲幅)\s*[:：]?\s*(\d+(?:\.\d+)?%?)$", text)
        if profit_alert_match:
            try:
                val_str = profit_alert_match.group(1).replace("%", "")
                val = float(val_str)
                val = val / 100.0 if val > 1.0 else val
                if val <= 0:
                    raise ValueError("漲幅通知門檻必須大於 0")

                cfg = await self.config_repo.get_config()
                cfg.profit_alert_threshold = round(val, 4)
                await self.config_repo.update_config(cfg)
                self.reply_message(
                    reply_token,
                    [TextMessage(text=f"✅ 持股漲幅通知門檻已更新為：+{cfg.profit_alert_threshold*100:.1f}%")]
                )
            except ValueError:
                self.reply_message(reply_token, [TextMessage(text="⚠️ 漲幅通知參數格式不正確，例如：「設定漲幅通知 5%」或「設定漲幅 0.05」。")])
            return

        # 4. 查詢庫存與總損益
        if text in ["查詢庫存", "庫存", "部位", "損益"]:
            sell_engine = engines_dict.get("sell_engine")
            if sell_engine:
                val, cfg = await sell_engine.get_portfolio_valuation()
                flex_dict = build_portfolio_flex_message(val, cfg)
                container = FlexContainer.from_dict(flex_dict)
                self.reply_message(reply_token, [FlexMessage(alt_text="📊 目前持股庫存與損益", contents=container)])
            else:
                self.reply_message(reply_token, [TextMessage(text="⚠️ 系統引擎尚未就緒，請稍後再試。")])
            return

        # 5. 查詢目前設定
        if text in ["查詢設定", "設定", "參數"]:
            cfg = await self.config_repo.get_config()
            msg = (
                f"⚙️ 【目前系統策略參數】\n"
                f"• 單次買入預算：NT$ {cfg.trade_budget_per_order:,.0f}\n"
                f"• 持股漲幅通知門檻：+{cfg.profit_alert_threshold*100:.1f}%\n"
                f"• 停利監控區間：{cfg.take_profit_min*100:.1f}% ~ {cfg.take_profit_max*100:.1f}%\n"
                f"• 停損觸發門檻：{cfg.stop_loss*100:.1f}%\n"
                f"• 基準幣別：{cfg.currency}"
            )
            self.reply_message(reply_token, [TextMessage(text=msg)])
            return

        # 6. 手動觸發買入建議 (非同步背景掃描，避免 reply_token 逾時)
        if text in ["買入建議", "觸發買入", "推薦", "買入掃描"]:
            buy_engine = engines_dict.get("buy_engine")
            if buy_engine:
                self.reply_message(reply_token, [TextMessage(text="🔍 已啟動量化過濾 (20MA) 與新聞 RAG 分析，掃描完成後將主動推播建議卡片，請稍候...")])
                import asyncio
                async def _background_buy_scan():
                    try:
                        recs = await buy_engine.scan_candidates()
                        if not recs:
                            self.push_message(user_id, [TextMessage(text="ℹ️ 今日掃描完成：目前觀察名單中尚無同時符合 20MA 與 AI 買入評級之標的。")])
                        else:
                            for rec, cand in recs:
                                await self.push_buy_recommendation(rec, cand, target_user_id=user_id)
                    except Exception as scan_err:
                        logger.exception(f"[LineBotService] Background buy scan failed: {scan_err}")
                        self.push_message(user_id, [TextMessage(text="⚠️ 買入策略掃描執行過程中發生異常，請稍後再試。")])

                asyncio.create_task(_background_buy_scan())
            return

        # 7. 手動觸發賣出檢查 (非同步背景掃描，避免 reply_token 逾時)
        if text in ["賣出檢查", "觸發賣出", "檢查庫存", "賣出掃描"]:
            sell_engine = engines_dict.get("sell_engine")
            if sell_engine:
                self.reply_message(reply_token, [TextMessage(text="🚨 已開始進行賣出風控檢查，正在獲取庫存即時報價，完成後將主動推播結果...")])
                import asyncio

                async def _background_sell_check():
                    try:
                        alerts = await sell_engine.scan_portfolio()
                        if not alerts:
                            self.push_message(user_id, [TextMessage(text="✅ 目前庫存部位皆在安全持有範圍內，未觸發停利 (>=10%) 或停損 (-50%) 條件。")])
                        else:
                            self.push_message(user_id, [TextMessage(text=f"🚨 賣出檢查完成，發現 {len(alerts)} 筆部位觸發警示條件，即將推播通知卡片...")])
                            for alert in alerts:
                                await self.push_sell_alert(alert, target_user_id=user_id)
                    except Exception as e:
                        logger.exception(f"[LineBotService] Error in background sell check: {e}")

                asyncio.create_task(_background_sell_check())
            return

        # 8. 智能市場快訊報告生成 (非同步背景生成，避免 reply_token 逾時)
        if text in ["市場報告", "市場快報", "今日報告", "盤勢分析", "報告", "盤勢"]:
            market_report_service = engines_dict.get("market_report_service")
            if market_report_service:
                self.reply_message(reply_token, [TextMessage(text="📑 正在為您彙整大盤指數、焦點新聞與持股綜合分析，市場快報生成中，請稍候...")])
                import asyncio

                async def _background_market_report():
                    try:
                        report, indices = await market_report_service.generate_market_report()
                        await self.push_market_report(report, indices, target_user_id=user_id)
                    except Exception as e:
                        logger.exception(f"[LineBotService] Error generating market report: {e}")
                        self.push_message(user_id, [TextMessage(text="⚠️ 市場報告生成失敗，請稍後再試。")])

                asyncio.create_task(_background_market_report())
            else:
                self.reply_message(reply_token, [TextMessage(text="⚠️ 市場報告模組正在初始化，請稍後再試。")])
            return

        # 9. 功能控制面板選單
        if text in ["選單", "menu", "功能", "主選單", "快捷鍵", "控制面板"]:
            menu_dict = build_menu_flex_message()
            container = FlexContainer.from_dict(menu_dict)
            self.reply_message(reply_token, [FlexMessage(alt_text="🎛️ 投資助理快捷控制面板", contents=container)])
            return

        # 10. 幫助與指令說明
        if text.lower() in ["說明", "help", "幫助", "指令", "?"]:
            help_text = (
                "🤖 【LLM 投資策略助理指令說明】\n\n"
                "📊 查詢與分析指令：\n"
                "• 查詢庫存 / 庫存：顯示現有持股評價與損益\n"
                "• 市場報告：AI 彙整大盤、總經新聞與持股受惠分析\n"
                "• 買入建議：立即執行量化+新聞 RAG 策略掃描\n"
                "• 賣出檢查：立即檢查庫存是否觸及停損利條件\n"
                "• 查詢設定 / 設定：查看當前預算與停損利參數\n"
                "• 選單：開啟圖文控制台按鈕卡片\n\n"
                "⚙️ 動態參數設定：\n"
                "• 設定預算 100000\n"
                "• 設定漲幅通知 5% (或 設定漲幅 0.05)\n"
                "• 設定停利 0.1 0.2 (或 設定停利 15%)\n"
                "• 設定停損 -0.5 (或 設定停損 -50%)\n\n"
                "📝 手動更新部位：\n"
                "• 買入 2330 1050 8 (代號 價格 股數)\n"
                "• 賣出 2330 1080 4 (代號 價格 股數)\n"
                "• 或點擊推播卡片之「手動已買入/手動已賣出」按鈕！"
            )
            self.reply_message(reply_token, [TextMessage(text=help_text)])
            return

        # 11. SLM 自然語言意圖理解與槽位提取 (若已配置且 SLM_ENABLED=True)
        if self.slm_service and settings.SLM_ENABLED:
            handled = await self._handle_slm_nlu(event, text, engines_dict)
            if handled:
                return

        # 預設回覆
        self.reply_message(
            reply_token,
            [TextMessage(text=f"收到訊息：「{text}」。輸入「說明」或「查詢庫存」查看更多操作！")]
        )

    async def _handle_slm_nlu(self, event: MessageEvent, text: str, engines_dict: Dict[str, Any]) -> bool:
        """
        委派 SLM (Small Language Model) 進行自然語言意圖辨識與自動操作
        """
        reply_token = event.reply_token
        user_id = event.source.user_id

        try:
            nlu_res = await self.slm_service.parse_intent(text)
        except Exception as e:
            logger.warning(f"[LineBotService] SLM NLU parsing failed: {e}")
            return False

        if nlu_res.intent == NLUIntent.UNKNOWN or nlu_res.confidence < 0.5:
            return False

        logger.info(f"[LineBotService] Handled via SLM NLU: {nlu_res.intent} (conf: {nlu_res.confidence})")

        # 1. 買入記帳
        if nlu_res.intent == NLUIntent.RECORD_BUY:
            if nlu_res.ticker and nlu_res.price and nlu_res.shares:
                if nlu_res.price <= 0 or nlu_res.shares <= 0:
                    self.reply_message(reply_token, [TextMessage(text="⚠️ 價格與股數必須大於 0。")])
                    return True
                await self.portfolio_repo.record_buy(
                    ticker=nlu_res.ticker, shares=nlu_res.shares, price=nlu_res.price
                )
                msg = (
                    f"🤖 【AI 語意記帳成功】\n"
                    f"標的：{nlu_res.ticker}\n"
                    f"成交價：${nlu_res.price:.2f}\n"
                    f"買入股數：{nlu_res.shares:,.0f} 股\n"
                    f"已成功加入庫存狀態表！"
                )
                self.reply_message(reply_token, [TextMessage(text=msg)])
                return True
            elif nlu_res.ticker:
                self.reply_message(
                    reply_token,
                    [TextMessage(text=f"🤖 偵測到您想買入 {nlu_res.ticker}，請補充價格與股數（例如：`買入 {nlu_res.ticker} 1050 8`）。")]
                )
                return True

        # 2. 賣出記帳
        if nlu_res.intent == NLUIntent.RECORD_SELL:
            if nlu_res.ticker and nlu_res.shares:
                if nlu_res.shares <= 0:
                    self.reply_message(reply_token, [TextMessage(text="⚠️ 賣出股數必須大於 0。")])
                    return True
                updated = await self.portfolio_repo.record_sell(
                    ticker=nlu_res.ticker, shares=nlu_res.shares
                )
                if updated:
                    rem_shares = updated.shares if updated.status == "holding" else 0
                    msg = (
                        f"🤖 【AI 語意記帳成功】\n"
                        f"標的：{nlu_res.ticker}\n"
                        f"賣出股數：{nlu_res.shares:,.0f} 股\n"
                        f"剩餘持股：{rem_shares:,.0f} 股"
                    )
                else:
                    msg = f"⚠️ 找不到庫存標的 {nlu_res.ticker}，無法執行賣出記錄。"
                self.reply_message(reply_token, [TextMessage(text=msg)])
                return True
            elif nlu_res.ticker:
                self.reply_message(
                    reply_token,
                    [TextMessage(text=f"🤖 偵測到您想賣出 {nlu_res.ticker}，請補充賣出股數（例如：`賣出 {nlu_res.ticker} 1080 4`）。")]
                )
                return True

        # 3. 查詢庫存
        if nlu_res.intent == NLUIntent.QUERY_PORTFOLIO:
            sell_engine = engines_dict.get("sell_engine")
            if sell_engine:
                val, cfg = await sell_engine.get_portfolio_valuation()
                flex_dict = build_portfolio_flex_message(val, cfg)
                container = FlexContainer.from_dict(flex_dict)
                self.reply_message(reply_token, [FlexMessage(alt_text="📊 目前持股庫存與損益", contents=container)])
            else:
                self.reply_message(reply_token, [TextMessage(text="⚠️ 系統引擎尚未就緒，請稍後再試。")])
            return True

        # 4. 查詢設定
        if nlu_res.intent == NLUIntent.QUERY_CONFIG:
            cfg = await self.config_repo.get_config()
            msg = (
                f"⚙️ 【目前系統策略參數】\n"
                f"• 單次買入預算：NT$ {cfg.trade_budget_per_order:,.0f}\n"
                f"• 持股漲幅通知門檻：+{cfg.profit_alert_threshold*100:.1f}%\n"
                f"• 停利監控區間：{cfg.take_profit_min*100:.1f}% ~ {cfg.take_profit_max*100:.1f}%\n"
                f"• 停損觸發門檻：{cfg.stop_loss*100:.1f}%\n"
                f"• 基準幣別：{cfg.currency}"
            )
            self.reply_message(reply_token, [TextMessage(text=msg)])
            return True

        # 5. 更新設定
        if nlu_res.intent == NLUIntent.UPDATE_CONFIG and nlu_res.config_field and nlu_res.config_value is not None:
            field = nlu_res.config_field.lower()
            val = nlu_res.config_value

            if "budget" in field or "預算" in field:
                budget_val = float(val) if not isinstance(val, list) else float(val[0])
                if budget_val > 0:
                    updated_cfg = await self.config_repo.update_field("trade_budget_per_order", budget_val)
                    self.reply_message(
                        reply_token,
                        [TextMessage(text=f"🤖 [AI 設定更新] 單次買入預算已更新為：NT$ {updated_cfg.trade_budget_per_order:,.0f}")]
                    )
                    return True

            elif "stop_loss" in field or "停損" in field:
                sl_val = float(val) if not isinstance(val, list) else float(val[0])
                sl_val = -abs(sl_val)
                sl_val = sl_val / 100.0 if abs(sl_val) > 1.0 else sl_val
                cfg = await self.config_repo.get_config()
                cfg.stop_loss = round(sl_val, 4)
                await self.config_repo.update_config(cfg)
                self.reply_message(
                    reply_token,
                    [TextMessage(text=f"🤖 [AI 設定更新] 停損點已更新為：{cfg.stop_loss*100:.1f}%")]
                )
                return True

            elif "take_profit" in field or "停利" in field:
                cfg = await self.config_repo.get_config()
                if isinstance(val, list) and len(val) >= 2:
                    v1, v2 = float(val[0]), float(val[1])
                    v1 = v1 / 100.0 if v1 > 1.0 else v1
                    v2 = v2 / 100.0 if v2 > 1.0 else v2
                    cfg.take_profit_min = round(min(v1, v2), 4)
                    cfg.take_profit_max = round(max(v1, v2), 4)
                else:
                    v = float(val if not isinstance(val, list) else val[0])
                    v = v / 100.0 if v > 1.0 else v
                    cfg.take_profit_min = round(v * 0.8, 4)
                    cfg.take_profit_max = round(v, 4)
                await self.config_repo.update_config(cfg)
                self.reply_message(
                    reply_token,
                    [TextMessage(text=f"🤖 [AI 設定更新] 停利區間已更新為：{cfg.take_profit_min*100:.1f}% ~ {cfg.take_profit_max*100:.1f}%")]
                )
                return True

            elif "profit_alert" in field or "漲幅" in field:
                pa_val = float(val) if not isinstance(val, list) else float(val[0])
                pa_val = pa_val / 100.0 if pa_val > 1.0 else pa_val
                cfg = await self.config_repo.get_config()
                cfg.profit_alert_threshold = round(pa_val, 4)
                await self.config_repo.update_config(cfg)
                self.reply_message(
                    reply_token,
                    [TextMessage(text=f"🤖 [AI 設定更新] 漲幅通知門檻已更新為：+{cfg.profit_alert_threshold*100:.1f}%")]
                )
                return True

        # 6. 市場快報
        if nlu_res.intent == NLUIntent.MARKET_REPORT:
            market_report_service = engines_dict.get("market_report_service")
            if market_report_service:
                self.reply_message(reply_token, [TextMessage(text="📑 正在為您彙整大盤指數、焦點新聞與持股綜合分析，市場快報生成中，請稍候...")])
                async def _background_market_report():
                    try:
                        report, indices = await market_report_service.generate_market_report()
                        await self.push_market_report(report, indices, target_user_id=user_id)
                    except Exception as e:
                        logger.exception(f"[LineBotService] Error generating market report: {e}")
                        self.push_message(user_id, [TextMessage(text="⚠️ 市場報告生成失敗，請稍後再試。")])
                asyncio.create_task(_background_market_report())
            return True

        # 7. 買入建議
        if nlu_res.intent == NLUIntent.BUY_SCAN:
            buy_engine = engines_dict.get("buy_engine")
            if buy_engine:
                self.reply_message(reply_token, [TextMessage(text="🔍 已啟動量化過濾與新聞分析，掃描完成後將主動推播建議卡片，請稍候...")])
                async def _background_buy_scan():
                    try:
                        recs = await buy_engine.scan_candidates()
                        if not recs:
                            self.push_message(user_id, [TextMessage(text="ℹ️ 今日掃描完成：目前觀察名單中尚無同時符合 20MA 與 AI 買入評級之標的。")])
                        else:
                            for rec, cand in recs:
                                await self.push_buy_recommendation(rec, cand, target_user_id=user_id)
                    except Exception as scan_err:
                        logger.exception(f"[LineBotService] Background buy scan failed: {scan_err}")
                        self.push_message(user_id, [TextMessage(text="⚠️ 買入策略掃描執行過程中發生異常，請稍後再試。")])
                asyncio.create_task(_background_buy_scan())
            return True

        # 8. 賣出檢查
        if nlu_res.intent == NLUIntent.SELL_CHECK:
            sell_engine = engines_dict.get("sell_engine")
            if sell_engine:
                self.reply_message(reply_token, [TextMessage(text="🚨 已開始進行賣出風控檢查，正在獲取庫存即時報價，完成後將主動推播結果...")])
                async def _background_sell_check():
                    try:
                        alerts = await sell_engine.scan_portfolio()
                        if not alerts:
                            self.push_message(user_id, [TextMessage(text="✅ 目前庫存部位皆在安全持有範圍內，未觸發停利 (>=10%) 或停損 (-50%) 條件。")])
                        else:
                            self.push_message(user_id, [TextMessage(text=f"🚨 賣出檢查完成，發現 {len(alerts)} 筆部位觸發警示條件，即將推播通知卡片...")])
                            for alert in alerts:
                                await self.push_sell_alert(alert, target_user_id=user_id)
                    except Exception as e:
                        logger.exception(f"[LineBotService] Error in background sell check: {e}")
                asyncio.create_task(_background_sell_check())
            return True

        # 9. 選單
        if nlu_res.intent == NLUIntent.MENU:
            menu_dict = build_menu_flex_message()
            container = FlexContainer.from_dict(menu_dict)
            self.reply_message(reply_token, [FlexMessage(alt_text="🎛️ 投資助理快捷控制面板", contents=container)])
            return True

        # 10. 說明
        if nlu_res.intent == NLUIntent.HELP:
            help_text = (
                "🤖 【LLM/SLM 智慧投資助理指令說明】\n\n"
                "📊 支援自然語言與標準指令：\n"
                "• 「查詢庫存」或「看一下賺多少」\n"
                "• 「市場報告」或「今天大盤怎樣」\n"
                "• 「買入建議」或「有沒有推薦買進的」\n"
                "• 「賣出檢查」或「幫我看哪些該賣了」\n"
                "• 「買入 台積電 1050 8」或「剛在1050買入8股台積電」\n"
                "• 「賣出 2330 4」或「賣出4股台積電」\n"
                "• 「設定預算 120000」或「把預算改成12萬」\n"
                "• 「設定停損 -20%」或「停損改 -20%」\n"
                "• 「選單」開啟圖文按鈕卡片"
            )
            self.reply_message(reply_token, [TextMessage(text=help_text)])
            return True

        return False

    async def _handle_postback_event(self, event: PostbackEvent, engines_dict: Dict[str, Any]) -> None:
        """
        處理點擊「手動已買入」或「手動已賣出」按鈕後的 Postback (含授權與格式白名單驗證)
        """
        user_id = event.source.user_id
        data_str = event.postback.data
        reply_token = event.reply_token

        # 0. 授權白名單防護
        if not self._is_user_authorized(user_id):
            logger.warning(f"[LineBotService] Unauthorized postback attempt by {user_id}")
            return

        params = dict(urllib.parse.parse_qsl(data_str))
        action = params.get("action")
        if action not in ["buy_click", "sell_click"]:
            logger.warning(f"[LineBotService] Rejected invalid postback action: {action}")
            return

        ticker = params.get("ticker", "").strip().upper()
        if not re.match(r"^[A-Z0-9.]{1,10}$", ticker):
            logger.warning(f"[LineBotService] Rejected invalid ticker in postback: {ticker}")
            self.reply_message(reply_token, [TextMessage(text="⚠️ 按鈕資料之標的代號格式無效。")])
            return

        try:
            shares = float(params.get("shares", 0))
            price = float(params.get("price", 0))
            if shares <= 0 or price <= 0:
                raise ValueError("Shares and price must be positive")
        except ValueError:
            logger.warning(f"[LineBotService] Invalid shares/price in postback: {data_str}")
            self.reply_message(reply_token, [TextMessage(text="⚠️ 按鈕資料之數值不正確或非正數。")])
            return

        if action == "buy_click":
            # 建立買入確認的對話狀態
            session = UserSession(
                user_id=user_id,
                awaiting_action="CONFIRM_BUY",
                ticker=ticker,
                default_price=price,
                default_shares=shares,
            )
            await self.session_repo.save_session(session)

            prompt_msg = (
                f"📝 【手動買入確認】\n"
                f"標的：{ticker}\n"
                f"建議價格：${price:.2f}\n"
                f"建議股數：{shares:,.0f} 股\n\n"
                f"👉 若完全照此價格成交，請直接回覆「確認」。\n"
                f"👉 若實際成交不同，請回覆：`買入 {ticker} <成交價> <股數>`\n"
                f"👉 若要取消請回覆「取消」。"
            )
            self.reply_message(reply_token, [TextMessage(text=prompt_msg)])

        elif action == "sell_click":
            # 建立賣出確認的對話狀態
            session = UserSession(
                user_id=user_id,
                awaiting_action="CONFIRM_SELL",
                ticker=ticker,
                default_price=price,
                default_shares=shares,
            )
            await self.session_repo.save_session(session)

            prompt_msg = (
                f"📝 【手動賣出確認】\n"
                f"標的：{ticker}\n"
                f"參考現價：${price:.2f}\n"
                f"原持有股數：{shares:,.0f} 股\n\n"
                f"👉 若全數照此賣出，請直接回覆「確認」。\n"
                f"👉 若部分賣出或價格不同，請回覆：`賣出 {ticker} <成交價> <股數>`\n"
                f"👉 若要取消請回覆「取消」。"
            )
            self.reply_message(reply_token, [TextMessage(text=prompt_msg)])

    async def _handle_session_reply(self, event: MessageEvent, session: UserSession, text: str) -> bool:
        """
        處理等待確認中的對話狀態 (確認 / 取消)
        """
        user_id = event.source.user_id
        reply_token = event.reply_token

        if text == "取消":
            await self.session_repo.clear_session(user_id)
            self.reply_message(reply_token, [TextMessage(text="❌ 已取消交易記錄操作。")])
            return True

        if text in ["確認", "OK", "ok", "好"]:
            ticker = session.ticker
            price = session.default_price or 0.0
            shares = session.default_shares or 0.0

            if session.awaiting_action == "CONFIRM_BUY":
                await self.portfolio_repo.record_buy(ticker=ticker, shares=shares, price=price)
                msg = f"🎉 成功記錄買入！\n標的：{ticker}\n價格：${price:.2f}\n股數：{shares:,.0f} 股\n已成功加入庫存狀態表。"
            elif session.awaiting_action == "CONFIRM_SELL":
                updated = await self.portfolio_repo.record_sell(ticker=ticker, shares=shares)
                rem_shares = updated.shares if updated and updated.status == "holding" else 0
                msg = f"🎉 成功記錄賣出！\n標的：{ticker}\n已扣減 {shares:,.0f} 股 (剩餘持股：{rem_shares:,.0f} 股)。"
            else:
                msg = "未知的確認動作。"

            await self.session_repo.clear_session(user_id)
            self.reply_message(reply_token, [TextMessage(text=msg)])
            return True

        # 若使用者直接輸入帶有數值的指令，清除當前 session 交給後續常規解析
        await self.session_repo.clear_session(user_id)
        return False
