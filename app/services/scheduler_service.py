from typing import Optional
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
from app.core.config import settings
from app.core.logging import logger
from app.services.sell_engine import SellMonitoringEngine
from app.services.buy_engine import BuyStrategyEngine
from app.services.line_bot_service import LineBotService
from app.services.market_report_service import MarketReportService
from app.services.industry_report_service import IndustryReportService


class SchedulerService:
    """
    排程服務管理器 (APScheduler)
    - 定期執行賣出監控引擎 (Rule Engine)
    - 定期執行買入策略引擎 (Quantitative + RAG + LLM)
    - 每日定時自動生成並推播早晨市場快報 (Market Report + Gemini LLM)
    """

    def __init__(
        self,
        sell_engine: SellMonitoringEngine,
        buy_engine: BuyStrategyEngine,
        line_service: LineBotService,
        market_report_service: Optional[MarketReportService] = None,
        industry_report_service: Optional[IndustryReportService] = None,
    ):
        self.sell_engine = sell_engine
        self.buy_engine = buy_engine
        self.line_service = line_service
        self.market_report_service = market_report_service
        self.industry_report_service = industry_report_service
        self.scheduler = AsyncIOScheduler()

    async def scheduled_sell_check(self) -> None:
        """排程檢查賣出條件 (停利 [10%, 20%] 或 停損 < -50%)"""
        logger.info("[Scheduler] Triggering scheduled sell check...")
        try:
            alerts = await self.sell_engine.scan_portfolio()
            for alert in alerts:
                await self.line_service.push_sell_alert(alert)
        except Exception as e:
            logger.error(f"[Scheduler] Error in scheduled sell check: {e}")

    async def scheduled_buy_scan(self) -> None:
        """排程掃描買入機會 (20MA + 動態預算 + 新聞 RAG + Gemini LLM)"""
        logger.info("[Scheduler] Triggering scheduled buy strategy scan...")
        try:
            recommendations = await self.buy_engine.scan_candidates()
            for rec, candidate in recommendations:
                await self.line_service.push_buy_recommendation(rec, candidate)
        except Exception as e:
            logger.error(f"[Scheduler] Error in scheduled buy scan: {e}")

    async def scheduled_market_report(self) -> None:
        """排程每日早上自動生成並推播市場報告 (AI 綜合分析)"""
        logger.info("[Scheduler] Triggering scheduled morning market report...")
        if not self.market_report_service:
            logger.warning("[Scheduler] MarketReportService is not configured, skipping morning report.")
            return
        try:
            report, indices = await self.market_report_service.generate_market_report()
            await self.line_service.push_market_report(report, indices)
            logger.info(f"[Scheduler] Successfully pushed morning market report: {report.title}")
        except Exception as e:
            logger.error(f"[Scheduler] Error generating/pushing scheduled market report: {e}")

    async def scheduled_industry_report(self) -> None:
        """排程每日自動生成並推播持股相關產業報告 (AI 綜合分析)"""
        logger.info("[Scheduler] Triggering scheduled industry report...")
        if not self.industry_report_service:
            logger.warning("[Scheduler] IndustryReportService is not configured, skipping industry report.")
            return
        try:
            report, performance = await self.industry_report_service.generate_industry_report()
            await self.line_service.push_industry_report(report, performance)
            logger.info(f"[Scheduler] Successfully pushed industry report: {report.title}")
        except ValueError as e:
            # 無持股屬正常情況，不當成錯誤
            logger.info(f"[Scheduler] Industry report skipped: {e}")
        except Exception as e:
            logger.error(f"[Scheduler] Error generating/pushing scheduled industry report: {e}")

    def start(self) -> None:
        if not settings.ENABLE_SCHEDULER:
            logger.info("[Scheduler] Scheduler is disabled in settings.")
            return

        from zoneinfo import ZoneInfo
        taipei_tz = ZoneInfo("Asia/Taipei")

        # 賣出監控：每 N 分鐘執行一次
        self.scheduler.add_job(
            self.scheduled_sell_check,
            trigger=IntervalTrigger(minutes=settings.SELL_CHECK_INTERVAL_MINUTES, timezone=taipei_tz),
            id="sell_monitoring_job",
            replace_existing=True,
        )
        logger.info(f"[Scheduler] Registered sell check job every {settings.SELL_CHECK_INTERVAL_MINUTES} minutes.")

        # 買入掃描：週一至週五指定時間執行 (例如台北時間 13:00 盤中/收盤前)
        self.scheduler.add_job(
            self.scheduled_buy_scan,
            trigger=CronTrigger(
                day_of_week="mon-fri",
                hour=settings.BUY_CHECK_CRON_HOUR,
                minute=settings.BUY_CHECK_CRON_MINUTE,
                timezone=taipei_tz,
            ),
            id="buy_strategy_job",
            replace_existing=True,
        )
        logger.info(
            f"[Scheduler] Registered buy scan job at {settings.BUY_CHECK_CRON_HOUR:02d}:{settings.BUY_CHECK_CRON_MINUTE:02d} Mon-Fri (Asia/Taipei)."
        )

        # 晨間市場報告：每天早上指定時間執行 (預設台北時間 08:00)
        self.scheduler.add_job(
            self.scheduled_market_report,
            trigger=CronTrigger(
                day_of_week=settings.MARKET_REPORT_CRON_DAYS,
                hour=settings.MARKET_REPORT_CRON_HOUR,
                minute=settings.MARKET_REPORT_CRON_MINUTE,
                timezone=taipei_tz,
            ),
            id="morning_market_report_job",
            replace_existing=True,
        )
        logger.info(
            f"[Scheduler] Registered morning market report job at {settings.MARKET_REPORT_CRON_HOUR:02d}:{settings.MARKET_REPORT_CRON_MINUTE:02d} {settings.MARKET_REPORT_CRON_DAYS} (Asia/Taipei)."
        )

        # 每日產業報告：預設台北時間 08:30 (晚市場快報 30 分鐘，避免兩份 Gemini 呼叫同時發生)
        self.scheduler.add_job(
            self.scheduled_industry_report,
            trigger=CronTrigger(
                day_of_week=settings.INDUSTRY_REPORT_CRON_DAYS,
                hour=settings.INDUSTRY_REPORT_CRON_HOUR,
                minute=settings.INDUSTRY_REPORT_CRON_MINUTE,
                timezone=taipei_tz,
            ),
            id="daily_industry_report_job",
            replace_existing=True,
        )
        logger.info(
            f"[Scheduler] Registered industry report job at {settings.INDUSTRY_REPORT_CRON_HOUR:02d}:{settings.INDUSTRY_REPORT_CRON_MINUTE:02d} {settings.INDUSTRY_REPORT_CRON_DAYS} (Asia/Taipei)."
        )

        self.scheduler.start()
        logger.info("[Scheduler] AsyncIOScheduler started.")

    def shutdown(self) -> None:
        if self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("[Scheduler] AsyncIOScheduler stopped.")
