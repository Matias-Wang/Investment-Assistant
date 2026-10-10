#!/usr/bin/env python3
"""
執行每日產業報告生成與推播腳本 (依使用者持股所屬產業)

使用方式：
    python scripts/send_industry_report.py            # 生成報告並推播至 LINE
    python scripts/send_industry_report.py --dry-run  # 僅生成並於終端機顯示，不發送 LINE
"""

import sys
import argparse
import asyncio
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app.core.config import settings
from app.core.logging import logger
from app.repositories.factory import (
    get_config_repository,
    get_portfolio_repository,
    get_session_repository,
)
from app.services.news_service import NewsService
from app.services.industry_report_service import IndustryReportService
from app.services.line_bot_service import LineBotService


async def run(dry_run: bool = False, target_user_id: str = None) -> None:
    portfolio_repo = get_portfolio_repository()
    news_service = NewsService(timeout_seconds=settings.HTTP_TIMEOUT_SECONDS)
    service = IndustryReportService(portfolio_repo=portfolio_repo, news_service=news_service)

    print("\n⏳ 正在彙整持股產業、行情與新聞，請稍候...")
    try:
        report, performance = await service.generate_industry_report()
    except ValueError as e:
        print(f"ℹ️ {e}")
        return
    except Exception as e:
        logger.error(f"產業報告生成失敗: {e}")
        print(f"❌ 錯誤: 產業報告生成失敗: {e}")
        sys.exit(1)

    print("\n" + "=" * 50)
    print(f"📑 報告標題: {report.title}")
    print(f"📅 日期時間: {report.date_str}")
    print(f"🌡️ 產業氛圍: {report.market_sentiment}")
    print("=" * 50)
    print("\n【持股產業表現】")
    for p in performance:
        sign = "+" if p.avg_change_percent >= 0 else ""
        print(f"  • {p.industry} ({' '.join(p.tickers)}): {sign}{p.avg_change_percent:.2f}%")
    print("\n【強弱對比】")
    print(f"  {report.index_summary}")
    print("\n【各產業趨勢】")
    for s in report.industries:
        print(f"  • {s.industry} [{s.outlook}] {s.summary}")
    print("\n【焦點重點】")
    for h in report.key_headlines:
        print(f"  • {h}")
    print("\n【持股洞察】")
    print(f"  {report.portfolio_insight}")
    print("\n【操盤叮嚀】")
    print(f"  {report.actionable_takeaway}")
    print("=" * 50 + "\n")

    if dry_run:
        print("🔍 模式為 --dry-run，已跳過 LINE 訊息推播。")
        return

    recipient = target_user_id or settings.LINE_USER_ID
    if not recipient:
        print("⚠️ 警告: 未設定 LINE_USER_ID，無法發送 LINE 推播。")
        return

    line_service = LineBotService(
        config_repo=get_config_repository(),
        portfolio_repo=portfolio_repo,
        session_repo=get_session_repository(),
    )
    print(f"🚀 正在推播產業報告至 LINE 使用者: {recipient} ...")
    await line_service.push_industry_report(report, performance, target_user_id=recipient)
    print("✅ LINE 產業報告推播完成！")


def main():
    parser = argparse.ArgumentParser(description="每日產業報告生成與推播腳本")
    parser.add_argument("--dry-run", action="store_true", help="僅在終端機輸出，不發送 LINE 推播")
    parser.add_argument("--user-id", type=str, default=None, help="指定接收報告的 LINE 使用者 ID")
    args = parser.parse_args()
    asyncio.run(run(dry_run=args.dry_run, target_user_id=args.user_id))


if __name__ == "__main__":
    main()
