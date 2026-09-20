#!/usr/bin/env python3
"""
執行晨間市場報告生成與推播腳本
可用於：
1. 本地即時測試市場報告產出
2. 透過作業系統排程 (Windows Task Scheduler / Linux crontab) 每天 08:00 自動執行
3. CI/CD 驗證流程

使用方式：
    python scripts/send_market_report.py         # 生成報告並主動推播至 LINE
    python scripts/send_market_report.py --dry-run  # 僅生成並於終端機顯示，不發送 LINE
"""

import sys
import os
import argparse
import asyncio
from pathlib import Path

# 將專案根目錄加入 PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Windows 控制台 UTF-8 編碼支援 (防止 cp950 無法印出 Emoji)
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
from app.services.market_report_service import MarketReportService
from app.services.line_bot_service import LineBotService


async def run(dry_run: bool = False, target_user_id: str = None) -> None:
    logger.info("=== 啟動早晨市場報告生成任務 ===")
    
    # 初始化倉儲與服務
    portfolio_repo = get_portfolio_repository()
    config_repo = get_config_repository()
    session_repo = get_session_repository()
    news_service = NewsService(timeout_seconds=settings.HTTP_TIMEOUT_SECONDS)
    
    market_report_service = MarketReportService(
        portfolio_repo=portfolio_repo,
        news_service=news_service,
    )
    
    line_service = LineBotService(
        config_repo=config_repo,
        portfolio_repo=portfolio_repo,
        session_repo=session_repo,
    )
    
    print("\n⏳ 正在抓取市場行情、焦點新聞與庫存持股，請稍候...")
    try:
        report, indices = await market_report_service.generate_market_report()
    except Exception as e:
        logger.error(f"市場報告生成失敗: {e}")
        print(f"❌ 錯誤: 市場報告生成失敗: {e}")
        sys.exit(1)

    print("\n" + "=" * 50)
    print(f"📑 報告標題: {report.title}")
    print(f"📅 日期時間: {report.date_str}")
    print(f"🌡️ 市場氛圍: {report.market_sentiment}")
    print("=" * 50)
    print("\n【主要市場指數】")
    for idx in indices:
        sign = "+" if idx.change_percent >= 0 else ""
        print(f"  • {idx.name} ({idx.symbol}): {idx.price} ({sign}{idx.change_percent:.2f}%)")
    
    print("\n【大盤總評】")
    print(f"  {report.index_summary}")
    
    print("\n【焦點重點】")
    for headline in report.key_headlines:
        print(f"  • {headline}")
        
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

    print(f"🚀 正在推播市場報告至 LINE 使用者: {recipient} ...")
    await line_service.push_market_report(report, indices, target_user_id=recipient)
    print("✅ LINE 市場報告推播完成！")


def main():
    parser = argparse.ArgumentParser(description="晨間市場報告生成與推播腳本")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="僅在終端機輸出報告內容，不發送 LINE 推播",
    )
    parser.add_argument(
        "--user-id",
        type=str,
        default=None,
        help="指定接收報告的 LINE 使用者 ID (若未指定則使用 .env 中的 LINE_USER_ID)",
    )
    args = parser.parse_args()

    asyncio.run(run(dry_run=args.dry_run, target_user_id=args.user_id))


if __name__ == "__main__":
    main()
