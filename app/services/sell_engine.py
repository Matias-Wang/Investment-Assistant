import asyncio
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import List, Optional, Tuple
from zoneinfo import ZoneInfo
from app.models.domain import (
    PortfolioItem,
    SellAlert,
    PositionValuation,
    PortfolioValuation,
    SystemConfig,
    CooldownPosition,
    SellScanResult,
)
from app.repositories.base import PortfolioRepository, ConfigRepository
from app.services.market_service import MarketService
from app.core.config import settings
from app.core.logging import logger

_ALERT_LABELS = {"TAKE_PROFIT": "停利", "PROFIT_ALERT": "漲幅通知", "STOP_LOSS": "停損"}


def fx_rate_to_base(currency: str, base_currency: str) -> Optional[Decimal]:
    """回傳 1 單位 `currency` 等於多少 `base_currency`；不支援的幣別回傳 None。

    目前僅支援 TWD / USD，匯率取自 `USD_TWD_RATE`（與買入引擎換算預算時相同）。
    改用即時匯率已列為後續優化，見 BACKLOG.md 的 B-001。
    """
    src = (currency or base_currency).upper()
    dst = base_currency.upper()
    if src == dst:
        return Decimal("1")
    rate = Decimal(str(settings.USD_TWD_RATE))
    if rate <= 0:
        return None
    if (src, dst) == ("USD", "TWD"):
        return rate
    if (src, dst) == ("TWD", "USD"):
        return Decimal("1") / rate
    return None


def build_no_alert_notice(result: SellScanResult, config: SystemConfig) -> str:
    """賣出檢查沒有新警示時回報給使用者的訊息。

    有部位在冷卻中時必須明講，不可說成「皆在安全範圍內」：
    該部位其實仍符合停損／停利條件，只是 6 小時內已通知過。
    """
    if not result.cooling_down:
        return (
            "✅ 賣出檢查完成：目前持股均未觸發停利 "
            f"(≥{config.take_profit_min * 100:.0f}%)、漲幅通知 (≥{config.profit_alert_threshold * 100:.0f}%) "
            f"或停損 (≤{config.stop_loss * 100:.0f}%) 條件。"
        )

    tz = ZoneInfo("Asia/Taipei")
    lines = []
    for pos in result.cooling_down:
        try:
            notified = datetime.fromisoformat(pos.last_alert_at).astimezone(tz).strftime("%H:%M")
        except ValueError:
            notified = "稍早"
        label = _ALERT_LABELS.get(pos.alert_type, pos.alert_type)
        lines.append(f"• {pos.ticker} {pos.name} {pos.pnl_rate * 100:+.2f}%（{label}，{notified} 已通知）")
    return (
        "✅ 賣出檢查完成：沒有新的警示。\n\n"
        f"⏳ 以下 {len(result.cooling_down)} 筆部位仍符合警示條件，但 6 小時內已通知過，冷卻期間不重複推播：\n"
        + "\n".join(lines)
    )


class SellMonitoringEngine:
    """
    賣出監控引擎 (Rule Engine)
    - 嚴格不依賴 LLM
    - 讀取資料庫庫存名單，使用 yfinance 獲取最新報價
    - 未實現損益率公式：(現價 - 成本) / 成本
    - 條件觸發：當損益率進入 [take_profit_min, take_profit_max] 區間，或低於 stop_loss 點時，生成賣出警示物件
    """

    def __init__(
        self,
        portfolio_repo: PortfolioRepository,
        config_repo: ConfigRepository,
        market_service: MarketService,
    ):
        self.portfolio_repo = portfolio_repo
        self.config_repo = config_repo
        self.market_service = market_service

    async def scan_portfolio(self) -> List[SellAlert]:
        """
        排程掃描持股庫存，檢查是否有符合停利或停損條件的部位
        """
        return (await self.scan_portfolio_detailed()).alerts

    async def scan_portfolio_detailed(self) -> SellScanResult:
        """同 `scan_portfolio`，另外回報因冷卻而未重複推播的部位。

        需要對使用者回報「沒有警示」時必須用這個版本，否則冷卻中的部位
        會被說成「皆在安全範圍內」。
        """
        logger.info("[SellEngine] Starting scheduled portfolio scan for sell rules...")
        config: SystemConfig = await self.config_repo.get_config()
        holdings: List[PortfolioItem] = await self.portfolio_repo.get_all(status="holding")

        alerts: List[SellAlert] = []
        cooling_down: List[CooldownPosition] = []
        now = datetime.now(timezone.utc)

        def _in_cooldown(item: PortfolioItem, alert_type: str) -> bool:
            if item.last_alert_type != alert_type or not item.last_alert_at:
                return False
            try:
                last_dt = datetime.fromisoformat(item.last_alert_at)
                if now - last_dt < timedelta(hours=6):
                    return True
            except Exception:
                return False
            return False

        def _record_cooldown(item: PortfolioItem, alert_type: str, pnl_rate: float) -> None:
            logger.info(f"[SellEngine] {item.ticker}: {alert_type} alert is within 6-hour cooldown, skipping duplicate push.")
            cooling_down.append(
                CooldownPosition(
                    ticker=item.ticker,
                    name=item.name,
                    alert_type=alert_type,
                    pnl_rate=round(pnl_rate, 4),
                    last_alert_at=item.last_alert_at,
                )
            )

        for item in holdings:
            if item.shares <= 0 or item.average_cost <= 0:
                continue

            current_price_raw = await asyncio.to_thread(
                self.market_service.get_current_price, item.ticker, item.asset_type
            )
            if current_price_raw is None:
                logger.warning(f"[SellEngine] Could not get real-time price for {item.ticker}, skipping.")
                continue

            current_price = Decimal(str(round(current_price_raw, 4)))

            # 計算未實現損益率與損益金額
            pnl_amount = (current_price - item.average_cost) * item.shares
            pnl_rate = float((current_price - item.average_cost) / item.average_cost)

            # 停利條件判斷：高於或等於 take_profit_min (消除超過上限之告警空窗)
            if pnl_rate >= config.take_profit_min:
                if _in_cooldown(item, "TAKE_PROFIT"):
                    _record_cooldown(item, "TAKE_PROFIT", pnl_rate)
                else:
                    if pnl_rate <= config.take_profit_max:
                        trigger_reason = (
                            f"未實現損益率 {pnl_rate * 100:+.2f}%，已進入停利區間 "
                            f"[{config.take_profit_min * 100:.0f}%, {config.take_profit_max * 100:.0f}%]"
                        )
                    else:
                        trigger_reason = (
                            f"未實現損益率 {pnl_rate * 100:+.2f}%，已超過停利上限 "
                            f"({config.take_profit_max * 100:.0f}%)"
                        )
                    alerts.append(
                        SellAlert(
                            ticker=item.ticker,
                            name=item.name,
                            shares=item.shares,
                            current_price=current_price.quantize(Decimal("0.01"), ROUND_HALF_UP),
                            average_cost=item.average_cost.quantize(Decimal("0.01"), ROUND_HALF_UP),
                            pnl_rate=round(pnl_rate, 4),
                            pnl_amount=pnl_amount.quantize(Decimal("0.01"), ROUND_HALF_UP),
                            currency=item.currency,
                            alert_type="TAKE_PROFIT",
                            trigger_reason=trigger_reason,
                        )
                    )
                    logger.info(f"[SellEngine] Triggered TAKE_PROFIT alert for {item.ticker}: {trigger_reason}")
                    item.last_alert_type = "TAKE_PROFIT"
                    item.last_alert_at = now.isoformat()
                    try:
                        await self.portfolio_repo.upsert(item)
                    except Exception as e:
                        logger.warning(f"[SellEngine] Failed to update alert cooldown for {item.ticker}: {e}")

            # 漲幅通知判斷：未實現損益率高於或等於 profit_alert_threshold (預設 5%)
            elif pnl_rate >= config.profit_alert_threshold:
                if _in_cooldown(item, "PROFIT_ALERT"):
                    _record_cooldown(item, "PROFIT_ALERT", pnl_rate)
                else:
                    trigger_reason = (
                        f"未實現損益率 {pnl_rate * 100:+.2f}%，已突破漲幅通知門檻 "
                        f"(+{config.profit_alert_threshold * 100:.1f}%)"
                    )
                    alerts.append(
                        SellAlert(
                            ticker=item.ticker,
                            name=item.name,
                            shares=item.shares,
                            current_price=current_price.quantize(Decimal("0.01"), ROUND_HALF_UP),
                            average_cost=item.average_cost.quantize(Decimal("0.01"), ROUND_HALF_UP),
                            pnl_rate=round(pnl_rate, 4),
                            pnl_amount=pnl_amount.quantize(Decimal("0.01"), ROUND_HALF_UP),
                            currency=item.currency,
                            alert_type="PROFIT_ALERT",
                            trigger_reason=trigger_reason,
                        )
                    )
                    logger.info(f"[SellEngine] Triggered PROFIT_ALERT for {item.ticker}: {trigger_reason}")
                    item.last_alert_type = "PROFIT_ALERT"
                    item.last_alert_at = now.isoformat()
                    try:
                        await self.portfolio_repo.upsert(item)
                    except Exception as e:
                        logger.warning(f"[SellEngine] Failed to update alert cooldown for {item.ticker}: {e}")

            # 停損條件判斷：低於 stop_loss 門檻
            elif pnl_rate <= config.stop_loss:
                if _in_cooldown(item, "STOP_LOSS"):
                    _record_cooldown(item, "STOP_LOSS", pnl_rate)
                else:
                    trigger_reason = (
                        f"未實現損益率 {pnl_rate * 100:+.2f}%，已跌破停損點 "
                        f"({config.stop_loss * 100:.0f}%)"
                    )
                    alerts.append(
                        SellAlert(
                            ticker=item.ticker,
                            name=item.name,
                            shares=item.shares,
                            current_price=current_price.quantize(Decimal("0.01"), ROUND_HALF_UP),
                            average_cost=item.average_cost.quantize(Decimal("0.01"), ROUND_HALF_UP),
                            pnl_rate=round(pnl_rate, 4),
                            pnl_amount=pnl_amount.quantize(Decimal("0.01"), ROUND_HALF_UP),
                            currency=item.currency,
                            alert_type="STOP_LOSS",
                            trigger_reason=trigger_reason,
                        )
                    )
                    logger.info(f"[SellEngine] Triggered STOP_LOSS alert for {item.ticker}: {trigger_reason}")
                    item.last_alert_type = "STOP_LOSS"
                    item.last_alert_at = now.isoformat()
                    try:
                        await self.portfolio_repo.upsert(item)
                    except Exception as e:
                        logger.warning(f"[SellEngine] Failed to update alert cooldown for {item.ticker}: {e}")

        logger.info(
            f"[SellEngine] Scan completed. Generated {len(alerts)} sell alert(s), "
            f"{len(cooling_down)} position(s) in cooldown."
        )
        return SellScanResult(alerts=alerts, cooling_down=cooling_down)

    async def get_portfolio_valuation(self) -> Tuple[PortfolioValuation, SystemConfig]:
        """
        計算現有所有持股的即時評價與總合損益 (供 LINE '查詢庫存' 指令使用)
        """
        config: SystemConfig = await self.config_repo.get_config()
        holdings: List[PortfolioItem] = await self.portfolio_repo.get_all(status="holding")

        valuations: List[PositionValuation] = []
        total_cost = Decimal("0.0")
        total_market_value = Decimal("0.0")

        for item in holdings:
            current_price_raw = await asyncio.to_thread(
                self.market_service.get_current_price, item.ticker, item.asset_type
            )
            if current_price_raw is None:
                current_price = item.average_cost  # 若暫無報價則暫以成本價計算
            else:
                current_price = Decimal(str(round(current_price_raw, 4)))

            cost_basis = item.average_cost * item.shares
            market_value = current_price * item.shares
            pnl_amount = market_value - cost_basis
            pnl_rate = float((current_price - item.average_cost) / item.average_cost) if item.average_cost > 0 else 0.0

            # 單一部位維持原幣別；加總前才換算為基準幣別，否則 USD 會被當成 TWD 直接相加
            fx = fx_rate_to_base(item.currency, config.currency)
            if fx is None:
                logger.warning(
                    f"[SellEngine] {item.ticker}: cannot convert {item.currency} to {config.currency}, "
                    f"excluded from portfolio totals."
                )
            else:
                total_cost += cost_basis * fx
                total_market_value += market_value * fx

            valuations.append(
                PositionValuation(
                    ticker=item.ticker,
                    name=item.name,
                    asset_type=item.asset_type,
                    shares=item.shares,
                    average_cost=item.average_cost.quantize(Decimal("0.01"), ROUND_HALF_UP),
                    current_price=current_price.quantize(Decimal("0.01"), ROUND_HALF_UP),
                    currency=item.currency,
                    cost_basis=cost_basis.quantize(Decimal("0.01"), ROUND_HALF_UP),
                    market_value=market_value.quantize(Decimal("0.01"), ROUND_HALF_UP),
                    pnl_rate=round(pnl_rate, 4),
                    pnl_amount=pnl_amount.quantize(Decimal("0.01"), ROUND_HALF_UP),
                )
            )

        total_pnl = total_market_value - total_cost
        total_pnl_rate = float(total_pnl / total_cost) if total_cost > 0 else 0.0

        result = PortfolioValuation(
            items=valuations,
            total_cost=total_cost.quantize(Decimal("0.01"), ROUND_HALF_UP),
            total_market_value=total_market_value.quantize(Decimal("0.01"), ROUND_HALF_UP),
            total_unrealized_pnl=total_pnl.quantize(Decimal("0.01"), ROUND_HALF_UP),
            total_unrealized_pnl_rate=round(total_pnl_rate, 4),
        )
        return result, config
