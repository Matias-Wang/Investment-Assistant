import asyncio
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import List, Optional, Tuple
from app.models.domain import PortfolioItem, SellAlert, PositionValuation, PortfolioValuation, SystemConfig
from app.repositories.base import PortfolioRepository, ConfigRepository
from app.services.market_service import MarketService
from app.core.logging import logger


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
        logger.info("[SellEngine] Starting scheduled portfolio scan for sell rules...")
        config: SystemConfig = await self.config_repo.get_config()
        holdings: List[PortfolioItem] = await self.portfolio_repo.get_all(status="holding")

        alerts: List[SellAlert] = []
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
                    logger.info(f"[SellEngine] {item.ticker}: TAKE_PROFIT alert is within 6-hour cooldown, skipping duplicate push.")
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
                    logger.info(f"[SellEngine] {item.ticker}: PROFIT_ALERT alert is within 6-hour cooldown, skipping duplicate push.")
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
                    logger.info(f"[SellEngine] {item.ticker}: STOP_LOSS alert is within 6-hour cooldown, skipping duplicate push.")
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

        logger.info(f"[SellEngine] Scan completed. Generated {len(alerts)} sell alert(s).")
        return alerts

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

            total_cost += cost_basis
            total_market_value += market_value

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
