import asyncio
from typing import List, Tuple, Optional, Dict
from app.models.domain import BuyCandidate, SystemConfig
from app.models.llm import BuyRecommendation
from app.repositories.base import ConfigRepository, PortfolioRepository
from app.services.market_service import MarketService
from app.services.news_service import NewsService
from app.services.llm_service import LLMService
from app.core.config import settings
from app.core.logging import logger


class BuyStrategyEngine:
    """
    買入策略引擎 (Quantitative + RAG + LLM)
    1. 量化過濾：收盤價站上 20 日均線 (Close > SMA20)
    2. 預算計算：以動態預算除以現價計算可買入零股股數，過濾預算外標的
    3. 文本檢索：獲取近 3 日財經新聞作為 RAG 背景文本
    4. LLM 推論：Gemini 結構化輸出 (JSON schema)，嚴格防幻覺限制
    """

    KNOWN_NAMES: Dict[str, str] = {
        "2330": "台積電",
        "6442": "光聖",
        "2313": "華通",
        "2454": "聯發科",
        "2382": "廣達",
        "SPCX": "Space Exploration ETF",
        "NVDA": "NVIDIA",
        "AAPL": "Apple",
        "TSLA": "Tesla",
    }

    def __init__(
        self,
        config_repo: ConfigRepository,
        portfolio_repo: PortfolioRepository,
        market_service: MarketService,
        news_service: NewsService,
        llm_service: LLMService,
    ):
        self.config_repo = config_repo
        self.portfolio_repo = portfolio_repo
        self.market_service = market_service
        self.news_service = news_service
        self.llm_service = llm_service

    async def _resolve_name(self, ticker: str) -> str:
        """解析股票中文名稱 (優先從持股庫存查找，次之靜態表)"""
        holding = await self.portfolio_repo.get_by_ticker(ticker)
        if holding and holding.name:
            return holding.name
        return self.KNOWN_NAMES.get(ticker.upper(), ticker)

    def _detect_asset_type(self, ticker: str) -> str:
        if ticker.isdigit() or ticker.endswith(".TW") or ticker.endswith(".TWO"):
            return "tw_stock"
        return "us_stock"

    async def scan_candidates(
        self, candidate_tickers: Optional[List[str]] = None
    ) -> List[Tuple[BuyRecommendation, BuyCandidate]]:
        """
        執行完整買入策略掃描流程
        回傳通過量化與 LLM 評估為 'BUY' 的建議清單
        """
        logger.info("[BuyEngine] Starting Buy Strategy Scan...")
        config: SystemConfig = await self.config_repo.get_config()
        budget = float(config.trade_budget_per_order)

        # 彙整候選股票名單
        tickers = candidate_tickers or settings.BUY_CANDIDATES
        # 加上已有庫存的標的
        holdings = await self.portfolio_repo.get_all()
        all_tickers = list(dict.fromkeys(tickers + [h.ticker for h in holdings]))

        qualified_candidates: List[BuyCandidate] = []

        # 步驟 1 & 2：量化技術面過濾 (Close > 20MA) 與 動態預算/零股計算
        for ticker in all_tickers:
            asset_type = self._detect_asset_type(ticker)
            name = await self._resolve_name(ticker)

            current_close, sma_20 = await asyncio.to_thread(
                self.market_service.get_price_and_sma20, ticker, asset_type
            )
            if current_close is None or sma_20 is None:
                logger.info(f"[BuyEngine] {ticker} {name}: Missing price or SMA20 data, skipping.")
                continue

            # 量化條件：收盤價站上 20 日均線
            if current_close <= sma_20:
                logger.info(
                    f"[BuyEngine] {ticker} {name}: Close (${current_close:.2f}) <= SMA20 (${sma_20:.2f}), filtered out."
                )
                continue

            # 預算計算：計算零股股數 (包含除零防呆與動態匯率)
            if current_close <= 0:
                logger.warning(f"[BuyEngine] {ticker} {name}: Invalid non-positive close price {current_close}, skipping.")
                continue

            effective_budget = budget
            if asset_type == "us_stock" and config.currency == "TWD":
                rate = getattr(settings, "USD_TWD_RATE", 32.0)
                effective_budget = budget / (rate if rate > 0 else 32.0)

            target_shares = int(effective_budget / current_close)
            if target_shares <= 0:
                logger.info(
                    f"[BuyEngine] {ticker} {name}: Current price ${current_close:.2f} exceeds budget ${effective_budget:.2f}, filtered out."
                )
                continue

            estimated_cost = round(target_shares * current_close, 2)
            candidate = BuyCandidate(
                ticker=ticker,
                name=name,
                asset_type=asset_type,
                current_price=round(current_close, 2),
                sma_20=round(sma_20, 2),
                target_shares=target_shares,
                estimated_amount=estimated_cost,
                currency="USD" if asset_type == "us_stock" else "TWD",
            )
            qualified_candidates.append(candidate)
            logger.info(
                f"[BuyEngine] Qualified quant filter: {ticker} {name} (Close: ${current_close:.2f} > 20MA: ${sma_20:.2f}, Shares: {target_shares})"
            )

        # 步驟 3 & 4：RAG 文本檢索與 Gemini LLM 推論
        buy_recommendations: List[Tuple[BuyRecommendation, BuyCandidate]] = []

        for candidate in qualified_candidates:
            logger.info(f"[BuyEngine] Fetching recent 3-day news for {candidate.ticker} {candidate.name}...")
            news_text = await self.news_service.get_recent_news_text(
                candidate.ticker, candidate.name, candidate.asset_type
            )

            try:
                rec: BuyRecommendation = await asyncio.to_thread(
                    self.llm_service.analyze_buy_candidate, candidate, news_text
                )
                logger.info(
                    f"[BuyEngine] LLM Decision for {candidate.ticker} {candidate.name}: {rec.decision} (Confidence: {rec.confidence_score:.2f})"
                )

                if rec.decision == "BUY":
                    buy_recommendations.append((rec, candidate))
            except Exception as e:
                logger.error(f"[BuyEngine] LLM analysis failed for {candidate.ticker}: {e}")

        logger.info(f"[BuyEngine] Buy scan completed. Generated {len(buy_recommendations)} recommendation(s).")
        return buy_recommendations
