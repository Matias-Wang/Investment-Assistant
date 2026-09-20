import asyncio
from typing import List, Tuple, Optional, Dict
from app.models.domain import BuyCandidate, SystemConfig
from app.models.llm import BuyRecommendation
from app.repositories.base import ConfigRepository, PortfolioRepository
from app.services.market_service import MarketService
from app.services.news_service import NewsService
from app.services.llm_service import LLMService
from app.services.news_filter import NewsRelevanceFilter
from app.services.grounding_verifier import GroundingVerifier
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
        news_filter: Optional[NewsRelevanceFilter] = None,
        grounding_verifier: Optional[GroundingVerifier] = None,
    ):
        self.config_repo = config_repo
        self.portfolio_repo = portfolio_repo
        self.market_service = market_service
        self.news_service = news_service
        self.llm_service = llm_service
        # 預設自建，既有呼叫端無需改動；測試可注入自訂詞表
        self.news_filter = news_filter or NewsRelevanceFilter()
        # 幻覺驗證層：模型為 lazy load，建構此物件不會載入任何權重
        self.grounding_verifier = grounding_verifier or GroundingVerifier()

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

        # 掃描為 I/O bound（yfinance / 新聞 / Gemini），以 semaphore 限流並行，
        # 既縮短牆鐘時間又不至於觸發上游 rate limit。
        semaphore = asyncio.Semaphore(max(1, settings.BUY_SCAN_CONCURRENCY))

        # 步驟 1 & 2：量化技術面過濾 (Close > 20MA) 與 動態預算/零股計算
        quant_results = await asyncio.gather(
            *(self._screen_ticker(t, budget, config, semaphore) for t in all_tickers)
        )
        qualified_candidates: List[BuyCandidate] = [c for c in quant_results if c is not None]

        # 步驟 3 & 4：RAG 文本檢索、關鍵字預篩與 Gemini LLM 推論
        analysis_results = await asyncio.gather(
            *(self._analyze_candidate(c, semaphore) for c in qualified_candidates)
        )
        # gather 保證回傳順序與輸入一致，故建議清單順序仍是決定性的
        buy_recommendations: List[Tuple[BuyRecommendation, BuyCandidate]] = [
            r for r in analysis_results if r is not None
        ]

        logger.info(f"[BuyEngine] Buy scan completed. Generated {len(buy_recommendations)} recommendation(s).")
        return buy_recommendations

    async def _screen_ticker(
        self,
        ticker: str,
        budget: float,
        config: SystemConfig,
        semaphore: asyncio.Semaphore,
    ) -> Optional[BuyCandidate]:
        """量化篩選單一標的；不符條件回傳 None。"""
        async with semaphore:
            asset_type = self._detect_asset_type(ticker)
            name = await self._resolve_name(ticker)

            current_close, sma_20 = await asyncio.to_thread(
                self.market_service.get_price_and_sma20, ticker, asset_type
            )
            if current_close is None or sma_20 is None:
                logger.info(f"[BuyEngine] {ticker} {name}: Missing price or SMA20 data, skipping.")
                return None

            # 量化條件：收盤價站上 20 日均線
            if current_close <= sma_20:
                logger.info(
                    f"[BuyEngine] {ticker} {name}: Close (${current_close:.2f}) <= SMA20 (${sma_20:.2f}), filtered out."
                )
                return None

            # 預算計算：計算零股股數 (包含除零防呆與動態匯率)
            if current_close <= 0:
                logger.warning(f"[BuyEngine] {ticker} {name}: Invalid non-positive close price {current_close}, skipping.")
                return None

            effective_budget = budget
            if asset_type == "us_stock" and config.currency == "TWD":
                rate = getattr(settings, "USD_TWD_RATE", 32.0)
                effective_budget = budget / (rate if rate > 0 else 32.0)

            target_shares = int(effective_budget / current_close)
            if target_shares <= 0:
                logger.info(
                    f"[BuyEngine] {ticker} {name}: Current price ${current_close:.2f} exceeds budget ${effective_budget:.2f}, filtered out."
                )
                return None

            estimated_cost = round(target_shares * current_close, 2)
            logger.info(
                f"[BuyEngine] Qualified quant filter: {ticker} {name} (Close: ${current_close:.2f} > 20MA: ${sma_20:.2f}, Shares: {target_shares})"
            )
            return BuyCandidate(
                ticker=ticker,
                name=name,
                asset_type=asset_type,
                current_price=round(current_close, 2),
                sma_20=round(sma_20, 2),
                target_shares=target_shares,
                estimated_amount=estimated_cost,
                currency="USD" if asset_type == "us_stock" else "TWD",
            )

    async def _analyze_candidate(
        self, candidate: BuyCandidate, semaphore: asyncio.Semaphore
    ) -> Optional[Tuple[BuyRecommendation, BuyCandidate]]:
        """檢索新聞、執行預篩、呼叫 Gemini；非 BUY 或失敗回傳 None。"""
        async with semaphore:
            logger.info(f"[BuyEngine] Fetching recent 3-day news for {candidate.ticker} {candidate.name}...")
            news_text = await self.news_service.get_recent_news_text(
                candidate.ticker, candidate.name, candidate.asset_type
            )

            if not self._passes_prefilter(candidate, news_text):
                return None

            try:
                rec: BuyRecommendation = await asyncio.to_thread(
                    self.llm_service.analyze_buy_candidate, candidate, news_text
                )
                logger.info(
                    f"[BuyEngine] LLM Decision for {candidate.ticker} {candidate.name}: {rec.decision} (Confidence: {rec.confidence_score:.2f})"
                )
                if rec.decision == "BUY":
                    if not await self._passes_grounding(candidate, rec, news_text):
                        return None
                    return (rec, candidate)
            except Exception as e:
                logger.error(f"[BuyEngine] LLM analysis failed for {candidate.ticker}: {e}")
            return None

    async def _passes_grounding(
        self, candidate: BuyCandidate, rec: BuyRecommendation, news_text: str
    ) -> bool:
        """SLM 幻覺驗證：檢查 rec.reasons 是否真被新聞文本支持。

        僅對 `decision == "BUY"` 執行——HOLD/AVOID 不會推播，驗證它們沒有意義且浪費算力。
        回傳 False 僅發生於 `GROUNDING_MODE=enforce` 且接地比例未達門檻。
        任何異常或模型未就緒一律 fail-open。
        """
        mode = (settings.GROUNDING_MODE or "shadow").strip().lower()
        if mode == "off":
            return True

        try:
            report = await asyncio.to_thread(
                self.grounding_verifier.verify, news_text, rec.reasons
            )
        except Exception as e:
            logger.warning(f"[BuyEngine] 幻覺驗證異常，fail-open 放行 {candidate.ticker}：{e}")
            return True

        logger.info(
            f"[BuyEngine] Grounding({mode}) {candidate.ticker} {candidate.name}: {report.summary}"
        )

        if mode == "annotate" and report.verified:
            # 把驗證狀態寫回理由文字，讓推播卡片直接呈現哪幾條有據
            rec.reasons = [f"{c.badge} {c.reason}" for c in report.checks]
            return True

        if mode == "enforce":
            if not report.passes(settings.GROUNDING_MIN_GROUNDED_RATIO):
                logger.warning(
                    f"[BuyEngine] Grounding(enforce) 攔下 {candidate.ticker} {candidate.name}："
                    f"{report.summary}"
                )
                return False

        return True

    def _passes_prefilter(self, candidate: BuyCandidate, news_text: str) -> bool:
        """關鍵字預篩。回傳 False 代表本標的不送 Gemini。

        僅在 `NEWS_FILTER_MODE=enforce` 時才真的淘汰；預設 `shadow` 只記錄判定以供
        量測 recall／precision。任何異常一律 fail-open（放行給 Gemini），
        因為漏掉真利多的代價遠高於多花一次呼叫。
        """
        mode = (settings.NEWS_FILTER_MODE or "shadow").strip().lower()
        if mode == "off":
            return True

        try:
            verdict = self.news_filter.evaluate(news_text)
        except Exception as e:
            logger.warning(f"[BuyEngine] News prefilter errored for {candidate.ticker}, failing open: {e}")
            return True

        if mode == "enforce":
            if not verdict.passed:
                logger.info(
                    f"[BuyEngine] Prefilter(enforce) dropped {candidate.ticker} {candidate.name}: {verdict.summary}"
                )
                return False
            logger.info(
                f"[BuyEngine] Prefilter(enforce) kept {candidate.ticker} {candidate.name}: {verdict.summary}"
            )
            return True

        # shadow：只觀測，不影響結果
        logger.info(
            f"[BuyEngine] Prefilter(shadow) {candidate.ticker} {candidate.name}: {verdict.summary}"
        )
        return True
