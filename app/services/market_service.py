import time
import math
import threading
from typing import Optional, Tuple, Dict, Any
import yfinance as yf
import pandas as pd
from app.core.logging import logger
from app.core.retry import with_retry


class MarketService:
    """
    市場報價與技術指標服務 (yfinance 封裝)
    含超時、重試、重試退避與例外防護機制
    """

    def __init__(self, cache_ttl_seconds: float = 60.0, max_cache_size: int = 256):
        self._price_cache: Dict[str, Tuple[float, float]] = {}  # ticker -> (price, timestamp)
        self._cache_ttl = cache_ttl_seconds
        self._max_cache_size = max_cache_size
        self._cache_lock = threading.Lock()

    def _resolve_ticker(self, ticker: str, asset_type: str = "tw_stock") -> str:
        """
        將台股代號 (如 2330) 轉換為 yfinance 格式 (如 2330.TW 或 2330.TWO)
        若已帶有後綴或為美股則直接回傳
        """
        ticker = ticker.strip().upper()
        if "." in ticker:
            return ticker
        if asset_type == "tw_stock" or ticker.isdigit():
            # 預設先使用集中市場 (.TW)
            return f"{ticker}.TW"
        return ticker

    @with_retry(max_attempts=3, initial_delay=1.0, backoff_factor=2.0)
    def _fetch_history(self, yf_ticker: str, period: str = "3mo") -> pd.DataFrame:
        """從 yfinance 抓取歷史 K 線資料 (帶重試與退避)"""
        t = yf.Ticker(yf_ticker)
        df = t.history(period=period, timeout=10)
        if df.empty and yf_ticker.endswith(".TW"):
            # 嘗試櫃買中心 (.TWO)
            alt_ticker = yf_ticker.replace(".TW", ".TWO")
            logger.info(f"[MarketService] Empty data for {yf_ticker}, trying OTC market: {alt_ticker}")
            t_alt = yf.Ticker(alt_ticker)
            df = t_alt.history(period=period, timeout=10)
        return df

    def get_current_price(self, ticker: str, asset_type: str = "tw_stock") -> Optional[float]:
        """
        取得標的最新即時價格 (有快取機制與數值安全驗證)
        """
        yf_symbol = self._resolve_ticker(ticker, asset_type)
        now = time.time()

        with self._cache_lock:
            if yf_symbol in self._price_cache:
                cached_price, ts = self._price_cache[yf_symbol]
                if now - ts < self._cache_ttl:
                    return cached_price

        try:
            df = self._fetch_history(yf_symbol, period="5d")
            if df.empty or "Close" not in df.columns:
                logger.warning(f"[MarketService] No price data returned for {ticker} ({yf_symbol})")
                return None

            closes = df["Close"].dropna()
            if closes.empty:
                return None

            latest_price = float(closes.iloc[-1])
            if not math.isfinite(latest_price) or latest_price <= 0:
                logger.warning(f"[MarketService] Invalid non-finite or non-positive price: {latest_price} for {ticker}")
                return None

            with self._cache_lock:
                if len(self._price_cache) >= self._max_cache_size:
                    # 淘汰最舊的一半
                    sorted_keys = sorted(self._price_cache.keys(), key=lambda k: self._price_cache[k][1])
                    for k in sorted_keys[: len(sorted_keys) // 2]:
                        del self._price_cache[k]
                self._price_cache[yf_symbol] = (latest_price, now)

            return latest_price
        except Exception as e:
            logger.error(f"[MarketService] Failed to get price for {ticker} ({yf_symbol}): {e}")
            return None

    def get_price_and_sma20(
        self, ticker: str, asset_type: str = "tw_stock"
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        取得最新收盤價與 20 日均線 (SMA 20)
        用於量化買入策略過濾 (含異常值防護)
        """
        yf_symbol = self._resolve_ticker(ticker, asset_type)
        try:
            df = self._fetch_history(yf_symbol, period="3mo")
            if df.empty or "Close" not in df.columns or len(df["Close"].dropna()) < 20:
                logger.warning(
                    f"[MarketService] Insufficient history to calculate 20MA for {ticker} (need >= 20 bars, got {len(df)})"
                )
                if not df.empty and "Close" in df.columns:
                    closes = df["Close"].dropna()
                    if not closes.empty:
                        latest = float(closes.iloc[-1])
                        if math.isfinite(latest) and latest > 0:
                            return latest, None
                return None, None

            close_series = df["Close"].dropna()
            latest_close = float(close_series.iloc[-1])
            sma_20 = float(close_series.rolling(window=20).mean().iloc[-1])

            if not math.isfinite(latest_close) or latest_close <= 0:
                logger.warning(f"[MarketService] Invalid non-positive close: {latest_close} for {ticker}")
                return None, None
            if not math.isfinite(sma_20) or sma_20 <= 0:
                logger.warning(f"[MarketService] Invalid non-positive SMA20: {sma_20} for {ticker}")
                return latest_close, None

            # 同時寫入現價快取
            with self._cache_lock:
                self._price_cache[yf_symbol] = (latest_close, time.time())

            return latest_close, sma_20
        except Exception as e:
            logger.error(f"[MarketService] Failed to calculate SMA20 for {ticker}: {e}")
            return None, None
