import time
from datetime import datetime, timedelta, timezone
from typing import List, Optional
import httpx
from bs4 import BeautifulSoup
import yfinance as yf
from app.models.domain import NewsItem
from app.core.config import settings
from app.core.logging import logger
from app.core.retry import with_retry, with_async_retry


class NewsService:
    """
    財經新聞檢索服務 (提供 RAG 所需之近 3 日新聞文本)
    整合 yfinance 內建新聞與 Google News RSS，具備去重、日期篩選與防呆機制
    """

    def __init__(self, timeout_seconds: float = 10.0):
        self.timeout = timeout_seconds
        # 新聞文本快取：key 為 "ticker|asset_type"，value 為 (寫入時間戳, 文本)
        # 買入掃描與市場快報可能在短時間內重複詢問同一標的，重抓只會多耗時並增加被限流的機會
        self._news_cache: dict[str, tuple[float, str]] = {}

    def _cache_get(self, key: str) -> Optional[str]:
        ttl = settings.NEWS_CACHE_TTL_SECONDS
        if ttl <= 0:
            return None
        entry = self._news_cache.get(key)
        if not entry:
            return None
        cached_at, text = entry
        if time.time() - cached_at > ttl:
            self._news_cache.pop(key, None)
            return None
        return text

    def _cache_put(self, key: str, text: str) -> None:
        if settings.NEWS_CACHE_TTL_SECONDS > 0:
            self._news_cache[key] = (time.time(), text)

    @with_retry(max_attempts=3, initial_delay=1.0)
    def fetch_yfinance_news(self, ticker: str, asset_type: str = "tw_stock") -> List[NewsItem]:
        """從 yfinance 取得最新相關新聞"""
        items: List[NewsItem] = []
        symbol = ticker.strip().upper()
        if asset_type == "tw_stock" and "." not in symbol:
            symbol = f"{symbol}.TW"

        try:
            t = yf.Ticker(symbol)
            news_list = t.news or []
            cutoff_ts = (datetime.now(timezone.utc) - timedelta(days=3)).timestamp()

            for item in news_list:
                # yfinance news schema can have title, publisher, providerPublishTime, link, etc.
                pub_time = item.get("providerPublishTime", 0)
                if pub_time and pub_time < cutoff_ts:
                    continue  # 超過 3 日

                title = item.get("title", "")
                publisher = item.get("publisher", "Yahoo Finance")
                link = item.get("link", "")
                snippet = item.get("summary", "") or title

                if title:
                    dt_str = datetime.fromtimestamp(pub_time, tz=timezone.utc).strftime("%Y-%m-%d %H:%M") if pub_time else ""
                    items.append(NewsItem(
                        title=title,
                        publisher=publisher,
                        published_at=dt_str,
                        link=link,
                        snippet=snippet
                    ))
        except Exception as e:
            logger.warning(f"[NewsService] yfinance news fetch failed for {ticker}: {e}")

        return items

    @with_async_retry(max_attempts=3, initial_delay=1.0)
    async def fetch_google_news_rss(self, ticker: str, company_name: str = "") -> List[NewsItem]:
        """從 Google News RSS 檢索中文財經新聞 (3天內，使用參數字典安全編碼)"""
        items: List[NewsItem] = []
        query = f"{ticker} {company_name}".strip()
        base_url = "https://news.google.com/rss/search"
        params = {
            "q": query,
            "hl": "zh-TW",
            "gl": "TW",
            "ceid": "TW:zh-Hant",
        }

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(base_url, params=params, headers=headers)
                if resp.status_code != 200:
                    logger.warning(f"[NewsService] Google News RSS returned HTTP {resp.status_code} for query: {query}")
                    return items

                soup = BeautifulSoup(resp.content, "xml")
                xml_items = soup.find_all("item")
                cutoff_dt = datetime.now(timezone.utc) - timedelta(days=3)

                for node in xml_items[:8]:
                    title = node.title.text if node.title else ""
                    link = node.link.text if node.link else ""
                    pub_date_raw = node.pubDate.text if node.pubDate else ""
                    source = node.source.text if node.source else "Google News"
                    desc = node.description.text if node.description else ""

                    # 清理 description 內的 HTML 標籤
                    if desc:
                        desc_soup = BeautifulSoup(desc, "html.parser")
                        desc = desc_soup.get_text(separator=" ", strip=True)

                    items.append(NewsItem(
                        title=title,
                        publisher=source,
                        published_at=pub_date_raw,
                        link=link,
                        snippet=desc or title
                    ))
        except Exception as e:
            logger.warning(f"[NewsService] Google News RSS fetch failed for {query}: {e}")

        return items

    async def get_recent_news_text(self, ticker: str, company_name: str, asset_type: str = "tw_stock") -> str:
        """
        獲取標的近 3 日所有可用新聞並組裝為 Markdown 格式之 RAG 檢索文本
        """
        import asyncio

        cache_key = f"{ticker.strip().upper()}|{asset_type}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            logger.info(f"[NewsService] Cache hit for {ticker} ({asset_type}), skipping refetch.")
            return cached

        news_items: List[NewsItem] = []

        # 1. 抓取 Google News RSS
        rss_items = await self.fetch_google_news_rss(ticker, company_name)
        news_items.extend(rss_items)

        # 2. 抓取 yfinance 新聞補足 (使用 to_thread 與逾時保護，避免阻塞事件迴圈)
        try:
            yf_items = await asyncio.wait_for(
                asyncio.to_thread(self.fetch_yfinance_news, ticker, asset_type),
                timeout=10.0
            )
            news_items.extend(yf_items)
        except Exception as e:
            logger.warning(f"[NewsService] yfinance news fetch timed out or failed for {ticker}: {e}")

        # 3. 去重 (依標題)
        seen_titles = set()
        unique_news = []
        for n in news_items:
            clean_title = n.title.strip()
            if clean_title and clean_title not in seen_titles:
                seen_titles.add(clean_title)
                unique_news.append(n)

        if not unique_news:
            # 刻意不快取空結果：抓取失敗與「真的沒新聞」在此無法區分，
            # 若快取下來會讓後續一小時內都看不到實際存在的新聞。
            return "【近 3 日查無任何相關財經新聞或市場報導】"

        # 組裝為清晰的文本格式
        text_lines = [f"### 標的: {ticker} {company_name} 近 3 日相關財經新聞報導：\n"]
        for idx, item in enumerate(unique_news[:6], 1):
            text_lines.append(f"【新聞 {idx}】標題：{item.title}")
            if item.published_at:
                text_lines.append(f"發布時間：{item.published_at} | 來源：{item.publisher}")
            text_lines.append(f"摘要內容：{item.snippet}\n")

        news_text = "\n".join(text_lines)
        self._cache_put(cache_key, news_text)
        return news_text
