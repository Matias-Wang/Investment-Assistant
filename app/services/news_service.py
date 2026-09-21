import time
from datetime import datetime, timedelta, timezone
from typing import Any, List, Optional
import httpx
from bs4 import BeautifulSoup
import yfinance as yf
from app.models.domain import NewsItem
from app.core.config import settings
from app.core.logging import logger
from app.core.retry import with_retry, with_async_retry


def _parse_yf_datetime(raw: Any) -> Optional[datetime]:
    """解析 yfinance 的時間欄位，容納 epoch 秒數與 ISO 8601 字串兩種格式。"""
    if not raw:
        return None
    if isinstance(raw, (int, float)):
        try:
            return datetime.fromtimestamp(raw, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(raw, str):
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def _normalize_yf_news_item(item: dict) -> Optional[tuple]:
    """把 yfinance 的新聞項目正規化為 (標題, 來源, 連結, 摘要, 時間)。

    yfinance 1.x 起改為嵌套結構，所有欄位移入 `content`：
        {'id': ..., 'content': {'title', 'description', 'summary', 'pubDate', 'provider', ...}}
    舊版則是扁平的 `title` / `publisher` / `providerPublishTime` / `link` / `summary`。

    本函式同時支援兩種格式。**這曾是一個無聲的失效**：程式只讀舊欄位，
    在 yfinance 1.7 下永遠取不到標題，因此 `fetch_yfinance_news` 一律回傳空清單，
    而本機因有 Google News RSS 可用而未被察覺——直到雲端的 RSS 被封鎖才暴露。
    """
    content = item.get("content")
    if isinstance(content, dict):
        title = (content.get("title") or "").strip()
        if not title:
            return None
        provider = content.get("provider") or {}
        publisher = (provider.get("displayName") if isinstance(provider, dict) else None) or "Yahoo Finance"
        link = ""
        for key in ("canonicalUrl", "clickThroughUrl", "previewUrl"):
            url_obj = content.get(key)
            if isinstance(url_obj, dict) and url_obj.get("url"):
                link = url_obj["url"]
                break
            if isinstance(url_obj, str) and url_obj:
                link = url_obj
                break
        snippet = (content.get("summary") or content.get("description") or title).strip()
        pub_dt = _parse_yf_datetime(content.get("pubDate") or content.get("displayTime"))
        return title, publisher, link, snippet, pub_dt

    # 舊版扁平格式
    title = (item.get("title") or "").strip()
    if not title:
        return None
    return (
        title,
        item.get("publisher") or "Yahoo Finance",
        item.get("link") or "",
        (item.get("summary") or title).strip(),
        _parse_yf_datetime(item.get("providerPublishTime")),
    )


class NewsService:
    """
    財經新聞檢索服務 (提供 RAG 所需之近 3 日新聞文本)
    整合 yfinance 內建新聞與 Google News RSS，具備去重、日期篩選與防呆機制

    兩個來源的可用性差異（2026-09-22 實測）：
    * **Google News RSS**：本機可用，但從 Cloud Run 一律回 HTTP 503——
      機房 IP 遭封鎖。因此雲端不可依賴此來源。
    * **yfinance**：本機與 Cloud Run 皆可用（行情與新聞同一通道）。
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
                fields = _normalize_yf_news_item(item)
                if not fields:
                    continue

                title, publisher, link, snippet, pub_dt = fields
                # 超過 3 日者略過；取不到時間者保留（寧可多給 LLM 一點素材）
                if pub_dt and pub_dt.timestamp() < cutoff_ts:
                    continue

                items.append(NewsItem(
                    title=title,
                    publisher=publisher,
                    published_at=pub_dt.strftime("%Y-%m-%d %H:%M") if pub_dt else "",
                    link=link,
                    snippet=snippet,
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
                if resp.status_code >= 500:
                    # 原本直接 return，使 @with_async_retry 形同虛設。5xx 多為暫時性，
                    # 拋出才會觸發重試。（註：從 Cloud Run 發出的請求會穩定收到 503，
                    # 屬機房 IP 封鎖，重試無用，此時改由 yfinance 來源承擔。）
                    raise httpx.HTTPStatusError(
                        f"Google News RSS 回傳 {resp.status_code}",
                        request=resp.request,
                        response=resp,
                    )
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
        except httpx.HTTPStatusError:
            # 5xx 需往上拋才會觸發 @with_async_retry；若在此吞掉，重試裝飾器形同虛設。
            raise
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
        #    必須攔下例外：5xx 會在重試耗盡後往上拋（見 fetch_google_news_rss），
        #    若不攔，單一來源失敗就會讓整個買入掃描崩潰。從 Cloud Run 發出的請求
        #    會穩定收到 503（機房 IP 封鎖），故此降級路徑是常態而非例外。
        try:
            rss_items = await self.fetch_google_news_rss(ticker, company_name)
            news_items.extend(rss_items)
        except Exception as e:
            logger.warning(
                f"[NewsService] Google News RSS 不可用（{type(e).__name__}），"
                f"改以 yfinance 來源繼續：{ticker}"
            )

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
