"""新聞來源解析的測試。

背景：`fetch_yfinance_news` 曾長期**無聲回傳空清單**——程式讀的是舊版 yfinance 的
扁平欄位（`title`／`providerPublishTime`），而 1.x 起已改為嵌套在 `content` 之下。
本機因 Google News RSS 可用而未被察覺，直到雲端的 RSS 被機房 IP 封鎖（HTTP 503）
才暴露：兩個來源同時失效，新聞 RAG 完全無素材，買入策略永遠判 HOLD。

因此這裡把**兩種格式**都釘住，避免日後升版又靜默失效。
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import httpx
import pytest

from app.services.news_service import (
    NewsService,
    _normalize_yf_news_item,
    _parse_yf_datetime,
)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


NOW = datetime.now(timezone.utc)


# ---------------------------------------------------------------- 時間解析

def test_parse_epoch_seconds():
    dt = _parse_yf_datetime(1_700_000_000)
    assert dt is not None and dt.tzinfo is not None


def test_parse_iso_with_z():
    dt = _parse_yf_datetime("2026-09-21T18:35:00Z")
    assert dt == datetime(2026, 9, 21, 18, 35, tzinfo=timezone.utc)


def test_parse_iso_without_timezone_assumes_utc():
    dt = _parse_yf_datetime("2026-09-21T18:35:00")
    assert dt is not None and dt.tzinfo == timezone.utc


@pytest.mark.parametrize("bad", [None, "", 0, "not-a-date", [], {}])
def test_parse_invalid_returns_none(bad):
    assert _parse_yf_datetime(bad) is None


# ---------------------------------------------------------------- 新格式（yfinance 1.x）

def test_new_nested_schema_parsed():
    item = {
        "id": "abc",
        "content": {
            "title": "TSMC 先進封裝需求強勁",
            "summary": "CoWoS 產能滿載，訂單能見度延續至明年。",
            "description": "備用描述",
            "pubDate": _iso(NOW),
            "provider": {"displayName": "Zacks"},
            "canonicalUrl": {"url": "https://example.com/a"},
        },
    }
    title, publisher, link, snippet, pub_dt = _normalize_yf_news_item(item)
    assert title == "TSMC 先進封裝需求強勁"
    assert publisher == "Zacks"
    assert link == "https://example.com/a"
    assert "CoWoS" in snippet
    assert pub_dt is not None


def test_new_schema_falls_back_to_description():
    item = {"content": {"title": "標題", "description": "只有描述", "pubDate": _iso(NOW)}}
    _, _, _, snippet, _ = _normalize_yf_news_item(item)
    assert snippet == "只有描述"


def test_new_schema_missing_provider_uses_default():
    item = {"content": {"title": "標題", "pubDate": _iso(NOW)}}
    _, publisher, _, _, _ = _normalize_yf_news_item(item)
    assert publisher == "Yahoo Finance"


def test_new_schema_link_fallback_order():
    item = {
        "content": {
            "title": "標題",
            "clickThroughUrl": {"url": "https://example.com/click"},
            "previewUrl": "https://example.com/preview",
        }
    }
    _, _, link, _, _ = _normalize_yf_news_item(item)
    assert link == "https://example.com/click"


def test_new_schema_without_title_is_skipped():
    assert _normalize_yf_news_item({"content": {"summary": "沒有標題"}}) is None


# ---------------------------------------------------------------- 舊格式（向下相容）

def test_old_flat_schema_still_parsed():
    item = {
        "title": "舊格式標題",
        "publisher": "Reuters",
        "link": "https://example.com/old",
        "summary": "舊格式摘要",
        "providerPublishTime": int(NOW.timestamp()),
    }
    title, publisher, link, snippet, pub_dt = _normalize_yf_news_item(item)
    assert (title, publisher, link, snippet) == (
        "舊格式標題", "Reuters", "https://example.com/old", "舊格式摘要",
    )
    assert pub_dt is not None


def test_old_schema_without_title_is_skipped():
    assert _normalize_yf_news_item({"publisher": "X"}) is None


def test_empty_item_is_skipped():
    assert _normalize_yf_news_item({}) is None


# ---------------------------------------------------------------- 取用流程

def _news(content_overrides):
    base = {"title": "標題", "pubDate": _iso(NOW), "summary": "摘要"}
    base.update(content_overrides)
    return {"id": "x", "content": base}


def test_fetch_filters_items_older_than_three_days():
    old = _news({"title": "四天前", "pubDate": _iso(NOW - timedelta(days=4))})
    fresh = _news({"title": "今天"})
    with patch("app.services.news_service.yf.Ticker") as T:
        T.return_value = MagicMock(news=[old, fresh])
        items = NewsService().fetch_yfinance_news("2330", "tw_stock")
    assert [i.title for i in items] == ["今天"]


def test_fetch_keeps_items_without_date():
    """取不到時間者保留——寧可多給 LLM 素材，也不要因缺欄位而丟掉真新聞。"""
    undated = {"id": "x", "content": {"title": "無日期", "summary": "s"}}
    with patch("app.services.news_service.yf.Ticker") as T:
        T.return_value = MagicMock(news=[undated])
        items = NewsService().fetch_yfinance_news("2330", "tw_stock")
    assert [i.title for i in items] == ["無日期"]
    assert items[0].published_at == ""


def test_fetch_appends_tw_suffix():
    with patch("app.services.news_service.yf.Ticker") as T:
        T.return_value = MagicMock(news=[])
        NewsService().fetch_yfinance_news("2330", "tw_stock")
    T.assert_called_once_with("2330.TW")


def test_fetch_us_ticker_unchanged():
    with patch("app.services.news_service.yf.Ticker") as T:
        T.return_value = MagicMock(news=[])
        NewsService().fetch_yfinance_news("NVDA", "us_stock")
    T.assert_called_once_with("NVDA")


def test_fetch_survives_yfinance_exception():
    with patch("app.services.news_service.yf.Ticker", side_effect=RuntimeError("boom")):
        assert NewsService().fetch_yfinance_news("2330", "tw_stock") == []


# ---------------------------------------------------------------- RSS 5xx 需觸發重試

@pytest.mark.asyncio
async def test_rss_5xx_raises_so_retry_applies():
    """原本 5xx 直接 return，使 @with_async_retry 形同虛設。"""
    svc = NewsService()
    resp = MagicMock(status_code=503, request=MagicMock())

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, *a, **k):
            return resp

    with patch("app.services.news_service.httpx.AsyncClient", return_value=FakeClient()):
        with pytest.raises((httpx.HTTPStatusError, RuntimeError, Exception)):
            # 裝飾器耗盡重試後會把最後一次的例外往上拋
            await svc.fetch_google_news_rss.__wrapped__(svc, "2330", "台積電")


@pytest.mark.asyncio
async def test_rss_4xx_returns_empty_without_raising():
    """4xx 屬用戶端錯誤，重試無意義，維持回傳空清單。"""
    svc = NewsService()
    resp = MagicMock(status_code=404, request=MagicMock())

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, *a, **k):
            return resp

    with patch("app.services.news_service.httpx.AsyncClient", return_value=FakeClient()):
        assert await svc.fetch_google_news_rss.__wrapped__(svc, "2330", "台積電") == []


@pytest.mark.asyncio
async def test_rss_failure_degrades_to_yfinance_without_raising():
    """RSS 失敗時必須降級而非往上拋。

    回歸測試：讓 5xx 觸發重試的改動，一度使例外穿過 get_recent_news_text 而
    讓整個買入掃描崩潰（buy_engine 只把 Gemini 呼叫包在 try 內）。
    從 Cloud Run 發出的請求會穩定收到 503，故此降級路徑是常態而非例外。
    """
    svc = NewsService()
    resp = MagicMock(status_code=503, request=MagicMock())

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, *a, **k):
            return resp

    yf_item = {"id": "x", "content": {"title": "yfinance 來源的新聞", "summary": "摘要", "pubDate": _iso(NOW)}}

    with patch("app.services.news_service.httpx.AsyncClient", return_value=FakeClient()):
        with patch("app.services.news_service.yf.Ticker") as T:
            T.return_value = MagicMock(news=[yf_item])
            text = await svc.get_recent_news_text("2330", "台積電")

    assert "查無" not in text
    assert "yfinance 來源的新聞" in text


@pytest.mark.asyncio
async def test_both_sources_failing_returns_no_news_marker():
    """兩個來源都失敗時仍須回傳標記字串，讓防幻覺規則能判 HOLD。"""
    svc = NewsService()
    resp = MagicMock(status_code=503, request=MagicMock())

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, *a, **k):
            return resp

    with patch("app.services.news_service.httpx.AsyncClient", return_value=FakeClient()):
        with patch("app.services.news_service.yf.Ticker", side_effect=RuntimeError("blocked")):
            text = await svc.get_recent_news_text("2330", "台積電")

    assert "查無" in text
