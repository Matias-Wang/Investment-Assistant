from datetime import date, datetime, time
from functools import lru_cache
from typing import List, Literal, Tuple, Optional
from zoneinfo import ZoneInfo
import yfinance as yf
from google import genai
from google.genai import types

from app.models.domain import MarketIndexItem, PortfolioItem
from app.models.llm import MarketReport, MarketReportDraft
from app.repositories.base import PortfolioRepository
from app.services.news_service import NewsService
from app.core.config import settings
from app.core.gemini_options import thinking_kwargs
from app.core.logging import logger
from app.core.retry import with_retry


Edition = Literal["pre", "post"]
EDITION_LABEL = {"pre": "盤前快報", "post": "盤後快報"}

TAIPEI_TZ = ZoneInfo("Asia/Taipei")
# 各市場所在時區與收盤時間：用來判斷某根日 K 是「已收盤」還是「盤中」
_TW_SESSION = (TAIPEI_TZ, time(13, 30))
_US_SESSION = (ZoneInfo("America/New_York"), time(16, 0))


class MarketNotTradedToday(Exception):
    """要求盤後快報，但台股今日休市或尚未收盤 (正常情況，排程應略過)。"""


class MarketDataUnavailable(Exception):
    """要求盤後快報，但交易日卻抓不到當天台股行情 (異常，需告警並由排程重試)。"""


@lru_cache(maxsize=1)
def _tw_calendar():
    import exchange_calendars  # 載入約需 1 秒，僅在盤後快報需要時才匯入

    return exchange_calendars.get_calendar("XTAI")


def is_tw_trading_day(day: date) -> Optional[bool]:
    """依台灣證交所交易日曆 (XTAI) 判斷是否開市；日曆不可用或超出範圍時回傳 None。"""
    try:
        return bool(_tw_calendar().is_session(day.isoformat()))
    except Exception as e:
        logger.warning(f"[MarketReportService] 交易日曆查詢失敗，改視為休市: {e}")
        return None


def _taipei_now() -> datetime:
    return datetime.now(TAIPEI_TZ)


def _session_of(symbol: str) -> Tuple[ZoneInfo, time]:
    is_tw = symbol.endswith((".TW", ".TWO")) or symbol == "^TWII"
    return _TW_SESSION if is_tw else _US_SESSION


def is_tw_symbol(symbol: str) -> bool:
    return _session_of(symbol) is _TW_SESSION


def describe_close(symbol: str, data_date: Optional[str], now: datetime) -> str:
    """把資料日期轉成給人看的時點，例如「11/16 收盤」或「11/17 盤中」。"""
    if not data_date:
        return "時點不明"
    tz, close_at = _session_of(symbol)
    local_now = now.astimezone(tz)
    mmdd = data_date[5:].replace("-", "/")
    if data_date == local_now.strftime("%Y-%m-%d") and local_now.time() < close_at:
        return f"{mmdd} 盤中"
    return f"{mmdd} 收盤"


def describe_data_time(item: MarketIndexItem, now: datetime) -> str:
    return describe_close(item.symbol, item.data_date, now)


def build_data_basis(indices: List[MarketIndexItem], now: datetime) -> str:
    """資料基準字串由代碼產生，確保使用者看得到行情是哪一天的。"""
    parts = []
    for symbol, label in (("^TWII", "台股"), ("^GSPC", "美股")):
        item = next((i for i in indices if i.symbol == symbol), None)
        if item:
            parts.append(f"{label} {describe_data_time(item, now)}")
    return "・".join(parts) or "行情資料取得失敗"


def resolve_edition(requested: Optional[Edition], indices: List[MarketIndexItem], now: datetime) -> Edition:
    """決定報告版別。

    - 指定 post：台股今日須已收盤。休市日或未收盤擲出 MarketNotTradedToday；
      交易日卻抓不到當天行情 (yfinance 延遲或故障) 擲出 MarketDataUnavailable，
      讓排程告警並重試，而不是誤判為休市默默略過。
    - 未指定 (LINE 手動查詢)：台股今日已收盤才給盤後版，其餘一律盤前版。
    """
    tw = next((i for i in indices if i.symbol == "^TWII"), None)
    today = now.astimezone(TAIPEI_TZ)
    closed_today = bool(
        tw
        and tw.data_date == today.strftime("%Y-%m-%d")
        and today.time() >= _TW_SESSION[1]
    )
    if requested == "post":
        if not tw or not tw.data_date:
            raise MarketDataUnavailable("無法取得台股加權指數行情，盤後快報未產出。")
        if closed_today:
            return "post"
        if today.time() < _TW_SESSION[1]:
            raise MarketNotTradedToday("台股今日尚未收盤，略過盤後快報。")
        if is_tw_trading_day(today.date()):
            raise MarketDataUnavailable(
                f"今日為台股交易日，但行情仍停在 {tw.data_date}，可能是資料延遲，稍後重試。"
            )
        raise MarketNotTradedToday("台股今日休市，略過盤後快報。")
    if requested == "pre":
        return "pre"
    return "post" if closed_today else "pre"


# 版別只描述報告角度；各指數「是哪一天、收盤或盤中」一律以每行標示為準，
# 不在此寫死 (週末、盤中、美股盤中查詢時寫死的說法會與實際資料矛盾)。
_EDITION_GUIDE = {
    "pre": "這是「盤前快報」：以回顧最近一個交易日的行情為主，並提示下一個交易時段的觀察重點。",
    "post": "這是「盤後快報」：台股今日已收盤，請以今日台股盤勢為分析主軸，其他市場僅作背景參考。",
}

_TIME_RULE = (
    "每個指數後方標有「資料時點」(MM/DD 收盤 或 MM/DD 盤中)。描述任何行情時，用字必須與該行標示一致："
    "標「收盤」者須點明是哪一天的收盤，日期不是今天 ({today}) 就不得說成今日走勢；"
    "標「盤中」者須說明為盤中暫定數據，不得當成收盤結果。"
)

_TAKEAWAY_GUIDE = {
    "pre": "給用戶一句針對下一個交易時段的明確操作心理建設或策略叮嚀",
    "post": "給用戶一句針對下一個交易日的明確操作心理建設或策略叮嚀",
}


class MarketReportService:
    """
    智能市場報告服務
    彙整大盤指數 (台股加權、S&P 500、費城半導體、台積電)、總經與產業新聞、現有庫存部位，
    交由 Gemini 3.5 生成結構化精煉市場快報
    """

    INDEX_TARGETS = [
        ("^TWII", "台股加權"),
        ("^GSPC", "標普 500"),
        ("^SOX", "費城半導體"),
        ("2330.TW", "台積電"),
    ]

    def __init__(self, portfolio_repo: PortfolioRepository, news_service: NewsService):
        self.portfolio_repo = portfolio_repo
        self.news_service = news_service
        self.api_key = settings.GEMINI_API_KEY
        self.model_name = settings.GEMINI_MODEL
        self._client: Optional[genai.Client] = None

    def _get_client(self) -> genai.Client:
        if not self._client:
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY is not configured.")
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    @with_retry(max_attempts=3, initial_delay=1.0)
    def fetch_market_indices(self) -> List[MarketIndexItem]:
        """抓取大盤與指標權值股之最新即時走勢"""
        items: List[MarketIndexItem] = []
        for symbol, name in self.INDEX_TARGETS:
            try:
                t = yf.Ticker(symbol)
                df = t.history(period="5d", timeout=10)
                if not df.empty and len(df["Close"]) >= 1:
                    closes = df["Close"].dropna()
                    latest_price = float(closes.iloc[-1])
                    data_date = closes.index[-1].strftime("%Y-%m-%d")
                    if len(closes) >= 2:
                        prev_price = float(closes.iloc[-2])
                    else:
                        prev_price = float(df["Open"].dropna().iloc[-1])

                    change = latest_price - prev_price
                    change_pct = (change / prev_price) * 100.0 if prev_price > 0 else 0.0

                    items.append(MarketIndexItem(
                        symbol=symbol,
                        name=name,
                        price=round(latest_price, 2),
                        change=round(change, 2),
                        change_percent=round(change_pct, 2),
                        data_date=data_date,
                    ))
            except Exception as e:
                logger.warning(f"[MarketReportService] Failed to fetch index {symbol}: {e}")

        return items

    @with_retry(max_attempts=3, initial_delay=1.0)
    def _call_gemini_report(self, client: genai.Client, prompt: str) -> MarketReportDraft:
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=MarketReportDraft,
            temperature=0.2,
            max_output_tokens=settings.MARKET_REPORT_MAX_OUTPUT_TOKENS,
            **thinking_kwargs(),
        )
        try:
            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )
        except Exception as e:
            if "404" in str(e) and self.model_name != "gemini-3.5-flash-lite":
                logger.warning(f"[MarketReportService] Model {self.model_name} returned 404, falling back to gemini-3.5-flash-lite: {e}")
                response = client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=prompt,
                    config=config,
                )
            else:
                raise

        if hasattr(response, "parsed") and response.parsed:
            if isinstance(response.parsed, MarketReportDraft):
                return response.parsed
            elif isinstance(response.parsed, dict):
                return MarketReportDraft(**response.parsed)

        raw_text = response.text
        if not raw_text:
            raise ValueError("Empty response received from Gemini API.")
        return MarketReportDraft.model_validate_json(raw_text)

    async def generate_market_report(
        self, edition: Optional[Edition] = None
    ) -> Tuple[MarketReport, List[MarketIndexItem]]:
        """
        生成市場分析報告。

        Parameters
        ----------
        edition : {"pre", "post"} or None
            pre 為盤前快報、post 為盤後快報；None 依目前時間與台股是否已收盤自動判斷。

        Raises
        ------
        MarketNotTradedToday
            指定 post 但台股今日無交易資料。
        """
        import asyncio
        logger.info(f"[MarketReportService] Generating AI Market Report (edition={edition})...")

        # 1. 抓取大盤指數行情，並依資料日期決定版別
        indices = await asyncio.to_thread(self.fetch_market_indices)
        now = _taipei_now()
        edition = resolve_edition(edition, indices, now)
        label = EDITION_LABEL[edition]

        # 2. 取得用戶當前庫存清單
        holdings: List[PortfolioItem] = await self.portfolio_repo.get_all(status="holding")
        holdings_summary = ", ".join([f"{h.ticker} {h.name}" for h in holdings]) if holdings else "無持股"

        # 3. 檢索近 3 日總經與科技市場焦點新聞
        news_items = await self.news_service.fetch_google_news_rss("台股 大盤 美股 半導體", "")
        news_texts = []
        for n in news_items[:6]:
            news_texts.append(f"- 【{n.title}】({n.publisher}): {n.snippet}")
        news_context = "\n".join(news_texts) if news_texts else "近期無重大突發新聞，市場處於平穩常態。"

        # 格式化指數資訊
        index_lines = []
        for idx in indices:
            sign = "+" if idx.change_percent >= 0 else ""
            index_lines.append(
                f"{idx.name} ({idx.symbol}): {idx.price} ({sign}{idx.change_percent:.2f}%)"
                f"｜資料時點：{describe_data_time(idx, now)}"
            )
        index_context = "\n".join(index_lines)

        # 4. 呼叫 Gemini 3.5 進行綜合結構化推論
        client = self._get_client()

        prompt = f"""
你是一位頂級投資顧問與市場首席宏觀分析師。
請根據以下【市場指數行情】、【近3日市場焦點新聞】與【用戶目前持股組合】，為用戶產出一份專業、精準、兼具客觀大局觀與個人化持股洞察的「{label}」。

【報告性質】{_EDITION_GUIDE[edition]}
【資料時點規則】{_TIME_RULE.format(today=now.strftime("%m/%d"))}

【市場指數行情 (含資料時點)】
{index_context}

【用戶現有持股清單】
{holdings_summary}

【近 3 日市場新聞脈動】
{news_context}

【分析任務指引】
1. title: 請撰寫一個具有專業財經視野的精簡快訊標題 (例如：「美科技股反彈引領台股高檔盤整，AI供應鏈動能持續」)，用字須符合上述資料時點。
2. market_sentiment: 請依據走勢與新聞綜合評估為："樂觀偏多"、"中立震盪"、"保守觀望" 或 "空方承壓"。
3. index_summary: 簡明扼要點評指數動態與量能結構 (1-2 句話)，須點明資料時點。
4. key_headlines: 提煉出 3 則最核心的市場大事件或產業關鍵變化 (每條 20 字以內，明確切中要點)。
5. portfolio_insight: 針對用戶現有持股 ({holdings_summary})，分析上述行情與消息面對其有利或不利、受惠/受害程度為何。
6. actionable_takeaway: {_TAKEAWAY_GUIDE[edition]} (例如：續抱強勢標的，或切忌追高留意均線支撐)。
(報告版別、資料基準與產出時間由系統填入，無需輸出。)

請務必嚴格輸出為符合 JSON Schema 之結構。
""".strip()

        try:
            draft = await asyncio.to_thread(self._call_gemini_report, client, prompt)
            report = MarketReport(
                **draft.model_dump(include=set(MarketReportDraft.model_fields)),
                date_str=now.strftime("%Y-%m-%d %H:%M"),
                edition=label,
                data_basis=build_data_basis(indices, now),
            )
            return report, indices
        except Exception as e:
            logger.error(f"[MarketReportService] Report generation failed: {e}")
            raise
