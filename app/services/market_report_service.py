from datetime import datetime, timezone, timedelta
from typing import List, Tuple, Optional
import yfinance as yf
from google import genai
from google.genai import types

from app.models.domain import MarketIndexItem, PortfolioItem
from app.models.llm import MarketReport
from app.repositories.base import PortfolioRepository
from app.services.news_service import NewsService
from app.core.config import settings
from app.core.gemini_options import thinking_kwargs
from app.core.logging import logger
from app.core.retry import with_retry


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
                        change_percent=round(change_pct, 2)
                    ))
            except Exception as e:
                logger.warning(f"[MarketReportService] Failed to fetch index {symbol}: {e}")

        return items

    @with_retry(max_attempts=3, initial_delay=1.0)
    def _call_gemini_report(self, client: genai.Client, prompt: str) -> MarketReport:
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=MarketReport,
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
            if isinstance(response.parsed, MarketReport):
                return response.parsed
            elif isinstance(response.parsed, dict):
                return MarketReport(**response.parsed)

        raw_text = response.text
        if not raw_text:
            raise ValueError("Empty response received from Gemini API.")
        return MarketReport.model_validate_json(raw_text)

    async def generate_market_report(self) -> Tuple[MarketReport, List[MarketIndexItem]]:
        """
        全自動生成即時市場分析報告
        """
        import asyncio
        logger.info("[MarketReportService] Generating AI Market Report...")

        # 1. 抓取大盤指數行情
        indices = await asyncio.to_thread(self.fetch_market_indices)

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
            index_lines.append(f"{idx.name} ({idx.symbol}): {idx.price} ({sign}{idx.change_percent:.2f}%)")
        index_context = "\n".join(index_lines)

        # 4. 呼叫 Gemini 3.5 進行綜合結構化推論
        client = self._get_client()
        today_str = (datetime.now(timezone.utc) + timedelta(hours=8)).strftime("%Y-%m-%d %H:%M")

        prompt = f"""
你是一位頂級投資顧問與市場首席宏觀分析師。
請根據以下最新【市場指數行情】、【近3日市場焦點新聞】與【用戶目前持股組合】，為用戶產出一份專業、精準、兼具客觀大局觀與個人化持股洞察的「市場每日快報」。

【當前市場指數走勢】
{index_context}

【用戶現有持股清單】
{holdings_summary}

【近 3 日市場新聞脈動】
{news_context}

【分析任務指引】
1. title: 請撰寫一個具有專業財經視野的精簡快訊標題 (例如：「美科技股反彈引領台股高檔盤整，AI供應鏈動能持續」)。
2. date_str: 請填入 "{today_str}"。
3. market_sentiment: 請依據走勢與新聞綜合評估為："樂觀偏多"、"中立震盪"、"保守觀望" 或 "空方承壓"。
4. index_summary: 簡明扼要點評指數動態與量能結構 (1-2 句話)。
5. key_headlines: 提煉出 3 則最核心的市場大事件或產業關鍵變化 (每條 20 字以內，明確切中要點)。
6. portfolio_insight: 針對用戶現有持股 ({holdings_summary})，分析今日市場氛圍與消息面到底對其有利或不利、受惠/受害程度為何。
7. actionable_takeaway: 給用戶一句明確的短線操作心理建設或策略叮嚀 (例如：續抱強勢標的，或切忌追高留意均線支撐)。

請務必嚴格輸出為符合 JSON Schema 之結構。
""".strip()

        try:
            report = await asyncio.to_thread(self._call_gemini_report, client, prompt)
            return report, indices
        except Exception as e:
            logger.error(f"[MarketReportService] Report generation failed: {e}")
            raise
