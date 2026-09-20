import json
import re
from typing import Optional, Dict, Any
import httpx

from app.core.config import settings
from app.core.logging import logger
from app.models.slm import NLUIntent, NLUIntentResult


class SLMService:
    """
    SLM (Small Language Model) 推論服務
    採用 OpenAI 相容通訊協議 (OpenAI-compatible /chat/completions)
    完美適配 Colab Pro (vLLM / llama.cpp / Ollama) 或本地端輕量模型
    專責高效能、低延遲之意圖識別 (NLU) 與槽位抽取 (Slot Filling)
    """

    KNOWN_TICKER_MAP: Dict[str, str] = {
        "台積電": "2330",
        "台積": "2330",
        "聯發科": "2454",
        "發哥": "2454",
        "廣達": "2382",
        "光聖": "6442",
        "華通": "2313",
        "鴻海": "2317",
        "大立光": "3008",
        "富邦金": "2881",
        "國泰金": "2882",
        "輝達": "NVDA",
        "特斯拉": "TSLA",
        "蘋果": "AAPL",
        "微軟": "MSFT",
        "谷哥": "GOOGL",
        "谷歌": "GOOGL",
    }

    SYSTEM_PROMPT = """你是一位極度嚴謹且高效的個人投資助理 NLU 解析器。
你的任務是將使用者的自然語言訊息精確解析為結構化 JSON，絕對不要輸出任何多餘文字或 Markdown 標籤。

可用的意圖 (intent)：
- RECORD_BUY: 使用者回報手動買入/買進股票 (提取 ticker, price, shares)
- RECORD_SELL: 使用者回報手動賣出股票 (提取 ticker, price, shares)
- QUERY_PORTFOLIO: 查詢現有持股庫存、損益、部位
- QUERY_CONFIG: 查詢目前的系統參數、預算或停損利設定
- UPDATE_CONFIG: 修改策略參數 (提取 config_field: 'budget' | 'take_profit' | 'stop_loss' | 'profit_alert', 與數值 config_value)
- MARKET_REPORT: 詢問今日大盤、盤勢、市場快報或宏觀分析
- BUY_SCAN: 要求買入建議、推薦標的、買入掃描
- SELL_CHECK: 要求檢查庫存是否該賣、賣出檢查、風控檢查
- MENU: 要求開啟功能選單
- HELP: 詢問使用說明或指令列表
- UNKNOWN: 其他閒聊或無關內容

規則：
1. 若提到常見公司名稱（如台積電），請盡量轉成股票代號（如 2330）。
2. 若提到「張」，請換算為股數（1張 = 1000股；若提到零股或未提張則直接取股數）。
3. 停利/停損/漲幅若為百分比，請換算為小數（如 -20% -> -0.2，15% -> 0.15；停損必須為負數）。
4. 只回傳純 JSON，格式如下：
{
  "intent": "RECORD_BUY",
  "confidence": 0.95,
  "ticker": "2330",
  "price": 1050.0,
  "shares": 8.0,
  "config_field": null,
  "config_value": null,
  "explanation": "手動買入台積電8股於1050元"
}"""

    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        self.endpoint_url = (endpoint_url or settings.SLM_ENDPOINT_URL).rstrip("/")
        self.model_name = model_name or settings.SLM_MODEL
        self.api_key = api_key or settings.SLM_API_KEY
        self.timeout = timeout or settings.SLM_TIMEOUT_SECONDS

    def _get_chat_url(self) -> str:
        if self.endpoint_url.endswith("/v1"):
            return f"{self.endpoint_url}/chat/completions"
        return f"{self.endpoint_url}/v1/chat/completions"

    def _resolve_ticker(self, ticker_or_name: Optional[str]) -> Optional[str]:
        if not ticker_or_name:
            return None
        cleaned = ticker_or_name.strip()
        # 優先透過已知名稱對照表轉換
        if cleaned in self.KNOWN_TICKER_MAP:
            return self.KNOWN_TICKER_MAP[cleaned]
        for name, code in self.KNOWN_TICKER_MAP.items():
            if name in cleaned:
                return code
        return cleaned.upper()

    async def is_available(self) -> bool:
        """檢查 SLM 服務端點是否在線可用"""
        if not settings.SLM_ENABLED:
            return False
        health_url = self.endpoint_url.replace("/v1", "") + "/health"
        models_url = f"{self.endpoint_url}/models" if self.endpoint_url.endswith("/v1") else f"{self.endpoint_url}/v1/models"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(models_url, headers=headers)
                if resp.status_code == 200:
                    return True
                # 若 /models 不通嘗試 /health
                resp2 = await client.get(health_url)
                return resp2.status_code == 200
        except Exception:
            return False

    async def parse_intent(self, user_text: str) -> NLUIntentResult:
        """
        利用 SLM 推論使用者意圖並提取結構化槽位
        """
        raw_text = user_text.strip()
        default_res = NLUIntentResult(intent=NLUIntent.UNKNOWN, raw_text=raw_text)

        if not raw_text:
            return default_res

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": raw_text},
            ],
            "temperature": 0.1,
            "max_tokens": 256,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        url = self._get_chat_url()
        try:
            logger.info(f"[SLMService] Calling SLM at {url} (model: {self.model_name}) for query: '{raw_text}'")
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload, headers=headers)
                if response.status_code != 200:
                    logger.warning(
                        f"[SLMService] SLM endpoint returned HTTP {response.status_code}: {response.text}"
                    )
                    return default_res

                data = response.json()
                content = data["choices"][0]["message"]["content"].strip()

                # 清理可能的 markdown code block 標籤 (```json ... ```)
                json_str = content
                if "```" in content:
                    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
                    if match:
                        json_str = match.group(1)

                parsed = json.loads(json_str)
                intent_str = parsed.get("intent", "UNKNOWN").upper()
                try:
                    intent_enum = NLUIntent(intent_str)
                except ValueError:
                    intent_enum = NLUIntent.UNKNOWN

                raw_ticker = parsed.get("ticker")
                resolved_ticker = self._resolve_ticker(raw_ticker)

                # 處理價格與股數數值轉換
                price = None
                if parsed.get("price") is not None:
                    try:
                        price = float(parsed["price"])
                    except (ValueError, TypeError):
                        pass

                shares = None
                if parsed.get("shares") is not None:
                    try:
                        shares = float(parsed["shares"])
                    except (ValueError, TypeError):
                        pass

                # 處理配置值轉換
                cfg_val = parsed.get("config_value")
                if cfg_val is not None:
                    if isinstance(cfg_val, list):
                        try:
                            cfg_val = [float(x) for x in cfg_val]
                        except (ValueError, TypeError):
                            pass
                    else:
                        try:
                            cfg_val = float(cfg_val)
                        except (ValueError, TypeError):
                            pass

                confidence = float(parsed.get("confidence", 0.9))

                result = NLUIntentResult(
                    intent=intent_enum,
                    confidence=confidence,
                    ticker=resolved_ticker,
                    price=price,
                    shares=shares,
                    config_field=parsed.get("config_field"),
                    config_value=cfg_val,
                    raw_text=raw_text,
                    explanation=parsed.get("explanation"),
                )
                logger.info(
                    f"[SLMService] Parsed intent: {result.intent} (confidence: {result.confidence:.2f}, ticker: {result.ticker})"
                )
                return result

        except httpx.RequestError as req_err:
            logger.warning(f"[SLMService] Network request failed to SLM endpoint: {req_err}")
            return default_res
        except json.JSONDecodeError as json_err:
            logger.warning(f"[SLMService] Failed to decode JSON from SLM output: {json_err}")
            return default_res
        except Exception as e:
            logger.error(f"[SLMService] Unexpected error during SLM intent parsing: {e}")
            return default_res
