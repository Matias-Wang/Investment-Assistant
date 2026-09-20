"""確定性口語解析器（P0-6）。

定位：在既有嚴格格式 regex 之後、選用的 SLM fallback 之前執行。
處理絕大多數日常口語記帳與參數修改，**延遲 0ms、成本 $0、可完整單元測試**。

之所以優先做這一層而不是先上模型：這些輸入的變化是有限且可枚舉的
（同義詞、中文數字、張/股、百分比），用規則解比用 1.5B 模型解更快、更便宜，
而且錯誤是可重現、可寫測試釘住的——模型的錯不是。

不處理的情況一律回傳 `UNKNOWN`，交給後續的 SLM 或預設引導訊息，絕不亂猜。
"""

import re
from typing import List, Optional, Tuple

from app.models.slm import NLUIntent, NLUIntentResult
from app.services.ticker_alias import TICKER_ALIASES, resolve_alias

# --- 文本正規化 --------------------------------------------------------------

# 刻意不收錄大寫數字（壹貳參…）：「參數」「參考」等常用詞會被誤判為數字 3，
# 而實務上沒有人會在 LINE 訊息裡打「買參張」。
CN_DIGITS = {
    "零": 0, "〇": 0, "一": 1, "二": 2, "兩": 2, "倆": 2,
    "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}

# 中文數字僅在緊接量詞或位於句尾時才展開。無條件展開會誤傷一般詞彙
# （「一下」→「1下」、「百貨」→「100貨」）。
CN_NUMBER_UNITS = "張股萬元塊%％倍成"

# 量詞前置條件仍擋不住的成語例外：「千萬別」不是金額，但「一千萬」「十萬」是。
CN_IDIOM_EXCEPTIONS = ("千萬別", "千萬不", "千萬要", "千萬沒")


def _to_halfwidth(text: str) -> str:
    """全形英數與標點轉半形（使用者從手機輸入常夾帶全形字元）。"""
    out = []
    for ch in text:
        code = ord(ch)
        if code == 0x3000:
            out.append(" ")
        elif 0xFF01 <= code <= 0xFF5E:
            out.append(chr(code - 0xFEE0))
        else:
            out.append(ch)
    return "".join(out)


def _cn_number_to_int(chunk: str) -> Optional[int]:
    """將「十二」「三十」「一百二十」等中文數字轉為整數；無法解析回傳 None。"""
    if not chunk:
        return None

    section, digit, seen = 0, 0, False
    for ch in chunk:
        if ch in CN_DIGITS:
            digit = CN_DIGITS[ch]
            seen = True
        elif ch in ("十", "百", "千"):
            scale = {"十": 10, "百": 100, "千": 1000}[ch]
            # 累加而非覆寫：「十二」= 10 + 2，「一百二十」= 100 + 20
            section += (digit or 1) * scale
            digit = 0
            seen = True
        else:
            return None
    if not seen:
        return None
    return section + digit


def _expand_cn_numbers(text: str) -> str:
    """把中文數字串換成阿拉伯數字，讓後續的數值 regex 能統一處理。

    僅在數字後緊接量詞（張/股/萬/元…）或位於句尾時才展開，以免誤傷一般詞彙。
    """
    pattern = re.compile(
        rf"[零〇一二兩倆三四五六七八九十百千]+(?=[{CN_NUMBER_UNITS}]|$)"
    )

    def repl(match: re.Match) -> str:
        run = match.group(0)
        # 成語例外：以命中位置往後取 3 字判斷，避免「千萬別」被讀成金額
        window = text[match.start() : match.start() + 3]
        if window in CN_IDIOM_EXCEPTIONS:
            return run
        value = _cn_number_to_int(run)
        return str(value) if value is not None else run

    return pattern.sub(repl, text)


def normalize(text: str) -> str:
    """正規化輸入：全形轉半形、中文數字轉阿拉伯數字、壓縮空白。"""
    t = _to_halfwidth(text or "").strip()
    t = _expand_cn_numbers(t)
    return re.sub(r"\s+", " ", t)


# --- 數值解析 ----------------------------------------------------------------

NUM = r"\d+(?:\.\d+)?"


def _apply_scale(value: float, suffix: str) -> float:
    """套用中文數量級單位。"""
    if "萬" in suffix:
        return value * 10000
    if "千" in suffix:
        return value * 1000
    return value


def _parse_ratio(raw: str) -> Optional[float]:
    """把「20%」「-20%」「0.2」「15 %」解析為小數比例。"""
    s = raw.strip().replace(" ", "")
    if not s:
        return None
    negative = s.startswith("-")
    s = s.lstrip("+-")
    is_percent = s.endswith("%")
    s = s.rstrip("%")
    try:
        val = float(s)
    except ValueError:
        return None
    # 百分號，或數值大於 1 都視為百分比（例如「停損改 20」意指 20%）
    if is_percent or val > 1.0:
        val = val / 100.0
    return -val if negative else val


def _extract_amount(text: str) -> Optional[float]:
    """抽出金額，支援「12萬」「120000」「12 萬元」。"""
    m = re.search(rf"({NUM})\s*([萬千])?\s*(?:元|塊)?", text)
    if not m:
        return None
    return _apply_scale(float(m.group(1)), m.group(2) or "")


# --- 標的與交易槽位 ----------------------------------------------------------

TW_CODE = r"\b\d{4}\b"
US_SYMBOL = r"\b[A-Z]{2,5}\b"
# 這些英文詞在訊息中不是股票代號，避免誤判
US_STOPWORDS = {"AI", "HPC", "ETF", "EPS", "USD", "TWD", "LINE", "OK", "APP", "IT"}


def _extract_ticker(text: str) -> Optional[str]:
    """從文本抽出股票代號：別名 → 4 位數台股代號 → 英文代號。"""
    lowered = text.lower()
    for alias in sorted(TICKER_ALIASES, key=len, reverse=True):
        if alias in text or alias in lowered:
            return TICKER_ALIASES[alias]

    m = re.search(TW_CODE, text)
    if m:
        return m.group(0)

    for m in re.finditer(US_SYMBOL, text):
        token = m.group(0)
        if token not in US_STOPWORDS:
            return token
    return None


def _extract_shares_and_price(text: str) -> Tuple[Optional[float], Optional[float]]:
    """抽出股數與價格。

    判讀依據為單位詞，而非位置，因此「1050 買 8 股」與「8 股 1050」都能解對：
      * 「N 張」→ N × 1000 股
      * 「N 股」→ N 股
      * 「N 元 / N 塊」→ 價格
      * 若已定出股數，剩下唯一的裸數字視為價格
    """
    shares: Optional[float] = None
    price: Optional[float] = None
    consumed: List[Tuple[int, int]] = []

    m = re.search(rf"({NUM})\s*張", text)
    if m:
        shares = float(m.group(1)) * 1000
        consumed.append(m.span())
    else:
        m = re.search(rf"({NUM})\s*股", text)
        if m:
            shares = float(m.group(1))
            consumed.append(m.span())

    m = re.search(rf"({NUM})\s*(?:元|塊)", text)
    if m:
        price = float(m.group(1))
        consumed.append(m.span())

    if price is None:
        # 找出未被股數/價格用掉的裸數字；排除 4 位數代號以免把 2330 當成價格
        for m in re.finditer(NUM, text):
            if any(start <= m.start() < end for start, end in consumed):
                continue
            token = m.group(0)
            if re.fullmatch(r"\d{4}", token) and token in TICKER_ALIASES.values():
                continue
            price = float(token)
            break

    return shares, price


# --- 意圖規則 ----------------------------------------------------------------

BUY_VERBS = r"(?:買入|買進|買了|買|購入|進場)"
SELL_VERBS = r"(?:賣出|賣掉|賣了|賣|出清|出場)"

PORTFOLIO_PAT = re.compile(
    r"(賺|賠|損益|庫存|持股|部位|現在多少|賺多少|賠多少|獲利多少|我的股票)"
)
CONFIG_QUERY_PAT = re.compile(r"(設定|參數|預算多少|停損多少|停利多少)")
MARKET_PAT = re.compile(r"(大盤|盤勢|盤面|行情|快報|市場報告|今天.*(盤|股市)|股市.*如何)")
BUY_SCAN_PAT = re.compile(r"(推薦|建議買|買什麼|該買|掃描|選股|有沒有.*買)")
SELL_CHECK_PAT = re.compile(r"(該賣|要賣|賣出檢查|風控|檢查.*賣|停損檢查)")
MENU_PAT = re.compile(r"(選單|功能表|控制面板|主選單)")
HELP_PAT = re.compile(r"(怎麼用|說明|使用方法|指令|教學|help)", re.IGNORECASE)

# 參數欄位關鍵字 → SystemConfig 欄位名
CONFIG_FIELDS = [
    (re.compile(r"(單次)?預算"), "budget"),
    (re.compile(r"停損"), "stop_loss"),
    (re.compile(r"停利|獲利了結"), "take_profit"),
    (re.compile(r"漲幅(通知|提醒)?"), "profit_alert"),
]
UPDATE_VERBS = re.compile(r"(改成|改為|改到|調整為|調成|設為|設成|設定為|更新為|改|設定)")

RULE_CONFIDENCE = 0.95


class RuleBasedNLU:
    """以規則解析口語訊息為 `NLUIntentResult`。無法確定時回傳 UNKNOWN。"""

    def parse(self, raw_text: str) -> NLUIntentResult:
        raw = (raw_text or "").strip()
        result = NLUIntentResult(intent=NLUIntent.UNKNOWN, raw_text=raw)
        if not raw:
            return result

        text = normalize(raw)

        # 1. 參數修改：需同時出現欄位關鍵字與修改動詞，避免與「查詢設定」混淆
        for pattern, field_name in CONFIG_FIELDS:
            if pattern.search(text) and UPDATE_VERBS.search(text):
                value = self._parse_config_value(field_name, text)
                if value is not None:
                    return NLUIntentResult(
                        intent=NLUIntent.UPDATE_CONFIG,
                        confidence=RULE_CONFIDENCE,
                        config_field=field_name,
                        config_value=value,
                        raw_text=raw,
                        explanation=f"規則解析：更新 {field_name}",
                    )

        # 2. 買入 / 賣出記帳
        if re.search(BUY_VERBS, text) and not re.search(BUY_SCAN_PAT, text):
            ticker = _extract_ticker(text)
            if ticker:
                shares, price = _extract_shares_and_price(text)
                return NLUIntentResult(
                    intent=NLUIntent.RECORD_BUY,
                    confidence=RULE_CONFIDENCE,
                    ticker=ticker,
                    price=price,
                    shares=shares,
                    raw_text=raw,
                    explanation="規則解析：買入記帳",
                )

        if re.search(SELL_VERBS, text) and not SELL_CHECK_PAT.search(text):
            ticker = _extract_ticker(text)
            if ticker:
                shares, price = _extract_shares_and_price(text)
                return NLUIntentResult(
                    intent=NLUIntent.RECORD_SELL,
                    confidence=RULE_CONFIDENCE,
                    ticker=ticker,
                    price=price,
                    shares=shares,
                    raw_text=raw,
                    explanation="規則解析：賣出記帳",
                )

        # 3. 純查詢與觸發類意圖（順序即優先序）
        for pattern, intent in (
            (MENU_PAT, NLUIntent.MENU),
            (SELL_CHECK_PAT, NLUIntent.SELL_CHECK),
            (BUY_SCAN_PAT, NLUIntent.BUY_SCAN),
            (MARKET_PAT, NLUIntent.MARKET_REPORT),
            (PORTFOLIO_PAT, NLUIntent.QUERY_PORTFOLIO),
            (CONFIG_QUERY_PAT, NLUIntent.QUERY_CONFIG),
            (HELP_PAT, NLUIntent.HELP),
        ):
            if pattern.search(text):
                return NLUIntentResult(
                    intent=intent,
                    confidence=RULE_CONFIDENCE,
                    raw_text=raw,
                    explanation=f"規則解析：{intent.value}",
                )

        return result

    @staticmethod
    def _parse_config_value(field_name: str, text: str):
        """依欄位型別解析數值：預算為金額，其餘為比例。"""
        if field_name == "budget":
            return _extract_amount(text)

        if field_name == "take_profit":
            # 停利可為區間：「停利改 10% 20%」
            ratios = [
                r for r in (
                    _parse_ratio(m.group(0))
                    for m in re.finditer(rf"-?{NUM}\s*%?", text)
                ) if r is not None
            ]
            # 濾掉可能夾帶的代號等雜訊值
            ratios = [r for r in ratios if 0 < abs(r) <= 10]
            if len(ratios) >= 2:
                return [min(ratios[0], ratios[1]), max(ratios[0], ratios[1])]
            return ratios[0] if ratios else None

        m = re.search(rf"(-?{NUM}\s*%?)", text)
        if not m:
            return None
        ratio = _parse_ratio(m.group(1))
        if ratio is None:
            return None
        if field_name == "stop_loss":
            # 停損一律為負值，使用者說「停損改 20%」意指 -20%
            return -abs(ratio)
        return abs(ratio)
