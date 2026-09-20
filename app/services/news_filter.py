"""新聞實質利多預篩器（規則基線）。

目的有兩個，優先序依序為：

1. **建立對照基線**：任何 SLM／模型方案都必須先證明自己贏得過這組關鍵字規則，
   否則不值得為它付出訓練、映像體積與冷啟動的代價。基線的 recall／precision
   由 `scripts/eval_news_filter.py` 量測。
2. **節省 Gemini 呼叫**：僅在 `NEWS_FILTER_MODE=enforce` 下才真的淘汰標的。

風險控管（重要）：
    漏掉一檔真利多的代價，遠大於省下的一次 Gemini 呼叫。因此本模組刻意偏向
    **高 recall**：詞表寬鬆、任一命中即通過，且預設為 `shadow` 模式不淘汰任何標的。
    提高 precision（少放行）永遠是次要目標。
"""

import re
from dataclasses import dataclass, field
from typing import List

# 實質基本面利多詞彙。刻意寬鬆：寧可多放行給 Gemini，也不要漏掉。
SUBSTANTIVE_TERMS: List[str] = [
    # 營收與獲利
    "營收", "營業額", "月營收", "季營收", "年營收", "獲利", "淨利", "毛利", "毛利率",
    "EPS", "每股盈餘", "稅後", "超預期", "優於預期", "創高", "新高", "成長", "增長",
    "轉盈", "虧轉盈", "獲利率",
    # 訂單與需求
    "訂單", "大單", "追單", "急單", "拉貨", "出貨", "供不應求", "滿載", "客戶認證",
    "打入", "供應鏈", "採用", "導入", "得標", "簽約", "合作",
    # 產能與投資
    "擴產", "產能", "建廠", "新廠", "投產", "量產", "資本支出", "擴充", "投資",
    # 價格與市場
    "漲價", "調價", "報價上漲", "市占", "市佔", "份額",
    # 法人與公司事件
    "法說", "法說會", "財報", "股東會", "目標價", "調升", "升評", "buy", "outperform",
    "評等", "看好", "受惠", "題材", "解禁", "認列",
    # 產業趨勢
    "AI", "人工智慧", "高速運算", "HPC", "資料中心", "車用", "光通訊", "半導體",
]

# 明確的負面／風險詞彙。僅用於 shadow 模式的觀測統計，不作為淘汰依據，
# 因為「有壞消息」不等於「沒有利多」，判斷孰輕孰重是 Gemini 的職責。
NEGATIVE_TERMS: List[str] = [
    "衰退", "下滑", "減少", "虧損", "轉虧", "砍單", "遞延", "延後", "停工", "裁員",
    "下修", "降評", "調降", "訴訟", "罰款", "召回", "違約", "警訊", "利空",
]

NO_NEWS_MARKER = "查無任何相關財經新聞"


@dataclass
class FilterVerdict:
    """預篩判定結果。"""

    passed: bool
    matched_terms: List[str] = field(default_factory=list)
    negative_terms: List[str] = field(default_factory=list)
    reason: str = ""

    @property
    def summary(self) -> str:
        pos = ",".join(self.matched_terms[:6]) or "無"
        neg = ",".join(self.negative_terms[:4]) or "無"
        return f"passed={self.passed} 利多詞[{pos}] 風險詞[{neg}] ({self.reason})"


class NewsRelevanceFilter:
    """以關鍵字規則判斷新聞文本是否含實質基本面利多。"""

    def __init__(
        self,
        substantive_terms: List[str] | None = None,
        negative_terms: List[str] | None = None,
        min_hits: int = 1,
    ):
        self.substantive_terms = substantive_terms or SUBSTANTIVE_TERMS
        self.negative_terms = negative_terms or NEGATIVE_TERMS
        # 預設 1 次命中即通過，刻意保守以維持高 recall
        self.min_hits = max(1, min_hits)

    @staticmethod
    def _find(terms: List[str], haystack: str) -> List[str]:
        """回傳出現於文本中的詞彙（英文詞不分大小寫）。"""
        hits: List[str] = []
        lowered = haystack.lower()
        for term in terms:
            if re.fullmatch(r"[A-Za-z][A-Za-z\s]*", term):
                if term.lower() in lowered:
                    hits.append(term)
            elif term in haystack:
                hits.append(term)
        return hits

    def evaluate(self, news_text: str) -> FilterVerdict:
        text = (news_text or "").strip()

        if not text:
            return FilterVerdict(passed=False, reason="新聞文本為空")

        if NO_NEWS_MARKER in text:
            # 與 llm_service 的防幻覺規則 4 一致：查無新聞時 Gemini 必然判 HOLD/AVOID，
            # 這一類可安全淘汰而不損及 recall。
            return FilterVerdict(passed=False, reason="近 3 日查無新聞")

        matched = self._find(self.substantive_terms, text)
        negative = self._find(self.negative_terms, text)

        if len(matched) >= self.min_hits:
            return FilterVerdict(
                passed=True,
                matched_terms=matched,
                negative_terms=negative,
                reason=f"命中 {len(matched)} 個實質利多詞",
            )

        return FilterVerdict(
            passed=False,
            matched_terms=matched,
            negative_terms=negative,
            reason="未命中任何實質利多詞",
        )
