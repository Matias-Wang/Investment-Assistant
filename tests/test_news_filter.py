"""P0-5 關鍵字預篩基線的單元測試。

此基線的存在目的是「SLM 必須先贏過它才值得做」，因此測試重點在於：
recall 優先（真利多不可被漏掉）、查無新聞可安全淘汰、負面詞不得單獨造成淘汰。
"""

import pytest

from app.services.news_filter import (
    NO_NEWS_MARKER,
    FilterVerdict,
    NewsRelevanceFilter,
)


@pytest.fixture
def nf() -> NewsRelevanceFilter:
    return NewsRelevanceFilter()


# --- recall：實質利多必須通過 -------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "台積電 11 月營收創同期新高，年增 34%。",
        "光聖獲得北美客戶大單，明年出貨量上調。",
        "華通擴產計畫啟動，新廠預計明年第二季投產。",
        "聯發科毛利率優於預期，法人調升目標價。",
        "廣達 AI 伺服器出貨動能強勁，資料中心需求滿載。",
        "公司宣布調價，報價上漲反映成本轉嫁能力。",
        "外資出具報告給予 Outperform 評等。",
        "本季 EPS 超預期，稅後淨利倍增。",
    ],
)
def test_substantive_positive_news_passes(nf: NewsRelevanceFilter, text: str):
    verdict = nf.evaluate(text)
    assert verdict.passed, f"真利多被漏掉（recall 破口）：{text} -> {verdict.summary}"
    assert verdict.matched_terms


def test_english_terms_are_case_insensitive(nf: NewsRelevanceFilter):
    assert nf.evaluate("analyst maintains a BUY rating").passed
    assert nf.evaluate("ai demand remains solid").passed


# --- 可安全淘汰的情境 ---------------------------------------------------------

def test_no_news_marker_is_dropped(nf: NewsRelevanceFilter):
    """查無新聞時 Gemini 依防幻覺規則必然判 HOLD/AVOID，淘汰不損 recall。"""
    verdict = nf.evaluate(f"【近 3 日{NO_NEWS_MARKER}或市場報導】")
    assert verdict.passed is False
    assert "查無新聞" in verdict.reason


def test_empty_text_is_dropped(nf: NewsRelevanceFilter):
    assert nf.evaluate("").passed is False
    assert nf.evaluate("   ").passed is False


def test_pure_noise_is_dropped(nf: NewsRelevanceFilter):
    """無任何基本面內容的雜訊新聞應被判為不通過。"""
    verdict = nf.evaluate("董事長昨日出席公益路跑活動，與員工一同完成五公里。")
    assert verdict.passed is False
    assert verdict.matched_terms == []


# --- 負面詞的處理：不得單獨造成淘汰 -----------------------------------------

def test_negative_terms_alone_do_not_block_when_positive_exists(nf: NewsRelevanceFilter):
    """同時有利多與風險時必須放行，孰輕孰重由 Gemini 判斷，不是預篩的職責。"""
    text = "營收年增 20%，但匯率波動可能使毛利下滑。"
    verdict = nf.evaluate(text)
    assert verdict.passed is True
    assert verdict.matched_terms
    assert verdict.negative_terms, "風險詞應被記錄下來以供觀測"


def test_negative_only_news_is_dropped(nf: NewsRelevanceFilter):
    verdict = nf.evaluate("客戶砍單，產線停工，公司宣布裁員。")
    assert verdict.passed is False
    assert verdict.negative_terms


# --- 可調參數 ---------------------------------------------------------------

def test_min_hits_threshold_is_respected():
    strict = NewsRelevanceFilter(min_hits=3)
    # 僅命中一個利多詞
    assert strict.evaluate("公司營收公布。").passed is False
    # 命中多個
    assert strict.evaluate("營收成長，獲利超預期，訂單滿載。").passed is True


def test_min_hits_never_below_one():
    """min_hits=0 會讓任何文本都通過，等同關閉篩選，故強制下限為 1。"""
    assert NewsRelevanceFilter(min_hits=0).min_hits == 1


def test_custom_term_list_is_used():
    nf = NewsRelevanceFilter(substantive_terms=["登月"])
    assert nf.evaluate("公司宣布登月計畫").passed is True
    assert nf.evaluate("公司營收成長").passed is False


def test_verdict_summary_is_readable():
    v = FilterVerdict(passed=True, matched_terms=["營收"], negative_terms=["匯率"], reason="測試")
    assert "passed=True" in v.summary
    assert "營收" in v.summary
