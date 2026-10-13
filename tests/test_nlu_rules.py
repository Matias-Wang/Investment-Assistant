"""P0-6 確定性口語解析器的單元測試。

這一層的價值就在於「錯誤可被釘住」：每個口語變體都是一條可重現的斷言，
模型做不到這件事。SLM_PLAN v2 的 Phase 3 是否需要存在，由這裡的覆蓋率決定。
"""

import pytest

from app.models.slm import NLUIntent
from app.services.nlu_rules import (
    RuleBasedNLU,
    _cn_number_to_int,
    _parse_ratio,
    normalize,
)


@pytest.fixture
def nlu() -> RuleBasedNLU:
    return RuleBasedNLU()


# --- 文本正規化 --------------------------------------------------------------

def test_fullwidth_is_converted():
    assert normalize("買入　２３３０") == "買入 2330"
    assert "%" in normalize("停損改－２０％")


def test_chinese_numerals_expanded():
    assert normalize("買一張台積電") == "買1張台積電"
    assert normalize("預算改成十二萬") == "預算改成12萬"
    assert normalize("賣兩張") == "賣2張"


def test_whitespace_collapsed():
    assert normalize("  買入    2330   ") == "買入 2330"


def test_compound_chinese_numerals():
    """回歸測試：「十二」曾被算成 2（累加寫成覆寫）。"""
    assert _cn_number_to_int("十二") == 12
    assert _cn_number_to_int("三十") == 30
    assert _cn_number_to_int("一百二十") == 120
    assert _cn_number_to_int("十") == 10
    assert _cn_number_to_int("兩") == 2


@pytest.mark.parametrize(
    "text",
    [
        "目前參數是多少",   # 「參」為三的大寫，曾被展開成「3數」
        "我參考一下",
        "百貨公司",
        "千萬別買",
        "看一下損益",
    ],
)
def test_chinese_numerals_not_expanded_without_unit(text):
    """回歸測試：無量詞的中文字不可被當成數字展開，否則會破壞一般詞彙。"""
    assert normalize(text) == text


def test_config_query_survives_normalization():
    """「參數」被誤展開時此意圖會整個消失，故單獨釘住。"""
    assert RuleBasedNLU().parse("目前參數是多少").intent == NLUIntent.QUERY_CONFIG


# --- 比例解析 ----------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("20%", 0.2),
        ("-20%", -0.2),
        ("0.15", 0.15),
        ("15", 0.15),      # 大於 1 視為百分比
        ("-0.5", -0.5),
        ("50 %", 0.5),
    ],
)
def test_parse_ratio(raw, expected):
    assert _parse_ratio(raw) == pytest.approx(expected)


# --- SLM_PLAN 文件所承諾的五個情境 -------------------------------------------

def test_natural_buy_with_price_and_shares(nlu: RuleBasedNLU):
    r = nlu.parse("剛在 1050 買進 8 股台積電")
    assert r.intent == NLUIntent.RECORD_BUY
    assert r.ticker == "2330"
    assert r.price == pytest.approx(1050.0)
    assert r.shares == pytest.approx(8.0)


def test_natural_sell(nlu: RuleBasedNLU):
    r = nlu.parse("賣掉 4 股台積電")
    assert r.intent == NLUIntent.RECORD_SELL
    assert r.ticker == "2330"
    assert r.shares == pytest.approx(4.0)


@pytest.mark.parametrize("text,ticker,shares,price", [
    ("賣出2313 成交價254.5 500股", "2313", 500, 254.5),
    ("賣出2330 1080 4股", "2330", 4, 1080),
    ("賣出 2330 4", "2330", 4, None),
])
def test_sell_report_slots(nlu, text, ticker, shares, price):
    result = nlu.parse(text)
    assert result.intent == NLUIntent.RECORD_SELL
    assert result.ticker == ticker
    assert result.shares == shares
    assert result.price == price


def test_update_budget_in_wan(nlu: RuleBasedNLU):
    r = nlu.parse("幫我把單次預算改成 12 萬")
    assert r.intent == NLUIntent.UPDATE_CONFIG
    assert r.config_field == "budget"
    assert r.config_value == pytest.approx(120000.0)


def test_update_stop_loss(nlu: RuleBasedNLU):
    r = nlu.parse("停損改 -20%")
    assert r.intent == NLUIntent.UPDATE_CONFIG
    assert r.config_field == "stop_loss"
    assert r.config_value == pytest.approx(-0.2)


def test_query_portfolio_colloquial(nlu: RuleBasedNLU):
    r = nlu.parse("看一下我現在賺多少")
    assert r.intent == NLUIntent.QUERY_PORTFOLIO


# --- 買入的其他口語變體 ------------------------------------------------------

@pytest.mark.parametrize(
    "text,ticker,shares,price",
    [
        ("用 1050 買了台積電 8 股", "2330", 8.0, 1050.0),
        ("台積電 8股 1050元 買入", "2330", 8.0, 1050.0),
        ("買一張台積電", "2330", 1000.0, None),
        ("剛買進兩張聯發科", "2454", 2000.0, None),
        ("在 180 買了 10 股輝達", "NVDA", 10.0, 180.0),
        # 2382 為代號而非價格，不可被當成價格抽走
        ("買進 2382 廣達 50 股", "2382", 50.0, None),
    ],
)
def test_buy_variants(nlu: RuleBasedNLU, text, ticker, shares, price):
    r = nlu.parse(text)
    assert r.intent == NLUIntent.RECORD_BUY, text
    assert r.ticker == ticker
    assert r.shares == pytest.approx(shares)
    if price is None:
        assert r.price is None
    else:
        assert r.price == pytest.approx(price)


def test_zhang_converts_to_thousand_shares(nlu: RuleBasedNLU):
    r = nlu.parse("買 3 張 2330")
    assert r.shares == pytest.approx(3000.0)


def test_buy_without_slots_still_identifies_ticker(nlu: RuleBasedNLU):
    """槽位不全時仍回報意圖與標的，由對話層追問其餘欄位。"""
    r = nlu.parse("我想買台積電")
    assert r.intent == NLUIntent.RECORD_BUY
    assert r.ticker == "2330"
    assert r.price is None and r.shares is None


# --- 參數修改的其他變體 ------------------------------------------------------

def test_take_profit_range(nlu: RuleBasedNLU):
    r = nlu.parse("停利改成 10% 20%")
    assert r.config_field == "take_profit"
    assert r.config_value == [pytest.approx(0.1), pytest.approx(0.2)]


def test_take_profit_single_value(nlu: RuleBasedNLU):
    r = nlu.parse("停利設為 15%")
    assert r.config_field == "take_profit"
    assert r.config_value == pytest.approx(0.15)


def test_profit_alert(nlu: RuleBasedNLU):
    r = nlu.parse("漲幅通知改 5%")
    assert r.config_field == "profit_alert"
    assert r.config_value == pytest.approx(0.05)


def test_stop_loss_is_forced_negative(nlu: RuleBasedNLU):
    """使用者說「停損改 30%」意指 -30%，不可寫成 +0.3。"""
    r = nlu.parse("停損改 30%")
    assert r.config_value == pytest.approx(-0.3)


def test_budget_plain_number(nlu: RuleBasedNLU):
    r = nlu.parse("預算改成 150000")
    assert r.config_value == pytest.approx(150000.0)


# --- 其他意圖 ----------------------------------------------------------------

@pytest.mark.parametrize(
    "text,intent",
    [
        ("今天大盤如何", NLUIntent.MARKET_REPORT),
        ("盤勢怎麼樣", NLUIntent.MARKET_REPORT),
        ("有沒有推薦買的股票", NLUIntent.BUY_SCAN),
        ("幫我掃描一下", NLUIntent.BUY_SCAN),
        ("檢查哪些該賣了", NLUIntent.SELL_CHECK),
        ("幫我做風控檢查", NLUIntent.SELL_CHECK),
        ("我的持股現在怎樣", NLUIntent.QUERY_PORTFOLIO),
        ("目前參數是多少", NLUIntent.QUERY_CONFIG),
        ("打開選單", NLUIntent.MENU),
        ("這個怎麼用", NLUIntent.HELP),
    ],
)
def test_other_intents(nlu: RuleBasedNLU, text, intent):
    assert nlu.parse(text).intent == intent


def test_buy_scan_not_confused_with_record_buy(nlu: RuleBasedNLU):
    """「該買什麼」是要掃描，不是要記帳，兩者不可混淆。"""
    r = nlu.parse("現在該買什麼")
    assert r.intent == NLUIntent.BUY_SCAN


def test_sell_check_not_confused_with_record_sell(nlu: RuleBasedNLU):
    r = nlu.parse("有哪些該賣了嗎")
    assert r.intent == NLUIntent.SELL_CHECK


# --- 不亂猜：無法辨識一律 UNKNOWN -------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "今天天氣真好",
        "謝謝你",
        "哈哈哈",
        "隨便聊聊",
    ],
)
def test_unrecognized_returns_unknown(nlu: RuleBasedNLU, text):
    r = nlu.parse(text)
    assert r.intent == NLUIntent.UNKNOWN


def test_raw_text_is_preserved(nlu: RuleBasedNLU):
    raw = "剛在 1050 買進 8 股台積電"
    assert nlu.parse(raw).raw_text == raw


def test_confidence_is_high_for_rule_match(nlu: RuleBasedNLU):
    """規則是確定性的，信心度不應低於執行門檻 0.5。"""
    assert nlu.parse("賣掉 4 股台積電").confidence >= 0.5
