import pytest
import re


def test_regex_parsers():
    # 測試預算解析
    match = re.match(r"^(?:設定)?預算\s*[:：]?\s*(\d+)$", "設定預算 150000")
    assert match is not None
    assert int(match.group(1)) == 150000

    match2 = re.match(r"^(?:設定)?預算\s*[:：]?\s*(\d+)$", "預算: 80000")
    assert match2 is not None
    assert int(match2.group(1)) == 80000

    # 測試停利解析 (雙值)
    tp_match = re.match(r"^(?:設定)?停利\s*[:：]?\s*([\d.]+%?)(?:\s+([\d.]+%?))?$", "設定停利 0.12 0.25")
    assert tp_match is not None
    assert tp_match.group(1) == "0.12"
    assert tp_match.group(2) == "0.25"

    # 測試停損解析
    sl_match = re.match(r"^(?:設定)?停損\s*[:：]?\s*(-?[\d.]+%?)$", "設定停損 -0.4")
    assert sl_match is not None
    assert sl_match.group(1) == "-0.4"

    # 測試買入指令
    trade_buy = re.match(r"^(買入|買進|賣出)\s+([A-Za-z0-9.]+)\s+([\d.]+)\s+([\d.]+)$", "買入 2330 1050 8")
    assert trade_buy is not None
    assert trade_buy.group(1) == "買入"
    assert trade_buy.group(2) == "2330"
    assert float(trade_buy.group(3)) == 1050.0
    assert float(trade_buy.group(4)) == 8.0

    # 測試賣出指令
    trade_sell = re.match(r"^(買入|買進|賣出)\s+([A-Za-z0-9.]+)\s+([\d.]+)\s+([\d.]+)$", "賣出 6442 1750 40")
    assert trade_sell is not None
    assert trade_sell.group(1) == "賣出"
    assert trade_sell.group(2) == "6442"
    assert float(trade_sell.group(3)) == 1750.0
    assert float(trade_sell.group(4)) == 40.0

    # 測試漲幅通知指令解析
    profit_match1 = re.match(r"^(?:設定)?(?:漲幅通知|漲幅提醒|漲幅)\s*[:：]?\s*(\d+(?:\.\d+)?%?)$", "設定漲幅通知 5%")
    assert profit_match1 is not None
    assert profit_match1.group(1) == "5%"

    profit_match2 = re.match(r"^(?:設定)?(?:漲幅通知|漲幅提醒|漲幅)\s*[:：]?\s*(\d+(?:\.\d+)?%?)$", "漲幅: 0.08")
    assert profit_match2 is not None
    assert profit_match2.group(1) == "0.08"
