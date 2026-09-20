import urllib.parse
from typing import Dict, Any, List, Optional
from app.models.domain import SellAlert, BuyCandidate, PortfolioValuation, SystemConfig
from app.models.llm import BuyRecommendation


def _safe_text(val: Any, limit: int = 120, default: str = "-") -> str:
    """防護空字串與超長字串，避免 LINE Flex Message 產生 400 錯誤"""
    if val is None:
        return default
    s = str(val).strip()
    if not s:
        return default
    if len(s) > limit:
        return s[: limit - 3] + "..."
    return s


def build_buy_flex_message(rec: BuyRecommendation, candidate: BuyCandidate) -> Dict[str, Any]:
    """
    建立買入建議 Flex Message (附「手動已買入」按鈕)
    """
    reasons_boxes = []
    for r in (rec.reasons or [])[:3]:
        safe_r = _safe_text(r, limit=100)
        reasons_boxes.append({
            "type": "text",
            "text": f"• {safe_r}",
            "size": "xs",
            "color": "#555555",
            "wrap": True
        })

    risk_boxes = []
    for risk in (rec.risk_factors or [])[:2]:
        safe_risk = _safe_text(risk, limit=100)
        risk_boxes.append({
            "type": "text",
            "text": f"• {safe_risk}",
            "size": "xs",
            "color": "#e05252",
            "wrap": True
        })

    contents = {
        "type": "bubble",
        "size": "mega",
        "header": {
            "type": "box",
            "layout": "vertical",
            "backgroundColor": "#0F9D58",
            "paddingAll": "15px",
            "contents": [
                {
                    "type": "text",
                    "text": "💡 買入策略建議",
                    "color": "#FFFFFF",
                    "weight": "bold",
                    "size": "sm"
                },
                {
                    "type": "text",
                    "text": f"{rec.ticker} {rec.company_name}",
                    "color": "#FFFFFF",
                    "weight": "bold",
                    "size": "xl",
                    "margin": "xs"
                },
                {
                    "type": "text",
                    "text": f"決策: {rec.decision} | 信心度: {int(rec.confidence_score * 100)}%",
                    "color": "#E8F5E9",
                    "size": "xs"
                }
            ]
        },
        "body": {
            "type": "box",
            "layout": "vertical",
            "spacing": "md",
            "contents": [
                {
                    "type": "box",
                    "layout": "horizontal",
                    "contents": [
                        {"type": "text", "text": "現價", "size": "sm", "color": "#888888", "flex": 2},
                        {"type": "text", "text": f"${candidate.current_price:.2f}", "size": "sm", "weight": "bold", "color": "#222222", "flex": 3},
                        {"type": "text", "text": "20MA", "size": "sm", "color": "#888888", "flex": 2},
                        {"type": "text", "text": f"${candidate.sma_20:.2f}", "size": "sm", "weight": "bold", "color": "#0F9D58", "flex": 3}
                    ]
                },
                {
                    "type": "box",
                    "layout": "horizontal",
                    "contents": [
                        {"type": "text", "text": "建議零股", "size": "sm", "color": "#888888", "flex": 2},
                        {"type": "text", "text": f"{candidate.target_shares:,} 股", "size": "sm", "weight": "bold", "color": "#1976D2", "flex": 3},
                        {"type": "text", "text": "預估花費", "size": "sm", "color": "#888888", "flex": 2},
                        {"type": "text", "text": f"${candidate.estimated_amount:,.0f}", "size": "sm", "weight": "bold", "color": "#222222", "flex": 3}
                    ]
                },
                {"type": "separator"},
                {
                    "type": "box",
                    "layout": "vertical",
                    "backgroundColor": "#F7F9FA",
                    "paddingAll": "10px",
                    "cornerRadius": "6px",
                    "contents": [
                        {"type": "text", "text": "📰 新聞情緒與基本面摘要 (3日內):", "size": "xs", "weight": "bold", "color": "#333333"},
                        {"type": "text", "text": rec.sentiment_summary, "size": "xs", "color": "#555555", "wrap": True, "margin": "xs"}
                    ]
                },
                {
                    "type": "box",
                    "layout": "vertical",
                    "spacing": "xs",
                    "contents": [
                        {"type": "text", "text": "✅ 支持理由 (新聞依據):", "size": "xs", "weight": "bold", "color": "#2E7D32"}
                    ] + reasons_boxes
                }
            ]
        },
        "footer": {
            "type": "box",
            "layout": "vertical",
            "spacing": "sm",
            "contents": [
                {
                    "type": "button",
                    "style": "primary",
                    "color": "#0F9D58",
                    "height": "sm",
                    "action": {
                        "type": "postback",
                        "label": "手動已買入",
                        "data": urllib.parse.urlencode({
                            "action": "buy_click",
                            "ticker": rec.ticker,
                            "shares": str(candidate.target_shares),
                            "price": str(candidate.current_price),
                        }),
                        "displayText": f"我已手動買入 {rec.ticker} (建議: {candidate.target_shares}股 @ {candidate.current_price})"
                    }
                }
            ]
        }
    }

    if risk_boxes:
        contents["body"]["contents"].append({
            "type": "box",
            "layout": "vertical",
            "spacing": "xs",
            "contents": [
                {"type": "text", "text": "⚠️ 風險提示:", "size": "xs", "weight": "bold", "color": "#D32F2F"}
            ] + risk_boxes
        })

    return contents


def build_sell_flex_message(alert: SellAlert) -> Dict[str, Any]:
    """
    建立賣出或漲幅警示 Flex Message (附「手動已賣出」按鈕)
    """
    if alert.alert_type == "PROFIT_ALERT":
        header_color = "#2E7D32"
        title_text = "🚀 持股漲幅通知"
        pnl_color = "#2E7D32"
        guidance_text = "標的漲幅已突破設定門檻，請評估是否分批停利或續抱。"
    elif alert.alert_type == "TAKE_PROFIT":
        header_color = "#E65100"
        title_text = "💰 停利監控警示"
        pnl_color = "#E65100"
        guidance_text = "依據停利策略設定，請評估是否手動執行賣出。"
    else:
        header_color = "#C62828"
        title_text = "🛑 停損監控警示"
        pnl_color = "#C62828"
        guidance_text = "依據停損策略設定，請評估是否手動執行停損賣出。"
    sign = "+" if alert.pnl_rate >= 0 else ""

    return {
        "type": "bubble",
        "size": "mega",
        "header": {
            "type": "box",
            "layout": "vertical",
            "backgroundColor": header_color,
            "paddingAll": "15px",
            "contents": [
                {
                    "type": "text",
                    "text": title_text,
                    "color": "#FFFFFF",
                    "weight": "bold",
                    "size": "sm"
                },
                {
                    "type": "text",
                    "text": f"{alert.ticker} {alert.name}",
                    "color": "#FFFFFF",
                    "weight": "bold",
                    "size": "xl",
                    "margin": "xs"
                },
                {
                    "type": "text",
                    "text": f"持股 {alert.shares:,.0f} 股 | {alert.trigger_reason}",
                    "color": "#FFFFFFCC",
                    "size": "xs"
                }
            ]
        },
        "body": {
            "type": "box",
            "layout": "vertical",
            "spacing": "md",
            "contents": [
                {
                    "type": "box",
                    "layout": "horizontal",
                    "contents": [
                        {"type": "text", "text": "現價", "size": "sm", "color": "#888888", "flex": 2},
                        {"type": "text", "text": f"${alert.current_price:.2f}", "size": "sm", "weight": "bold", "color": "#222222", "flex": 3},
                        {"type": "text", "text": "平均成本", "size": "sm", "color": "#888888", "flex": 2},
                        {"type": "text", "text": f"${alert.average_cost:.2f}", "size": "sm", "weight": "bold", "color": "#555555", "flex": 3}
                    ]
                },
                {
                    "type": "box",
                    "layout": "horizontal",
                    "contents": [
                        {"type": "text", "text": "損益率", "size": "sm", "color": "#888888", "flex": 2},
                        {"type": "text", "text": f"{sign}{alert.pnl_rate * 100:.2f}%", "size": "lg", "weight": "bold", "color": pnl_color, "flex": 3},
                        {"type": "text", "text": "預估損益", "size": "sm", "color": "#888888", "flex": 2},
                        {"type": "text", "text": f"{sign}${alert.pnl_amount:,.0f}", "size": "sm", "weight": "bold", "color": pnl_color, "flex": 3}
                    ]
                },
                {"type": "separator"},
                {
                    "type": "text",
                    "text": guidance_text,
                    "size": "xs",
                    "color": "#666666",
                    "wrap": True
                }
            ]
        },
        "footer": {
            "type": "box",
            "layout": "vertical",
            "spacing": "sm",
            "contents": [
                {
                    "type": "button",
                    "style": "primary",
                    "color": header_color,
                    "height": "sm",
                    "action": {
                        "type": "postback",
                        "label": "手動已賣出",
                        "data": urllib.parse.urlencode({
                            "action": "sell_click",
                            "ticker": alert.ticker,
                            "shares": str(alert.shares),
                            "price": str(alert.current_price),
                        }),
                        "displayText": f"我已手動賣出 {alert.ticker} ({alert.shares}股 @ {alert.current_price})"
                    }
                }
            ]
        }
    }


def build_portfolio_flex_message(val: PortfolioValuation, config: SystemConfig) -> Dict[str, Any]:
    """
    建立庫存與損益查詢 Flex Message
    """
    total_sign = "+" if val.total_unrealized_pnl >= 0 else ""
    total_color = "#E65100" if val.total_unrealized_pnl >= 0 else "#C62828"

    position_boxes: List[Dict[str, Any]] = []
    display_items = val.items[:20]
    for item in display_items:
        pnl_sign = "+" if item.pnl_rate >= 0 else ""
        item_pnl_color = "#E65100" if item.pnl_rate >= 0 else "#C62828"

        position_boxes.append({
            "type": "box",
            "layout": "vertical",
            "margin": "md",
            "backgroundColor": "#F8F9FA",
            "paddingAll": "10px",
            "cornerRadius": "6px",
            "contents": [
                {
                    "type": "box",
                    "layout": "horizontal",
                    "contents": [
                        {"type": "text", "text": f"{item.ticker} {item.name}", "size": "sm", "weight": "bold", "color": "#212121", "flex": 4},
                        {"type": "text", "text": f"{pnl_sign}{item.pnl_rate * 100:.2f}%", "size": "sm", "weight": "bold", "color": item_pnl_color, "align": "end", "flex": 3}
                    ]
                },
                {
                    "type": "box",
                    "layout": "horizontal",
                    "margin": "xs",
                    "contents": [
                        {"type": "text", "text": f"股數: {item.shares:,.0f} | 成本: ${item.average_cost:.2f}", "size": "xs", "color": "#757575", "flex": 4},
                        {"type": "text", "text": f"現價: ${item.current_price:.2f}", "size": "xs", "color": "#424242", "align": "end", "flex": 3}
                    ]
                }
            ]
        })

    if len(val.items) > 20:
        position_boxes.append({
            "type": "text",
            "text": f"... 另有 {len(val.items) - 20} 筆持股，請至管理後台查看",
            "size": "xs",
            "color": "#888888",
            "align": "center",
            "margin": "md"
        })

    if not position_boxes:
        position_boxes.append({
            "type": "text",
            "text": "目前無庫存持股",
            "size": "sm",
            "color": "#888888",
            "align": "center",
            "margin": "md"
        })

    return {
        "type": "bubble",
        "size": "mega",
        "header": {
            "type": "box",
            "layout": "vertical",
            "backgroundColor": "#283593",
            "paddingAll": "15px",
            "contents": [
                {
                    "type": "text",
                    "text": "📊 投資組合庫存損益",
                    "color": "#FFFFFF",
                    "weight": "bold",
                    "size": "md"
                },
                {
                    "type": "text",
                    "text": f"未實現總損益: {total_sign}{val.total_unrealized_pnl_rate * 100:.2f}% ({total_sign}${val.total_unrealized_pnl:,.0f})",
                    "color": "#E8EAF6",
                    "size": "xs",
                    "margin": "xs"
                }
            ]
        },
        "body": {
            "type": "box",
            "layout": "vertical",
            "contents": [
                {
                    "type": "box",
                    "layout": "horizontal",
                    "contents": [
                        {"type": "text", "text": "總建倉成本", "size": "xs", "color": "#757575"},
                        {"type": "text", "text": f"${val.total_cost:,.0f}", "size": "xs", "weight": "bold", "color": "#212121", "align": "end"}
                    ]
                },
                {
                    "type": "box",
                    "layout": "horizontal",
                    "margin": "xs",
                    "contents": [
                        {"type": "text", "text": "總現行市值", "size": "xs", "color": "#757575"},
                        {"type": "text", "text": f"${val.total_market_value:,.0f}", "size": "xs", "weight": "bold", "color": "#212121", "align": "end"}
                    ]
                },
                {"type": "separator", "margin": "md"},
                {
                    "type": "box",
                    "layout": "vertical",
                    "margin": "sm",
                    "contents": position_boxes
                }
            ]
        },
        "footer": {
            "type": "box",
            "layout": "vertical",
            "contents": [
                {
                    "type": "text",
                    "text": f"⚙️ 設定: 預算 ${config.trade_budget_per_order:,.0f} | 停利 {config.take_profit_min*100:.0f}%~{config.take_profit_max*100:.0f}% | 停損 {config.stop_loss*100:.0f}%",
                    "size": "xxs",
                    "color": "#9E9E9E",
                    "align": "center",
                    "wrap": True
                }
            ]
        }
    }


def build_menu_flex_message() -> Dict[str, Any]:
    """
    建立主功能控制面板 Flex Message
    """
    return {
        "type": "bubble",
        "size": "mega",
        "header": {
            "type": "box",
            "layout": "vertical",
            "backgroundColor": "#1A237E",
            "paddingAll": "15px",
            "contents": [
                {
                    "type": "text",
                    "text": "🎛️ 投資助理快捷控制面板",
                    "color": "#FFFFFF",
                    "weight": "bold",
                    "size": "md"
                },
                {
                    "type": "text",
                    "text": "點擊下方按鈕即可快速執行各項功能",
                    "color": "#C5CAE9",
                    "size": "xs",
                    "margin": "xs"
                }
            ]
        },
        "body": {
            "type": "box",
            "layout": "vertical",
            "spacing": "sm",
            "contents": [
                {
                    "type": "button",
                    "style": "primary",
                    "color": "#3949AB",
                    "height": "sm",
                    "action": {
                        "type": "message",
                        "label": "📊 查詢庫存與即時損益",
                        "text": "查詢庫存"
                    }
                },
                {
                    "type": "button",
                    "style": "primary",
                    "color": "#0F9D58",
                    "height": "sm",
                    "action": {
                        "type": "message",
                        "label": "💡 執行買入策略掃描 (AI)",
                        "text": "買入建議"
                    }
                },
                {
                    "type": "button",
                    "style": "primary",
                    "color": "#E65100",
                    "height": "sm",
                    "action": {
                        "type": "message",
                        "label": "🚨 執行賣出停損利檢查",
                        "text": "賣出檢查"
                    }
                },
                {
                    "type": "button",
                    "style": "primary",
                    "color": "#6A1B9A",
                    "height": "sm",
                    "action": {
                        "type": "message",
                        "label": "📑 每日市場報告 (AI)",
                        "text": "市場報告"
                    }
                },
                {
                    "type": "button",
                    "style": "secondary",
                    "color": "#ECEFF1",
                    "height": "sm",
                    "action": {
                        "type": "message",
                        "label": "⚙️ 查詢當前策略參數",
                        "text": "查詢設定"
                    }
                },
                {
                    "type": "button",
                    "style": "secondary",
                    "color": "#ECEFF1",
                    "height": "sm",
                    "action": {
                        "type": "message",
                        "label": "📖 完整指令與操作說明",
                        "text": "說明"
                    }
                }
            ]
        }
    }


def build_market_report_flex_message(report: Any, indices: List[Any]) -> Dict[str, Any]:
    """
    建立每日市場快報 Flex Message
    """
    # 決定氛圍顏色
    sentiment_color = "#2E7D32"  # 綠色
    if "震盪" in report.market_sentiment:
        sentiment_color = "#E65100"  # 橘色
    elif "保守" in report.market_sentiment or "承壓" in report.market_sentiment:
        sentiment_color = "#C62828"  # 紅色

    # 指數小方塊
    index_boxes = []
    for idx in indices:
        sign = "+" if idx.change_percent >= 0 else ""
        color = "#C62828" if idx.change_percent >= 0 else "#2E7D32"  # 台股習慣紅漲綠跌
        index_boxes.append({
            "type": "box",
            "layout": "vertical",
            "backgroundColor": "#F1F3F4",
            "cornerRadius": "6px",
            "paddingAll": "6px",
            "flex": 1,
            "contents": [
                {"type": "text", "text": idx.name, "size": "xxs", "color": "#666666", "align": "center"},
                {"type": "text", "text": f"${idx.price:,.1f}", "size": "xs", "weight": "bold", "color": "#212121", "align": "center"},
                {"type": "text", "text": f"{sign}{idx.change_percent:.2f}%", "size": "xxs", "weight": "bold", "color": color, "align": "center"}
            ]
        })

    # 將指數分為 2 列，每列 2 個
    grid_rows = []
    if len(index_boxes) >= 2:
        grid_rows.append({
            "type": "box",
            "layout": "horizontal",
            "spacing": "xs",
            "contents": index_boxes[:2]
        })
    if len(index_boxes) >= 4:
        grid_rows.append({
            "type": "box",
            "layout": "horizontal",
            "spacing": "xs",
            "margin": "xs",
            "contents": index_boxes[2:4]
        })

    # 頭條條目
    headline_boxes = []
    for h in report.key_headlines[:4]:
        headline_boxes.append({
            "type": "text",
            "text": f"• {h}",
            "size": "xs",
            "color": "#424242",
            "wrap": True,
            "margin": "xs"
        })

    return {
        "type": "bubble",
        "size": "mega",
        "header": {
            "type": "box",
            "layout": "vertical",
            "backgroundColor": "#4A148C",
            "paddingAll": "15px",
            "contents": [
                {
                    "type": "box",
                    "layout": "horizontal",
                    "contents": [
                        {"type": "text", "text": "📑 AI 每日市場快訊", "color": "#FFFFFF", "weight": "bold", "size": "md", "flex": 3},
                        {
                            "type": "text",
                            "text": report.market_sentiment,
                            "size": "xs",
                            "color": "#FFFFFF",
                            "weight": "bold",
                            "align": "end",
                            "flex": 2
                        }
                    ]
                },
                {
                    "type": "text",
                    "text": report.title,
                    "color": "#E1BEE7",
                    "size": "sm",
                    "weight": "bold",
                    "wrap": True,
                    "margin": "xs"
                },
                {
                    "type": "text",
                    "text": f"生成時間：{report.date_str}",
                    "color": "#BA68C8",
                    "size": "xxs",
                    "margin": "xs"
                }
            ]
        },
        "body": {
            "type": "box",
            "layout": "vertical",
            "spacing": "sm",
            "contents": [
                # 大盤指數區塊
                {
                    "type": "box",
                    "layout": "vertical",
                    "contents": grid_rows
                },
                {
                    "type": "text",
                    "text": report.index_summary,
                    "size": "xs",
                    "color": "#555555",
                    "wrap": True,
                    "margin": "xs"
                },
                {"type": "separator", "margin": "sm"},
                # 重大新聞焦點
                {
                    "type": "box",
                    "layout": "vertical",
                    "margin": "sm",
                    "contents": [
                        {"type": "text", "text": "🔥 今日市場關鍵焦點:", "size": "xs", "weight": "bold", "color": "#311B92"}
                    ] + headline_boxes
                },
                {"type": "separator", "margin": "sm"},
                # 現有持股影響評估
                {
                    "type": "box",
                    "layout": "vertical",
                    "margin": "sm",
                    "backgroundColor": "#F3E5F5",
                    "paddingAll": "8px",
                    "cornerRadius": "6px",
                    "contents": [
                        {"type": "text", "text": "🎯 對您目前持股的影響評估:", "size": "xs", "weight": "bold", "color": "#4A148C"},
                        {"type": "text", "text": report.portfolio_insight, "size": "xs", "color": "#424242", "wrap": True, "margin": "xs"}
                    ]
                },
                # 短線策略叮嚀
                {
                    "type": "box",
                    "layout": "vertical",
                    "margin": "xs",
                    "backgroundColor": "#E8F5E9",
                    "paddingAll": "8px",
                    "cornerRadius": "6px",
                    "contents": [
                        {"type": "text", "text": "💡 今日操作與風控叮嚀:", "size": "xs", "weight": "bold", "color": "#2E7D32"},
                        {"type": "text", "text": report.actionable_takeaway, "size": "xs", "color": "#2E7D32", "wrap": True, "margin": "xs"}
                    ]
                }
            ]
        },
        "footer": {
            "type": "box",
            "layout": "horizontal",
            "spacing": "sm",
            "contents": [
                {
                    "type": "button",
                    "style": "secondary",
                    "height": "sm",
                    "action": {
                        "type": "message",
                        "label": "📊 查詢庫存",
                        "text": "查詢庫存"
                    }
                },
                {
                    "type": "button",
                    "style": "primary",
                    "color": "#4A148C",
                    "height": "sm",
                    "action": {
                        "type": "message",
                        "label": "💡 買入建議",
                        "text": "買入建議"
                    }
                }
            ]
        }
    }


