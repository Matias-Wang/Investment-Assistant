"""SLM 幻覺驗證層測試。

不下載任何模型權重：以 stub session 與 stub tokenizer 注入固定 logits，
專注驗證我們自己的邏輯（切塊、取最大值、門檻、fail-open、四種模式）。
模型本身的品質由 `scripts/eval_grounding.py` 以對抗測試集量測，屬另一件事。
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.config import settings
from app.models.domain import SystemConfig
from app.models.llm import BuyRecommendation
from app.services.buy_engine import BuyStrategyEngine
from app.services.grounding_verifier import (
    GroundingReport,
    GroundingVerifier,
    ReasonCheck,
)

NEWS = (
    "### 標的: 2330 台積電 近 3 日相關財經新聞報導：\n\n"
    "【新聞 1】標題：台積電 11 月營收創同期新高\n"
    "摘要內容：營收年增 34%，主要受惠 AI 訂單。\n\n"
    "【新聞 2】標題：外資調升目標價\n"
    "摘要內容：法人看好先進製程動能。\n"
)


class StubTokenizer:
    """模擬 tokenizers.Tokenizer 的 pair encoding。"""

    def __init__(self):
        self.calls = []

    def enable_truncation(self, max_length):
        self.max_length = max_length

    def encode(self, premise, hypothesis):
        self.calls.append((premise, hypothesis))
        enc = MagicMock()
        enc.ids = [1, 2, 3]
        enc.attention_mask = [1, 1, 1]
        enc.type_ids = [0, 0, 1]
        return enc


class StubSession:
    """模擬 onnxruntime.InferenceSession；logits 由 callable 決定。"""

    def __init__(self, logits_fn, input_names=("input_ids", "attention_mask", "token_type_ids")):
        self._logits_fn = logits_fn
        self._inputs = [MagicMock(name=n) for n in input_names]
        for m, n in zip(self._inputs, input_names):
            m.name = n
        self.run_count = 0

    def get_inputs(self):
        return self._inputs

    def run(self, _outputs, feeds):
        self.run_count += 1
        return [[self._logits_fn(self.run_count)]]


def _verifier(logits_fn) -> GroundingVerifier:
    """建立已注入 stub 的 verifier，跳過真實載入。"""
    v = GroundingVerifier(label_order="entailment,neutral,contradiction")
    v._tokenizer = StubTokenizer()
    v._session = StubSession(logits_fn)
    v._loaded = True
    return v


# ---------------------------------------------------------------- 前提切塊

def test_split_premise_by_news_blocks():
    chunks = GroundingVerifier.split_premise(NEWS)
    assert len(chunks) == 2
    assert chunks[0].startswith("【新聞 1】")
    assert chunks[1].startswith("【新聞 2】")
    # 標題行不應成為獨立前提
    assert not any(c.startswith("### 標的") for c in chunks)


def test_split_premise_without_markers():
    chunks = GroundingVerifier.split_premise("一段沒有新聞標記的純文字")
    assert chunks == ["一段沒有新聞標記的純文字"]


def test_split_premise_empty():
    assert GroundingVerifier.split_premise("") == []
    assert GroundingVerifier.split_premise("   ") == []


# ---------------------------------------------------------------- 核心判定

def test_grounded_reason_passes():
    # entailment 分數最高
    v = _verifier(lambda n: [5.0, 0.0, 0.0])
    report = v.verify(NEWS, ["營收年增 34%"])
    assert report.verified
    assert len(report.checks) == 1
    assert report.checks[0].grounded
    assert report.checks[0].entailment > 0.9
    assert report.grounded_ratio == 1.0


def test_ungrounded_reason_flagged():
    # contradiction 分數最高 -> entailment 機率低
    v = _verifier(lambda n: [0.0, 0.0, 5.0])
    report = v.verify(NEWS, ["公司將發放特別股利每股 10 元"])
    assert report.verified
    assert not report.checks[0].grounded
    assert report.ungrounded
    assert report.grounded_ratio == 0.0


def test_takes_max_entailment_across_chunks():
    """一條理由只要被任一則新聞支持就算接地，故取各塊最大值。"""
    # 第 1 次（新聞1）低分、第 2 次（新聞2）高分
    v = _verifier(lambda n: [5.0, 0.0, 0.0] if n == 2 else [0.0, 0.0, 5.0])
    report = v.verify(NEWS, ["法人調升目標價"])
    assert report.checks[0].grounded
    assert report.checks[0].best_chunk_index == 1


def test_mixed_reasons_ratio():
    # 共 2 條理由 x 2 塊 = 4 次推論；讓前兩次高分、後兩次低分
    v = _verifier(lambda n: [5.0, 0.0, 0.0] if n <= 2 else [0.0, 0.0, 5.0])
    report = v.verify(NEWS, ["營收年增 34%", "取得獨家專利授權"])
    assert report.grounded_count == 1
    assert report.grounded_ratio == 0.5


def test_threshold_is_respected(monkeypatch):
    # logits 使 entailment 機率約 0.5 附近
    v = _verifier(lambda n: [1.0, 0.9, 0.9])
    monkeypatch.setattr(settings, "GROUNDING_ENTAILMENT_THRESHOLD", 0.99)
    assert not v.verify(NEWS, ["某理由"]).checks[0].grounded
    monkeypatch.setattr(settings, "GROUNDING_ENTAILMENT_THRESHOLD", 0.01)
    v2 = _verifier(lambda n: [1.0, 0.9, 0.9])
    assert v2.verify(NEWS, ["某理由"]).checks[0].grounded


# ---------------------------------------------------------------- fail-open

def test_missing_model_is_unverified_not_failed():
    """模型不存在時必須是「未驗證」，不可等同於「未通過」。"""
    v = GroundingVerifier(model_path="does/not/exist.onnx", tokenizer_path="nope.json")
    report = v.verify(NEWS, ["營收年增"])
    assert report.verified is False
    assert report.passes(min_ratio=1.0) is True  # fail-open
    assert "不存在" in report.skip_reason


def test_inference_failure_is_unverified():
    def boom(n):
        raise RuntimeError("onnx boom")

    v = _verifier(boom)
    report = v.verify(NEWS, ["營收年增"])
    assert report.verified is False
    assert report.passes(min_ratio=1.0) is True


def test_no_reasons_is_unverified():
    v = _verifier(lambda n: [5.0, 0.0, 0.0])
    report = v.verify(NEWS, [])
    assert report.verified is False
    assert report.passes(min_ratio=1.0) is True


def test_no_news_marker_skips_verification():
    v = _verifier(lambda n: [5.0, 0.0, 0.0])
    report = v.verify("【近 3 日查無任何相關財經新聞或市場報導】", ["營收年增"])
    assert report.verified is False
    assert "查無新聞" in report.skip_reason


def test_bad_label_order_disables_layer():
    """標籤順序設錯會讓 entailment 與 contradiction 對調，必須擋在載入階段。"""
    v = GroundingVerifier(label_order="foo,bar,baz")
    assert v._ensure_loaded() is False
    assert "entailment" in (v._load_error or "")


def test_label_order_mismatch_with_output_dim():
    v = GroundingVerifier(label_order="entailment,contradiction")  # 只有 2 個標籤
    v._tokenizer = StubTokenizer()
    v._session = StubSession(lambda n: [1.0, 2.0, 3.0])  # 模型輸出 3 維
    v._loaded = True
    report = v.verify(NEWS, ["某理由"])
    assert report.verified is False


# ---------------------------------------------------------------- 報告呈現

def test_report_summary_lists_ungrounded():
    report = GroundingReport(
        checks=[
            ReasonCheck("營收年增 34%", 0.95, True),
            ReasonCheck("將併購同業", 0.10, False),
        ],
        verified=True,
    )
    assert "接地 1/2" in report.summary
    assert "將併購同業" in report.summary


def test_badge_text():
    assert "有據" in ReasonCheck("x", 0.9, True).badge
    assert "未能佐證" in ReasonCheck("x", 0.1, False).badge


# ---------------------------------------------------------------- 買入引擎整合

def _rec(decision="BUY", reasons=None) -> BuyRecommendation:
    return BuyRecommendation(
        ticker="2330", company_name="台積電", decision=decision,
        confidence_score=0.9, target_shares=10, estimated_amount=10000.0,
        sentiment_summary="摘要", reasons=reasons or ["營收年增 34%", "取得大單"],
        risk_factors=["匯率"],
    )


def _engine(verifier):
    config_repo, portfolio_repo = MagicMock(), MagicMock()
    market_service, news_service, llm_service = MagicMock(), MagicMock(), MagicMock()
    config_repo.get_config = AsyncMock(return_value=SystemConfig(trade_budget_per_order=100000.0))
    portfolio_repo.get_all = AsyncMock(return_value=[])
    portfolio_repo.get_by_ticker = AsyncMock(return_value=None)
    market_service.get_price_and_sma20 = MagicMock(return_value=(100.0, 90.0))
    news_service.get_recent_news_text = AsyncMock(return_value=NEWS)
    llm_service.analyze_buy_candidate = MagicMock(return_value=_rec())
    return BuyStrategyEngine(
        config_repo, portfolio_repo, market_service, news_service, llm_service,
        grounding_verifier=verifier,
    ), llm_service


@pytest.mark.asyncio
async def test_shadow_mode_does_not_block(monkeypatch):
    monkeypatch.setattr(settings, "GROUNDING_MODE", "shadow")
    monkeypatch.setattr(settings, "NEWS_FILTER_MODE", "off")
    verifier = MagicMock()
    verifier.verify = MagicMock(
        return_value=GroundingReport(checks=[ReasonCheck("x", 0.1, False)], verified=True)
    )
    engine, _ = _engine(verifier)
    results = await engine.scan_candidates(candidate_tickers=["2330"])
    assert len(results) == 1  # 全未接地也照樣推播


@pytest.mark.asyncio
async def test_enforce_mode_blocks_ungrounded(monkeypatch):
    monkeypatch.setattr(settings, "GROUNDING_MODE", "enforce")
    monkeypatch.setattr(settings, "NEWS_FILTER_MODE", "off")
    monkeypatch.setattr(settings, "GROUNDING_MIN_GROUNDED_RATIO", 0.5)
    verifier = MagicMock()
    verifier.verify = MagicMock(
        return_value=GroundingReport(
            checks=[ReasonCheck("a", 0.1, False), ReasonCheck("b", 0.1, False)], verified=True
        )
    )
    engine, _ = _engine(verifier)
    assert await engine.scan_candidates(candidate_tickers=["2330"]) == []


@pytest.mark.asyncio
async def test_enforce_mode_allows_grounded(monkeypatch):
    monkeypatch.setattr(settings, "GROUNDING_MODE", "enforce")
    monkeypatch.setattr(settings, "NEWS_FILTER_MODE", "off")
    monkeypatch.setattr(settings, "GROUNDING_MIN_GROUNDED_RATIO", 0.5)
    verifier = MagicMock()
    verifier.verify = MagicMock(
        return_value=GroundingReport(
            checks=[ReasonCheck("a", 0.9, True), ReasonCheck("b", 0.1, False)], verified=True
        )
    )
    engine, _ = _engine(verifier)
    assert len(await engine.scan_candidates(candidate_tickers=["2330"])) == 1


@pytest.mark.asyncio
async def test_enforce_fails_open_when_unverified(monkeypatch):
    """模型未就緒時 enforce 也不得攔阻，否則模型缺檔會讓系統再也不推播任何建議。"""
    monkeypatch.setattr(settings, "GROUNDING_MODE", "enforce")
    monkeypatch.setattr(settings, "NEWS_FILTER_MODE", "off")
    verifier = MagicMock()
    verifier.verify = MagicMock(
        return_value=GroundingReport(verified=False, skip_reason="模型檔不存在")
    )
    engine, _ = _engine(verifier)
    assert len(await engine.scan_candidates(candidate_tickers=["2330"])) == 1


@pytest.mark.asyncio
async def test_verifier_exception_fails_open(monkeypatch):
    monkeypatch.setattr(settings, "GROUNDING_MODE", "enforce")
    monkeypatch.setattr(settings, "NEWS_FILTER_MODE", "off")
    verifier = MagicMock()
    verifier.verify = MagicMock(side_effect=RuntimeError("boom"))
    engine, _ = _engine(verifier)
    assert len(await engine.scan_candidates(candidate_tickers=["2330"])) == 1


@pytest.mark.asyncio
async def test_annotate_mode_rewrites_reasons(monkeypatch):
    monkeypatch.setattr(settings, "GROUNDING_MODE", "annotate")
    monkeypatch.setattr(settings, "NEWS_FILTER_MODE", "off")
    verifier = MagicMock()
    verifier.verify = MagicMock(
        return_value=GroundingReport(
            checks=[
                ReasonCheck("營收年增 34%", 0.95, True),
                ReasonCheck("將併購同業", 0.05, False),
            ],
            verified=True,
        )
    )
    engine, _ = _engine(verifier)
    results = await engine.scan_candidates(candidate_tickers=["2330"])
    rec, _cand = results[0]
    assert any("有據" in r for r in rec.reasons)
    assert any("未能佐證" in r for r in rec.reasons)


@pytest.mark.asyncio
async def test_off_mode_skips_verifier(monkeypatch):
    monkeypatch.setattr(settings, "GROUNDING_MODE", "off")
    monkeypatch.setattr(settings, "NEWS_FILTER_MODE", "off")
    verifier = MagicMock()
    engine, _ = _engine(verifier)
    await engine.scan_candidates(candidate_tickers=["2330"])
    verifier.verify.assert_not_called()


@pytest.mark.asyncio
async def test_non_buy_decision_skips_verification(monkeypatch):
    """HOLD/AVOID 不會推播，驗證它們沒有意義且浪費 CPU。"""
    monkeypatch.setattr(settings, "GROUNDING_MODE", "enforce")
    monkeypatch.setattr(settings, "NEWS_FILTER_MODE", "off")
    verifier = MagicMock()
    engine, llm_service = _engine(verifier)
    llm_service.analyze_buy_candidate = MagicMock(return_value=_rec(decision="HOLD"))
    await engine.scan_candidates(candidate_tickers=["2330"])
    verifier.verify.assert_not_called()
