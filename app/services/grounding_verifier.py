"""SLM 幻覺驗證層：以 NLI entailment 檢查 LLM 輸出是否真被來源文本支持。

背景
----
`llm_service` 的提示詞第 3 條寫著「嚴格禁止使用預訓練知識庫或臆測」，第 5 條要求
`reasons` 必須列出「新聞中具體提及的事實」。但這些限制**原本只是提示詞裡的宣告**，
系統從未驗證 Gemini 是否真的遵守。本模組把該宣告變成可量測的檢查。

做法
----
將新聞文本視為前提 (premise)、Gemini 給出的每條理由視為假設 (hypothesis)，
以小型多語 NLI 模型判斷前提是否蘊含 (entail) 該假設。

* 新聞文本長度遠超模型上限，故依 `news_service` 既有的「【新聞 N】」區塊切塊，
  對每條理由取各塊 entailment 機率的最大值——一條理由只要被任一則新聞支持即算接地。
* 推論在容器內 CPU 以 onnxruntime 執行，無外部依賴、無隧道。
  模型權重由 `slm/colab_nli_export.ipynb` 在 Colab 量化匯出（Colab 僅用於訓練／匯出）。

風險取向
--------
此層**預設為 shadow**，且任何載入或推論失敗一律 fail-open（視為未驗證而非未通過）。
理由與新聞預篩相同的非對稱性：NLI 模型誤判一條合法理由，會讓一個真實買入機會被
擋掉；而漏放一條幻覺理由，最多是維持現狀。啟用 enforce 前必須先以
`scripts/eval_grounding.py` 量測攔截率與誤殺率。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

from app.core.config import settings
from app.core.logging import logger

# news_service 組裝文本時的新聞區塊標記
NEWS_BLOCK_PATTERN = re.compile(r"(?=【新聞\s*\d+】)")
NO_NEWS_MARKER = "查無任何相關財經新聞"


@dataclass
class ReasonCheck:
    """單一理由的驗證結果。"""

    reason: str
    entailment: float
    grounded: bool
    best_chunk_index: int = -1

    @property
    def badge(self) -> str:
        return "✅ 有據" if self.grounded else "⚠️ 未能佐證"


@dataclass
class GroundingReport:
    """一次驗證的彙總結果。"""

    checks: List[ReasonCheck] = field(default_factory=list)
    verified: bool = False          # 模型是否真的跑了（False = 未驗證，非未通過）
    skip_reason: str = ""

    @property
    def grounded_count(self) -> int:
        return sum(1 for c in self.checks if c.grounded)

    @property
    def grounded_ratio(self) -> float:
        return self.grounded_count / len(self.checks) if self.checks else 0.0

    @property
    def ungrounded(self) -> List[ReasonCheck]:
        return [c for c in self.checks if not c.grounded]

    def passes(self, min_ratio: float) -> bool:
        """未驗證時一律視為通過（fail-open）。"""
        if not self.verified or not self.checks:
            return True
        return self.grounded_ratio >= min_ratio

    @property
    def summary(self) -> str:
        if not self.verified:
            return f"未驗證（{self.skip_reason}）"
        if not self.checks:
            return "無理由可驗證"
        return (
            f"接地 {self.grounded_count}/{len(self.checks)}"
            f"（{self.grounded_ratio:.0%}）"
            + (f"，未佐證：{'；'.join(c.reason[:28] for c in self.ungrounded)}" if self.ungrounded else "")
        )


def _softmax(values: Sequence[float]) -> List[float]:
    m = max(values)
    exps = [math.exp(v - m) for v in values]
    total = sum(exps) or 1.0
    return [e / total for e in exps]


class GroundingVerifier:
    """以 ONNX NLI 模型驗證理由是否被來源文本支持。

    模型在第一次使用時才載入（lazy load），且只會在買入掃描的背景路徑被觸發，
    不掛在 LINE Webhook 的啟動路徑上，以免影響冷啟動與回覆時限。
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        label_order: Optional[str] = None,
        max_seq_length: Optional[int] = None,
    ):
        self.model_path = model_path or settings.GROUNDING_MODEL_PATH
        self.tokenizer_path = tokenizer_path or settings.GROUNDING_TOKENIZER_PATH
        self.label_order = [
            s.strip().lower()
            for s in (label_order or settings.GROUNDING_LABEL_ORDER).split(",")
            if s.strip()
        ]
        self.max_seq_length = max_seq_length or settings.GROUNDING_MAX_SEQ_LENGTH

        self._session = None
        self._tokenizer = None
        self._load_error: Optional[str] = None
        self._loaded = False

    # --- 模型載入 ---------------------------------------------------------

    def _ensure_loaded(self) -> bool:
        """載入模型與 tokenizer；失敗時記錄原因並回傳 False（不拋出）。"""
        if self._loaded:
            return self._session is not None
        self._loaded = True

        if "entailment" not in self.label_order:
            self._load_error = f"GROUNDING_LABEL_ORDER 未包含 entailment：{self.label_order}"
            logger.warning(f"[Grounding] {self._load_error}")
            return False

        model_file, tok_file = Path(self.model_path), Path(self.tokenizer_path)
        if not model_file.exists() or not tok_file.exists():
            self._load_error = f"模型檔不存在（{self.model_path} / {self.tokenizer_path}）"
            logger.info(f"[Grounding] {self._load_error}，驗證層停用")
            return False

        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as e:
            self._load_error = f"缺少推論套件：{e}"
            logger.warning(f"[Grounding] {self._load_error}")
            return False

        try:
            tokenizer = Tokenizer.from_file(str(tok_file))
            tokenizer.enable_truncation(max_length=self.max_seq_length)
            opts = ort.SessionOptions()
            # Cloud Run 的 vCPU 有限，放任 ORT 開滿執行緒反而與事件迴圈搶資源
            opts.intra_op_num_threads = 1
            session = ort.InferenceSession(
                str(model_file), sess_options=opts, providers=["CPUExecutionProvider"]
            )
        except Exception as e:
            self._load_error = f"載入失敗：{type(e).__name__}: {e}"
            logger.warning(f"[Grounding] {self._load_error}")
            return False

        self._tokenizer, self._session = tokenizer, session
        logger.info(f"[Grounding] NLI 模型已載入：{self.model_path}")
        return True

    # --- 推論 -------------------------------------------------------------

    @staticmethod
    def split_premise(news_text: str) -> List[str]:
        """依【新聞 N】標記切塊；無標記時整段視為一塊。"""
        text = (news_text or "").strip()
        if not text:
            return []
        parts = [p.strip() for p in NEWS_BLOCK_PATTERN.split(text) if p.strip()]
        # 第一段通常是「### 標的: ...」標題，僅在它是唯一內容時才保留
        blocks = [p for p in parts if p.startswith("【新聞")]
        return blocks or parts

    def _entailment_prob(self, premise: str, hypothesis: str) -> Optional[float]:
        """回傳 premise 蘊含 hypothesis 的機率；推論失敗回傳 None。"""
        try:
            encoding = self._tokenizer.encode(premise, hypothesis)
            feeds = {}
            for inp in self._session.get_inputs():
                if inp.name == "input_ids":
                    feeds[inp.name] = [encoding.ids]
                elif inp.name == "attention_mask":
                    feeds[inp.name] = [encoding.attention_mask]
                elif inp.name == "token_type_ids":
                    feeds[inp.name] = [encoding.type_ids]
            logits = self._session.run(None, feeds)[0][0]
            probs = _softmax([float(x) for x in logits])
            if len(probs) != len(self.label_order):
                logger.warning(
                    f"[Grounding] 模型輸出維度 {len(probs)} 與標籤數 {len(self.label_order)} 不符"
                )
                return None
            return probs[self.label_order.index("entailment")]
        except Exception as e:
            logger.warning(f"[Grounding] 推論失敗：{type(e).__name__}: {e}")
            return None

    def verify(self, news_text: str, reasons: Sequence[str]) -> GroundingReport:
        """逐條驗證理由是否被新聞文本支持。"""
        cleaned = [r.strip() for r in (reasons or []) if r and r.strip()]
        if not cleaned:
            return GroundingReport(verified=False, skip_reason="無理由可驗證")

        if NO_NEWS_MARKER in (news_text or ""):
            # 查無新聞時任何理由都不可能有據，但這屬於 llm_service 防幻覺規則 4
            # 的職責範圍，此處不重複判定，僅標記為未驗證。
            return GroundingReport(verified=False, skip_reason="查無新聞，無前提可比對")

        chunks = self.split_premise(news_text)
        if not chunks:
            return GroundingReport(verified=False, skip_reason="前提文本為空")

        if not self._ensure_loaded():
            return GroundingReport(verified=False, skip_reason=self._load_error or "模型未就緒")

        threshold = settings.GROUNDING_ENTAILMENT_THRESHOLD
        checks: List[ReasonCheck] = []
        for reason in cleaned:
            best_prob, best_idx = 0.0, -1
            for idx, chunk in enumerate(chunks):
                prob = self._entailment_prob(chunk, reason)
                if prob is None:
                    continue
                if prob > best_prob:
                    best_prob, best_idx = prob, idx
            if best_idx < 0:
                # 這條理由的所有推論都失敗 -> 整份報告視為未驗證，避免半套結果誤導
                return GroundingReport(verified=False, skip_reason="推論全數失敗")
            checks.append(
                ReasonCheck(
                    reason=reason,
                    entailment=best_prob,
                    grounded=best_prob >= threshold,
                    best_chunk_index=best_idx,
                )
            )

        return GroundingReport(checks=checks, verified=True)
