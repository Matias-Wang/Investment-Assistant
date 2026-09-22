"""模型載入的併發安全與提早喚醒。

買入掃描以 `asyncio.to_thread` 呼叫 `verify`，併發度預設 3，所以載入必然是
多執行緒競爭的。修正前 `_ensure_loaded` 會在真正載入**之前**就把 `_loaded`
設為 True，導致第二、三個執行緒看到「已載入」但 session 仍是 None，於是
fail-open 跳過驗證——生產環境的實際症狀是冷啟動後前幾檔標的都是
「未驗證（模型未就緒）」。
"""

import threading
import time
from unittest.mock import MagicMock, patch

from app.services.grounding_verifier import GroundingVerifier


def _slow_loading_verifier(load_seconds=0.3):
    """_load 會慢慢成功的 verifier，用來暴露載入期間的競爭。"""
    v = GroundingVerifier(label_order="entailment,neutral,contradiction")
    calls = []

    def fake_load():
        calls.append(1)
        time.sleep(load_seconds)
        v._tokenizer = MagicMock()
        v._session = MagicMock()
        return True

    return v, calls, fake_load


def test_concurrent_load_does_not_report_model_unready():
    """載入期間湧入的執行緒必須等待，不可被判為模型不可用。"""
    v, calls, fake_load = _slow_loading_verifier()
    results = []

    with patch.object(v, "_load", fake_load):
        threads = [
            threading.Thread(target=lambda: results.append(v._ensure_loaded()))
            for _ in range(3)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert results == [True, True, True], f"有執行緒在載入期間被誤判：{results}"
    assert len(calls) == 1, f"模型被重複載入 {len(calls)} 次"


def test_loaded_flag_not_set_before_load_finishes():
    """_loaded 不得在載入完成前就為 True，否則等於對其他執行緒說謊。"""
    v = GroundingVerifier(label_order="entailment,neutral,contradiction")
    seen_mid_load = {}

    def fake_load():
        seen_mid_load["loaded"] = v._loaded
        v._tokenizer = MagicMock()
        v._session = MagicMock()
        return True

    with patch.object(v, "_load", fake_load):
        assert v._ensure_loaded() is True

    assert seen_mid_load["loaded"] is False
    assert v._loaded is True


def test_failed_load_is_cached_and_not_retried():
    """載入失敗只嘗試一次，不要每檔標的都重試一遍 338MB 的載入。"""
    v = GroundingVerifier(label_order="entailment,neutral,contradiction")
    calls = []

    def fake_load():
        calls.append(1)
        return False

    with patch.object(v, "_load", fake_load):
        assert v._ensure_loaded() is False
        assert v._ensure_loaded() is False
        assert v._ensure_loaded() is False

    assert len(calls) == 1


def test_warmup_loads_model_ahead_of_verify():
    """warmup 先載入，之後的 verify 就不必再等。"""
    v, calls, fake_load = _slow_loading_verifier(load_seconds=0)

    with patch.object(v, "_load", fake_load):
        assert v.warmup() is True
        assert len(calls) == 1
        # 已就緒，後續呼叫不再載入
        assert v._ensure_loaded() is True
        assert len(calls) == 1


def test_warmup_on_missing_model_is_harmless():
    """模型檔不存在時 warmup 安靜回 False，不得拋出而中斷啟動流程。"""
    v = GroundingVerifier(
        model_path="does/not/exist.onnx",
        tokenizer_path="nope.json",
        label_order="entailment,neutral,contradiction",
    )
    assert v.warmup() is False
