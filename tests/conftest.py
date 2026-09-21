"""測試全域預設值。

`models/nli/` 內若存在真實模型檔（338MB），`BuyStrategyEngine` 預設建立的
`GroundingVerifier` 會在 `GROUNDING_MODE=shadow`（正式環境預設值）下真的載入它，
使整套測試從約 5 秒變成 40 秒，並讓並行度測試的計時斷言失準。

因此測試環境一律預設 `off`；需要驗證該層行為的測試（`test_grounding_verifier.py`）
會自行 monkeypatch 為 shadow / annotate / enforce，並注入 stub session，
不依賴磁碟上的模型檔。
"""

import pytest

from app.core.config import settings


@pytest.fixture(autouse=True)
def disable_grounding_by_default(monkeypatch):
    monkeypatch.setattr(settings, "GROUNDING_MODE", "off")
