# 幻覺驗證層模型檔放置位置

此目錄存放 `app/services/grounding_verifier.py` 所需的 NLI 模型。
**權重檔已由 `.gitignore` 排除**（110~340MB 二進位檔不進 git），需依下列步驟自行產生。

## 需要的檔案

| 檔案 | 預設路徑設定 |
| :--- | :--- |
| `model_quantized.onnx` | `GROUNDING_MODEL_PATH` |
| `tokenizer.json` | `GROUNDING_TOKENIZER_PATH` |

## 產生方式

在 Colab 開啟 [`slm/colab_nli_export.ipynb`](../../slm/colab_nli_export.ipynb)，依序執行後下載三個產出物：

1. `model_quantized.onnx` → 放入本目錄
2. `tokenizer.json` → 放入本目錄
3. `label_order.txt` → 內容填入 `.env` 的 `GROUNDING_LABEL_ORDER`

> **標籤順序填錯會讓 entailment 與 contradiction 對調**，使驗證結果完全相反。
> `grounding_verifier` 會在載入時檢查該值是否包含 `entailment`，但無法察覺順序被調換，
> 因此請務必以筆記本印出的值為準，不要憑印象填。

## 沒有模型檔時的行為

驗證層為 **fail-open**：找不到模型檔時只會記錄一行 log 並停用驗證，
買入掃描照常運作、不會拋出例外。因此本目錄為空時系統仍完全可用，
只是少了幻覺驗證這道防線。

## 部署注意

* `.dockerignore` **未**排除 `models/`，故本目錄內容會進入映像層。
* 映像體積因此增加約 110MB（MiniLMv2-L6 int8）或 338MB（mDeBERTa 原廠量化版）。
* 模型為 lazy load，且只在買入掃描的背景路徑觸發，不影響 LINE Webhook 的回應時限。
