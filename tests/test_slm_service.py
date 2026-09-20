import pytest
from unittest.mock import patch, MagicMock
import httpx
from app.models.slm import NLUIntent, NLUIntentResult
from app.services.slm_service import SLMService


@pytest.mark.asyncio
async def test_slm_service_parse_buy_intent():
    service = SLMService(
        endpoint_url="http://mock-slm:8000/v1",
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        api_key="mock_key",
    )

    mock_response = {
        "choices": [
            {
                "message": {
                    "content": '```json\n{\n  "intent": "RECORD_BUY",\n  "confidence": 0.98,\n  "ticker": "台積電",\n  "price": 1050.0,\n  "shares": 8.0,\n  "explanation": "買進台積電8股"\n}\n```'
                }
            }
        ]
    }

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_response
        mock_post.return_value = mock_resp

        res = await service.parse_intent("剛剛在 1050 買進 8 股台積電")

        assert res.intent == NLUIntent.RECORD_BUY
        assert res.ticker == "2330"  # 自動解析中文名稱至代號
        assert res.price == 1050.0
        assert res.shares == 8.0
        assert res.confidence == 0.98


@pytest.mark.asyncio
async def test_slm_service_parse_update_config():
    service = SLMService(
        endpoint_url="http://mock-slm:8000/v1",
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
    )

    mock_response = {
        "choices": [
            {
                "message": {
                    "content": '{"intent": "UPDATE_CONFIG", "confidence": 0.95, "config_field": "budget", "config_value": 150000}'
                }
            }
        ]
    }

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_response
        mock_post.return_value = mock_resp

        res = await service.parse_intent("把單筆預算改為15萬")

        assert res.intent == NLUIntent.UPDATE_CONFIG
        assert res.config_field == "budget"
        assert res.config_value == 150000.0


@pytest.mark.asyncio
async def test_slm_service_network_failure_fallback():
    service = SLMService(
        endpoint_url="http://unreachable-host:8000/v1",
    )

    with patch("httpx.AsyncClient.post", side_effect=httpx.ConnectError("Connection refused")):
        res = await service.parse_intent("今天大盤如何")

        assert res.intent == NLUIntent.UNKNOWN
        assert res.raw_text == "今天大盤如何"


@pytest.mark.asyncio
async def test_slm_service_malformed_json_fallback():
    service = SLMService(endpoint_url="http://mock-slm:8000/v1")

    mock_response = {
        "choices": [
            {
                "message": {
                    "content": "This is not valid json content from model"
                }
            }
        ]
    }

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_response
        mock_post.return_value = mock_resp

        res = await service.parse_intent("隨意說說")
        assert res.intent == NLUIntent.UNKNOWN
