import pytest
from unittest.mock import MagicMock, patch
from linebot.v3.messaging import TextMessage
from app.services.line_bot_service import LineBotService


def test_line_bot_service_context_manager():
    config_repo = MagicMock()
    portfolio_repo = MagicMock()
    session_repo = MagicMock()

    service = LineBotService(config_repo, portfolio_repo, session_repo)
    service.channel_token = "test_token"

    with patch("app.services.line_bot_service.ApiClient") as mock_api_client_cls, \
         patch("app.services.line_bot_service.MessagingApi") as mock_messaging_api_cls:
        
        mock_client_instance = MagicMock()
        mock_api_client_cls.return_value = mock_client_instance
        mock_client_instance.__enter__.return_value = mock_client_instance

        mock_api_instance = MagicMock()
        mock_messaging_api_cls.return_value = mock_api_instance

        # Test reply_message
        service.reply_message("dummy_token", [TextMessage(text="test")])
        mock_messaging_api_cls.assert_called_with(mock_client_instance)
        assert mock_api_instance.reply_message.called

        # Test push_message
        service.push_message("dummy_user", [TextMessage(text="test")])
        assert mock_api_instance.push_message.called
