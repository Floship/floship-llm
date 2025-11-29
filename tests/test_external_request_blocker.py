"""Tests to verify the external request blocker works correctly."""

import socket
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import ExternalRequestBlockedError


class TestExternalRequestBlocker:
    """Tests for the external HTTP request blocker."""

    def test_external_request_is_blocked(self) -> None:
        """Verify that external HTTP requests are blocked with a clear error."""
        # Try to create a socket connection to an external host
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with pytest.raises(ExternalRequestBlockedError) as exc_info:
                sock.connect(("api.openai.com", 443))

            error_message = str(exc_info.value)
            assert "BLOCKED" in error_message
            assert "external HTTP request" in error_message
            assert "api.openai.com" in error_message
            assert "must be mocked" in error_message
        finally:
            sock.close()

    def test_localhost_is_allowed(self) -> None:
        """Verify that localhost connections are still allowed."""
        # This should not raise - localhost is allowed
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            # We can't actually connect (nothing listening), but the blocker
            # should not raise ExternalRequestBlockedError
            with pytest.raises(OSError):  # Connection refused is expected
                sock.connect(("127.0.0.1", 59999))  # Use unlikely port
        except ExternalRequestBlockedError:
            pytest.fail("Localhost connections should not be blocked")
        finally:
            sock.close()

    def test_mocked_httpx_passes(self) -> None:
        """Verify that properly mocked httpx requests work."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "success"}

        with patch("httpx.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value.__enter__.return_value = mock_instance
            mock_instance.post.return_value = mock_response

            # This should work fine - it's mocked
            import httpx

            with httpx.Client() as client:
                response = client.post("https://api.example.com/test")
                assert response.status_code == 200
                assert response.json() == {"result": "success"}

    def test_error_message_suggests_solutions(self) -> None:
        """Verify the error message provides helpful guidance."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with pytest.raises(ExternalRequestBlockedError) as exc_info:
                sock.connect(("example.com", 80))

            error_message = str(exc_info.value)
            # Should suggest httpx mocking
            assert "httpx.Client" in error_message
            assert "patch" in error_message
            # Should suggest requests mocking
            assert "requests.post" in error_message
        finally:
            sock.close()


class TestMockHttpxClientFixture:
    """Tests for the mock_httpx_client fixture."""

    def test_fixture_provides_working_mock(self, mock_httpx_client: MagicMock) -> None:
        """Verify the mock_httpx_client fixture works correctly."""
        # The fixture should provide a pre-configured mock
        response = mock_httpx_client.post("https://api.example.com/test")
        assert response.status_code == 200
        assert "choices" in response.json()

    def test_fixture_mock_can_be_customized(self, mock_httpx_client: MagicMock) -> None:
        """Verify the mock can be customized for specific tests."""
        # Customize the mock for this specific test
        mock_httpx_client.post.return_value = MagicMock(
            status_code=400,
            json=lambda: {"error": "Bad request"},
        )

        response = mock_httpx_client.post("https://api.example.com/test")
        assert response.status_code == 400
        assert response.json() == {"error": "Bad request"}
