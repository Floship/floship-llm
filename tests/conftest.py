"""
Pytest configuration and fixtures for floship-llm tests.

This module ensures that no tests make actual external HTTP requests.
All HTTP requests must be mocked.
"""

import logging
import socket
from typing import Any
from unittest.mock import MagicMock

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Configure logging based on verbosity."""
    verbose = config.getoption("verbose")
    if verbose >= 2:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.CRITICAL)


class ExternalRequestBlockedError(Exception):
    """Raised when a test attempts to make an unmocked external HTTP request."""

    pass


def _block_socket_connect(self: socket.socket, address: Any) -> None:
    """Block all socket connections during tests."""
    host = address[0] if isinstance(address, tuple) else str(address)

    # Allow localhost connections for any test servers
    allowed_hosts = {"localhost", "127.0.0.1", "::1"}
    if host in allowed_hosts:
        # Call the original connect for localhost
        return _original_socket_connect(self, address)

    raise ExternalRequestBlockedError(
        f"\n\n"
        f"{'=' * 70}\n"
        f"BLOCKED: Test attempted to make an external HTTP request!\n"
        f"{'=' * 70}\n"
        f"\n"
        f"Host: {host}\n"
        f"Address: {address}\n"
        f"\n"
        f"All external HTTP requests must be mocked in tests.\n"
        f"\n"
        f"For httpx, use:\n"
        f"    from unittest.mock import patch, MagicMock\n"
        f"    with patch('httpx.Client') as mock_client:\n"
        f"        mock_instance = MagicMock()\n"
        f"        mock_client.return_value.__enter__.return_value = mock_instance\n"
        f"        mock_instance.post.return_value = MagicMock(...)\n"
        f"\n"
        f"For requests, use:\n"
        f"    with patch('requests.post') as mock_post:\n"
        f"        mock_post.return_value = MagicMock(...)\n"
        f"\n"
        f"Or use pytest-httpx or responses libraries for more convenient mocking.\n"
        f"{'=' * 70}\n"
    )


# Store original socket connect
_original_socket_connect = socket.socket.connect


@pytest.fixture(autouse=True)
def block_external_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Automatically block all external HTTP requests in tests.

    This fixture is applied to ALL tests automatically (autouse=True).
    It prevents any test from accidentally making real HTTP requests.

    Localhost connections are still allowed for any test servers.
    """
    monkeypatch.setattr(socket.socket, "connect", _block_socket_connect)


@pytest.fixture
def allow_external_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Explicitly allow external requests for a specific test.

    Use this fixture only when you intentionally need to make real HTTP requests
    (e.g., integration tests that should be skipped in CI).

    Usage:
        @pytest.mark.integration
        def test_real_api(allow_external_requests):
            # This test can make real HTTP requests
            ...
    """
    monkeypatch.setattr(socket.socket, "connect", _original_socket_connect)


@pytest.fixture
def mock_httpx_client() -> MagicMock:
    """
    Provide a pre-configured mock for httpx.Client.

    Returns a MagicMock that can be used to mock httpx.Client responses.

    Usage:
        def test_something(mock_httpx_client):
            mock_httpx_client.post.return_value = MagicMock(
                status_code=200,
                json=lambda: {"choices": [{"message": {"content": "Hello"}}]}
            )
            with patch('httpx.Client') as mock:
                mock.return_value.__enter__.return_value = mock_httpx_client
                # Your test code here
    """
    mock = MagicMock()
    mock.post.return_value = MagicMock(
        status_code=200,
        json=lambda: {
            "choices": [{"message": {"content": "Test response", "role": "assistant"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
        text="Test response",
        raise_for_status=lambda: None,
    )
    return mock
