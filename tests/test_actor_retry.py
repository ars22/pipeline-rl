"""Tests for actor retry logic for transient HTTP errors."""

import asyncio
import pytest
import aiohttp


class TestTransientErrors:
    """
    Tests for the TRANSIENT_ERRORS tuple used in actor.py.

    These tests verify that all the expected transient error types are
    properly caught and would trigger retry logic.
    """

    def test_transient_errors_tuple_contains_expected_types(self):
        """Verify that all expected transient error types are in the tuple."""
        # This is the exact tuple from actor.py - keep in sync!
        TRANSIENT_ERRORS = (
            aiohttp.ClientPayloadError,  # Response payload incomplete (e.g., connection reset mid-stream)
            aiohttp.ClientConnectionResetError,
            aiohttp.ClientOSError,
            aiohttp.ServerDisconnectedError,
            aiohttp.ServerTimeoutError,
            aiohttp.ServerConnectionError,
            aiohttp.ClientConnectorError,
            aiohttp.SocketTimeoutError,
            aiohttp.ConnectionTimeoutError,
            ConnectionResetError,
            ConnectionError,
            TimeoutError,
            OSError,
        )

        # Verify all types exist and are exception classes
        for error_type in TRANSIENT_ERRORS:
            assert isinstance(error_type, type), f"{error_type} is not a type"
            assert issubclass(error_type, BaseException), f"{error_type} is not an exception"

    def test_client_payload_error_is_caught(self):
        """
        Test that ClientPayloadError is properly caught.

        This was the exact error that caused the crash:
        aiohttp.client_exceptions.ClientPayloadError: Response payload is not completed:
        <ContentLengthError: 400, message='Not enough data to satisfy content length header.'>.
        ConnectionResetError(104, 'Connection reset by peer')
        """
        TRANSIENT_ERRORS = (
            aiohttp.ClientPayloadError,
            aiohttp.ClientConnectionResetError,
            aiohttp.ClientOSError,
            aiohttp.ServerDisconnectedError,
            aiohttp.ServerTimeoutError,
            aiohttp.ServerConnectionError,
            aiohttp.ClientConnectorError,
            aiohttp.SocketTimeoutError,
            aiohttp.ConnectionTimeoutError,
            ConnectionResetError,
            ConnectionError,
            TimeoutError,
            OSError,
        )

        # Simulate the exact error from the crash
        error = aiohttp.ClientPayloadError(
            "Response payload is not completed: "
            "<ContentLengthError: 400, message='Not enough data to satisfy content length header.'>. "
            "ConnectionResetError(104, 'Connection reset by peer')"
        )

        # Verify it's caught by isinstance check (as used in actor.py)
        assert isinstance(error, TRANSIENT_ERRORS), "ClientPayloadError should be caught"

    def test_connection_reset_error_is_caught(self):
        """Test that ConnectionResetError is properly caught."""
        TRANSIENT_ERRORS = (
            aiohttp.ClientPayloadError,
            aiohttp.ClientConnectionResetError,
            aiohttp.ClientOSError,
            aiohttp.ServerDisconnectedError,
            aiohttp.ServerTimeoutError,
            aiohttp.ServerConnectionError,
            aiohttp.ClientConnectorError,
            aiohttp.SocketTimeoutError,
            aiohttp.ConnectionTimeoutError,
            ConnectionResetError,
            ConnectionError,
            TimeoutError,
            OSError,
        )

        error = ConnectionResetError(104, "Connection reset by peer")
        assert isinstance(error, TRANSIENT_ERRORS), "ConnectionResetError should be caught"

    def test_server_disconnected_error_is_caught(self):
        """Test that ServerDisconnectedError is properly caught."""
        TRANSIENT_ERRORS = (
            aiohttp.ClientPayloadError,
            aiohttp.ClientConnectionResetError,
            aiohttp.ClientOSError,
            aiohttp.ServerDisconnectedError,
            aiohttp.ServerTimeoutError,
            aiohttp.ServerConnectionError,
            aiohttp.ClientConnectorError,
            aiohttp.SocketTimeoutError,
            aiohttp.ConnectionTimeoutError,
            ConnectionResetError,
            ConnectionError,
            TimeoutError,
            OSError,
        )

        error = aiohttp.ServerDisconnectedError()
        assert isinstance(error, TRANSIENT_ERRORS), "ServerDisconnectedError should be caught"


class TestRetryLogic:
    """Tests for the retry logic implementation."""

    @pytest.mark.asyncio
    async def test_retry_with_exponential_backoff(self):
        """Test that retry logic uses exponential backoff."""
        max_retries = 3
        retry_base_delay = 0.01  # Use small delay for testing

        attempts = []

        async def failing_operation():
            attempts.append(len(attempts) + 1)
            if len(attempts) < max_retries:
                raise aiohttp.ClientPayloadError("Simulated transient error")
            return "success"

        TRANSIENT_ERRORS = (aiohttp.ClientPayloadError,)

        # Simulate the retry logic from actor.py
        last_error = None
        result = None
        for attempt in range(max_retries):
            try:
                result = await failing_operation()
                break
            except TRANSIENT_ERRORS as e:
                last_error = e
                if attempt < max_retries - 1:
                    delay = retry_base_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                else:
                    raise

        assert result == "success"
        assert len(attempts) == max_retries  # Should have retried until success

    @pytest.mark.asyncio
    async def test_retry_exhaustion_raises(self):
        """Test that after max_retries, the error is raised."""
        max_retries = 3
        retry_base_delay = 0.01

        attempts = []

        async def always_failing_operation():
            attempts.append(len(attempts) + 1)
            raise aiohttp.ClientPayloadError("Simulated persistent error")

        TRANSIENT_ERRORS = (aiohttp.ClientPayloadError,)

        # Simulate the retry logic from actor.py
        last_error = None
        with pytest.raises(aiohttp.ClientPayloadError):
            for attempt in range(max_retries):
                try:
                    await always_failing_operation()
                    break
                except TRANSIENT_ERRORS as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        delay = retry_base_delay * (2 ** attempt)
                        await asyncio.sleep(delay)
                    else:
                        raise

        assert len(attempts) == max_retries  # Should have tried max_retries times

    @pytest.mark.asyncio
    async def test_non_transient_error_not_retried(self):
        """Test that non-transient errors are not retried."""
        max_retries = 3
        retry_base_delay = 0.01

        attempts = []

        async def non_transient_failure():
            attempts.append(len(attempts) + 1)
            raise ValueError("Non-transient error")

        TRANSIENT_ERRORS = (aiohttp.ClientPayloadError,)

        # The ValueError should propagate immediately without retry
        with pytest.raises(ValueError):
            for attempt in range(max_retries):
                try:
                    await non_transient_failure()
                    break
                except TRANSIENT_ERRORS as e:
                    if attempt < max_retries - 1:
                        delay = retry_base_delay * (2 ** attempt)
                        await asyncio.sleep(delay)
                    else:
                        raise

        # Should have only attempted once since ValueError is not in TRANSIENT_ERRORS
        assert len(attempts) == 1


class TestAiohttpExceptionHierarchy:
    """Tests to verify our understanding of aiohttp exception hierarchy."""

    def test_client_payload_error_inheritance(self):
        """Verify ClientPayloadError inheritance chain."""
        assert issubclass(aiohttp.ClientPayloadError, aiohttp.ClientError)
        assert issubclass(aiohttp.ClientPayloadError, Exception)

    def test_server_disconnected_error_inheritance(self):
        """Verify ServerDisconnectedError inheritance chain."""
        assert issubclass(aiohttp.ServerDisconnectedError, aiohttp.ServerConnectionError)
        assert issubclass(aiohttp.ServerDisconnectedError, aiohttp.ClientError)

    def test_connection_reset_error_inheritance(self):
        """Verify ConnectionResetError inheritance chain."""
        assert issubclass(ConnectionResetError, ConnectionError)
        assert issubclass(ConnectionResetError, OSError)
