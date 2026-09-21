from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from queue import Queue
from threading import Event, Lock
from unittest.mock import Mock

import pytest
from azure.core.credentials import AccessToken, TokenCredential

from deltabridge.azure import AzureDeltaClient


def test_token_client_token_not_expired():
    # Create a mock credential
    mock_credential = Mock(spec=TokenCredential)
    mock_credential.get_token.side_effect = [
        AccessToken(
            token='test-token',
            expires_on=int((datetime.now() + timedelta(hours=1)).timestamp()),
        ),
        AssertionError('Token should not be refreshed'),
    ]

    delta_client = AzureDeltaClient(credential=mock_credential)
    for _ in range(2):
        assert delta_client._get_storage_options() == {
            'azure_storage_token': 'test-token'
        }, 'Token should not be refreshed'


def test_token_client_token_expired():
    # Create a mock credential
    mock_credential = Mock(spec=TokenCredential)
    mock_credential.get_token.side_effect = [
        AccessToken(
            token='old-token',
            expires_on=int(datetime.now().timestamp()),
        ),
        AccessToken(
            token='new-token',
            expires_on=int((datetime.now() + timedelta(hours=1)).timestamp()),
        ),
        AssertionError('Token should not be refreshed'),
    ]

    delta_client = AzureDeltaClient(credential=mock_credential)
    for _ in range(3):
        assert delta_client._get_storage_options() == {
            'azure_storage_token': 'new-token'
        }, 'Token should be refreshed'


def test_concurrent_consumers_share_one_token_refresh():
    refresh_started = Event()
    release_refresh = Event()
    attempts = Queue()

    class ObservedLock:
        def __init__(self):
            self.lock = Lock()

        def __enter__(self):
            attempts.put(None)
            assert self.lock.acquire(timeout=5)

        def __exit__(self, *_):
            self.lock.release()

    old_token = AccessToken('old-token', 0)
    fresh_token = AccessToken(
        'new-token',
        int((datetime.now() + timedelta(hours=1)).timestamp()),
    )
    credential = Mock(spec=TokenCredential)
    credential.get_token.return_value = old_token
    client = AzureDeltaClient(credential=credential)
    client._token_lock = ObservedLock()

    def refresh(_scope):
        refresh_started.set()
        assert release_refresh.wait(timeout=5)
        return fresh_token

    credential.get_token.side_effect = refresh
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(client._get_storage_options)
        try:
            assert refresh_started.wait(timeout=5)
            second = executor.submit(client._get_storage_options)
            attempts.get(timeout=5)
            attempts.get(timeout=5)
        finally:
            release_refresh.set()

        expected = {'azure_storage_token': 'new-token'}
        assert first.result(timeout=5) == expected
        assert second.result(timeout=5) == expected

    assert credential.get_token.call_count == 2  # Initial token + one refresh.


def test_failed_token_refresh_releases_lock_and_can_retry():
    credential = Mock(spec=TokenCredential)
    credential.get_token.side_effect = [
        AccessToken('old-token', 0),
        RuntimeError('temporary credential failure'),
        AccessToken(
            'new-token',
            int((datetime.now() + timedelta(hours=1)).timestamp()),
        ),
    ]
    client = AzureDeltaClient(credential=credential)

    with pytest.raises(RuntimeError, match='temporary credential failure'):
        client._get_storage_options()

    assert client._token_lock.acquire(timeout=1)
    client._token_lock.release()
    assert client._get_storage_options() == {
        'azure_storage_token': 'new-token'
    }
    assert credential.get_token.call_count == 3
