from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from threading import Barrier
from unittest.mock import Mock

import polars as pl
import pytest
from azure.core.credentials import AccessToken, TokenCredential
from deltalake import DeltaTable
from deltalake.writer import write_deltalake
from polars.testing import assert_frame_equal

from deltabridge.azure import AzureDeltaClient
from deltabridge.client import DeltaTableClient


def test_shared_azure_provider_rotates_across_real_table_clients(tmp_path):
    expires = int((datetime.now() + timedelta(hours=1)).timestamp())
    credential = Mock(spec=TokenCredential)
    credential.get_token.side_effect = [
        AccessToken('old', expires),
        AccessToken('new', expires),
    ]
    provider = AzureDeltaClient(credential)
    clients = []
    pending_reads = []
    rebuilds = []
    for index in range(4):
        path = tmp_path / str(index)
        original = pl.DataFrame({'table': [index], 'generation': [0]})
        write_deltalake(path, original)
        client = provider.get_table_client(str(path))
        pending_reads.append(client.load_as_polars())
        create = Mock(wraps=client._create_delta_table)
        client._create_delta_table = create
        rebuilds.append(create)
        clients.append(client)
        write_deltalake(
            path,
            pl.DataFrame({'table': [index], 'generation': [1]}),
            mode='overwrite',
        )

    provider._token_obj = AccessToken('old', 0)
    start = Barrier(9, timeout=10)

    def read(index):
        start.wait()
        return clients[index].load_as_polars().collect()

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(read, index % 4) for index in range(8)]
        start.wait()
        results = [future.result(timeout=10) for future in futures]

    for index, result in enumerate(results):
        assert_frame_equal(
            result,
            pl.DataFrame({'table': [index % 4], 'generation': [1]}),
        )
    for index, pending in enumerate(pending_reads):
        assert_frame_equal(
            pending.collect(),
            pl.DataFrame({'table': [index], 'generation': [0]}),
        )
        rebuilds[index].assert_called_once_with({'azure_storage_token': 'new'})
        assert clients[index]._storage_options == {
            'azure_storage_token': 'new'
        }
    assert credential.get_token.call_count == 2


@pytest.mark.parametrize(
    'failure_stage', ['credentials', 'rebuild', 'update', 'dataset']
)
def test_concurrent_read_recovers_after_transient_failure(
    tmp_path, mocker, failure_stage
):
    expected = pl.DataFrame({'value': [1, 2, 3]})
    write_deltalake(tmp_path, expected)
    options = {'token': 'old'}
    client = DeltaTableClient(str(tmp_path), lambda: dict(options))
    original = client._delta_table

    if failure_stage == 'credentials':
        target, name = client, '_storage_options_fn'
    elif failure_stage == 'rebuild':
        options['token'] = 'new'
        target, name = client, '_create_delta_table'
    elif failure_stage == 'update':
        target, name = original, 'update_incremental'
    else:
        target, name = original, 'to_pyarrow_dataset'
    operation = getattr(target, name)
    failed = False

    def fail_once(*args, **kwargs):
        nonlocal failed
        # These operations run inside the table client's lock.
        if not failed:
            failed = True
            raise ConnectionError('injected transient failure')
        return operation(*args, **kwargs)

    mocker.patch.object(target, name, side_effect=fail_once)
    start = Barrier(9, timeout=10)

    def read():
        start.wait()
        return client.load_as_polars().collect()

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(read) for _ in range(8)]
        start.wait()
        errors = 0
        for future in futures:
            try:
                result = future.result(timeout=10)
            except ConnectionError as error:
                assert str(error) == 'injected transient failure'
                errors += 1
            else:
                assert_frame_equal(result, expected)
    assert errors == 1
    assert_frame_equal(client.load_as_polars().collect(), expected)
    assert isinstance(client._delta_table, DeltaTable)
    assert client._storage_options == options
