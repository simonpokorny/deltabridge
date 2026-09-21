import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from queue import Queue
from threading import Event, Lock

import polars as pl
import pytest
from deltalake import DeltaTable
from deltalake.writer import write_deltalake
from polars.testing import assert_frame_equal

from deltabridge import PartitionFilterOperator
from deltabridge.client import DeltaTableClient


@pytest.fixture
def sample_df():
    return pl.DataFrame(
        {
            'id': [1, 2, 2, 3],
            'value': ['a', 'b', 'c', 'c'],
            'datetime': [
                datetime.now(),
                datetime.now(),
                datetime.now(),
                datetime.now(),
            ],
        },
        schema={'id': pl.Int64, 'value': pl.Utf8, 'datetime': pl.Datetime},
    )


@pytest.fixture
def temp_delta_table_uri(sample_df):
    with tempfile.TemporaryDirectory() as tmpdir:
        write_deltalake(tmpdir, sample_df, partition_by=['id', 'value'])
        yield tmpdir


def test_load_as_delta(temp_delta_table_uri):
    delta_table_client = DeltaTableClient(
        table_uri=temp_delta_table_uri,
        storage_options_fn=lambda: {},
    )
    loaded_delta_table = delta_table_client.load_as_delta()
    assert isinstance(loaded_delta_table, DeltaTable)
    # delta-rs started prefixing file:// to the table URI in an unknwon version
    # removing the prefix ensures compatibility with both old and new versions
    # samefile() compares by inode, so symlinked temp dirs
    # (e.g. macOS /var -> /private/var) still match instead of needing
    # identical textual paths.
    assert Path(loaded_delta_table.table_uri.replace('file:', '')).samefile(
        temp_delta_table_uri
    )


def test_load_as_polars(temp_delta_table_uri, sample_df):
    delta_table_client = DeltaTableClient(
        table_uri=temp_delta_table_uri,
        storage_options_fn=lambda: {},
    )
    assert_frame_equal(
        # Sort both frames to ensure the order is the same
        delta_table_client.load_as_polars().sort('id', 'value').collect(),
        sample_df,
    )


def test_polars_snapshot_stays_stable_while_delta_handle_refreshes(tmp_path):
    original = pl.DataFrame({'id': [1], 'value': ['original']})
    replacement = pl.DataFrame({'id': [2], 'value': [42]})
    write_deltalake(tmp_path, original)
    client = DeltaTableClient(str(tmp_path), lambda: {})
    delta_table = client.load_as_delta()
    pending_read = client.load_as_polars()
    assert delta_table.version() == 0

    write_deltalake(
        tmp_path, replacement, mode='overwrite', schema_mode='overwrite'
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        refreshed = executor.submit(client.load_as_delta).result(timeout=10)
    assert refreshed is not delta_table
    assert refreshed.version() == 1
    assert delta_table.version() == 0
    assert_frame_equal(pl.from_arrow(delta_table.to_pyarrow_table()), original)

    assert_frame_equal(pending_read.collect(), original)
    assert_frame_equal(client.load_as_polars().collect(), replacement)


def test_concurrent_polars_reads_serialize_dataset_and_refresh(
    temp_delta_table_uri, sample_df, mocker
):
    client = DeltaTableClient(temp_delta_table_uri, lambda: {})
    dataset_started = Event()
    release_dataset = Event()
    lock = Lock()
    attempts = Queue()

    class ObservedLock:
        def __enter__(self):
            attempts.put(None)
            assert lock.acquire(timeout=10)

        def __exit__(self, *args):
            lock.release()

    client._refresh_lock = ObservedLock()
    create_dataset = client._delta_table.to_pyarrow_dataset
    update_incremental = client._delta_table.update_incremental
    operations = []

    def refresh():
        assert lock.locked()
        operations.append('refresh')
        update_incremental()

    def prepare_dataset(**kwargs):
        assert lock.locked()
        operations.append('dataset')
        if not dataset_started.is_set():
            dataset_started.set()
            assert release_dataset.wait(timeout=10)
        return create_dataset(**kwargs)

    mocker.patch.object(
        client._delta_table, 'update_incremental', side_effect=refresh
    )
    mocker.patch.object(
        client._delta_table, 'to_pyarrow_dataset', side_effect=prepare_dataset
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(client.load_as_polars)
        try:
            assert dataset_started.wait(timeout=5)
            second = executor.submit(client.load_as_polars)
            attempts.get(timeout=5)
            attempts.get(timeout=5)
            assert operations == ['refresh', 'dataset']
        finally:
            release_dataset.set()
        frames = [first.result(timeout=10), second.result(timeout=10)]

    assert operations == ['refresh', 'dataset', 'refresh', 'dataset']
    for frame in frames:
        assert_frame_equal(frame.sort('id', 'value').collect(), sample_df)


@pytest.mark.parametrize(
    ('partition_filter', 'expected_filter'),
    [
        (
            [],
            lambda df: df,
        ),
        (
            [
                ('id', PartitionFilterOperator.EQUAL, '2'),
                ('value', '=', 'c'),  # Test with string literal
            ],
            lambda df: df.filter(
                (pl.col('id') == 2) & (pl.col('value') == 'c')
            ),
        ),
        (
            [('id', PartitionFilterOperator.IN, ['2', '3'])],
            lambda df: df.filter(pl.col('id').is_in([2, 3])),
        ),
        (
            [('value', PartitionFilterOperator.NOT_IN, ['b', 'c'])],
            lambda df: df.filter(~pl.col('value').is_in(['b', 'c'])),
        ),
        (
            [('id', PartitionFilterOperator.NOT_EQUAL, '2')],
            lambda df: df.filter(pl.col('id') != 2),
        ),
    ],
    ids=['empty', 'eq', 'in', 'not-in', 'not-eq'],
)
def test_load_as_polars_with_partition_operators(
    temp_delta_table_uri, sample_df, partition_filter, expected_filter
):
    delta_table_client = DeltaTableClient(
        table_uri=temp_delta_table_uri,
        storage_options_fn=lambda: {},
    )
    loaded_df = (
        delta_table_client.load_as_polars(
            partition_filter=partition_filter,
        )
        .sort('id', 'value')
        .collect()
    )
    correct_partition_df = expected_filter(sample_df).sort('id', 'value')
    assert_frame_equal(
        loaded_df,
        correct_partition_df,
    )


def test_load_invalid_partition_filter(temp_delta_table_uri):
    delta_table_client = DeltaTableClient(
        table_uri=temp_delta_table_uri,
        storage_options_fn=lambda: {},
    )
    with pytest.raises(ValueError):
        delta_table_client.load_as_polars(
            partition_filter=[('id', 'invalid', '2')],
        )


def test_storage_options_rotation_rebuilds_table(temp_delta_table_uri, mocker):
    options = {'token': 'old'}
    client = DeltaTableClient(
        table_uri=temp_delta_table_uri,
        storage_options_fn=lambda: dict(options),
    )
    new_table = DeltaTable(temp_delta_table_uri)
    mocker.patch.object(client, '_create_delta_table', return_value=new_table)

    options['token'] = 'new'
    result = client.load_as_delta()

    assert result is not new_table
    assert result.version() == new_table.version()
    assert client._delta_table is new_table
    assert client._storage_options == {'token': 'new'}
    # The rebuild must use the REFRESHED options, not the stale ones.
    client._create_delta_table.assert_called_once_with({'token': 'new'})


def test_storage_options_rotation_failed_rebuild(temp_delta_table_uri, mocker):
    options = {'token': 'old'}
    client = DeltaTableClient(
        table_uri=temp_delta_table_uri,
        storage_options_fn=lambda: dict(options),
    )
    original_table = client._delta_table
    new_table = DeltaTable(temp_delta_table_uri)
    mocker.patch.object(
        client,
        '_create_delta_table',
        side_effect=[ConnectionError('transient'), new_table],
    )

    options['token'] = 'new'

    # First call: rebuild fails -> error propagates, state unchanged
    with pytest.raises(ConnectionError):
        client.load_as_delta()
    assert client._delta_table is original_table
    assert client._storage_options == {'token': 'old'}

    # Second call: rebuild succeeds -> new table committed
    result = client.load_as_delta()
    assert result is not new_table
    assert result.version() == new_table.version()
    assert client._storage_options == {'token': 'new'}
    assert client._create_delta_table.call_count == 2
