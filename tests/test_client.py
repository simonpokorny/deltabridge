import tempfile
from datetime import datetime
from pathlib import Path

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


def test_load_as_delta_reuses_cache_after_append(tmp_path, mocker):
    original = pl.DataFrame({'id': [1]})
    appended = pl.DataFrame({'id': [2]})
    write_deltalake(tmp_path, original)
    client = DeltaTableClient(str(tmp_path), lambda: {})
    constructor = mocker.patch(
        'deltabridge.client.DeltaTable',
        side_effect=AssertionError('unchanged options must reuse the cache'),
    )

    table = client.load_as_delta()
    assert table.version() == 0
    assert client.load_as_delta() is table

    write_deltalake(tmp_path, appended, mode='append')
    assert client.load_as_delta() is table
    assert table.version() == 1
    assert_frame_equal(
        pl.from_arrow(table.to_pyarrow_table()).sort('id'),
        pl.concat([original, appended]),
    )
    constructor.assert_not_called()


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


@pytest.mark.parametrize('load_method', ['load_as_delta', 'load_as_polars'])
def test_cached_delta_refresh_preserves_polars_snapshot(tmp_path, load_method):
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
    refreshed = getattr(client, load_method)()
    if load_method == 'load_as_delta':
        assert refreshed is delta_table
    assert client._delta_table is delta_table
    assert delta_table.version() == 1
    assert_frame_equal(
        pl.from_arrow(delta_table.to_pyarrow_table()), replacement
    )

    assert_frame_equal(pending_read.collect(), original)
    assert_frame_equal(client.load_as_polars().collect(), replacement)


@pytest.mark.parametrize('load_method', ['load_as_delta', 'load_as_polars'])
def test_loads_hold_refresh_lock(temp_delta_table_uri, mocker, load_method):
    client = DeltaTableClient(temp_delta_table_uri, lambda: {})
    update_incremental = client._delta_table.update_incremental

    def refresh():
        assert client._refresh_lock.locked()
        update_incremental()

    refresh_mock = mocker.patch.object(
        client._delta_table, 'update_incremental', side_effect=refresh
    )
    if load_method == 'load_as_polars':
        create_dataset = client._delta_table.to_pyarrow_dataset

        def dataset(**kwargs):
            assert client._refresh_lock.locked()
            return create_dataset(**kwargs)

        dataset_mock = mocker.patch.object(
            client._delta_table, 'to_pyarrow_dataset', side_effect=dataset
        )

    getattr(client, load_method)()

    refresh_mock.assert_called_once_with()
    if load_method == 'load_as_polars':
        dataset_mock.assert_called_once_with(partitions=None)


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

    assert result is new_table
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
    assert result is new_table
    assert client._storage_options == {'token': 'new'}
    assert client._create_delta_table.call_count == 2
