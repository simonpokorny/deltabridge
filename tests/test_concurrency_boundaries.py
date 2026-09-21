"""Characterize unsupported concurrency boundaries, using real Delta files."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import polars as pl
import pytest
from deltalake import DeltaTable
from deltalake.writer import write_deltalake
from polars.testing import assert_frame_equal

from deltabridge.client import DeltaTableClient


def test_raw_handles_share_version_changes_from_other_thread(tmp_path):
    """A returned raw handle is a shared object, not an immutable snapshot."""
    write_deltalake(tmp_path, pl.DataFrame({'id': [1]}))
    client = DeltaTableClient(str(tmp_path), lambda: {})
    first_handle = client.load_as_delta()
    assert first_handle.version() == 0
    write_deltalake(tmp_path, pl.DataFrame({'id': [2]}), mode='append')

    with ThreadPoolExecutor(max_workers=1) as executor:
        second_handle = executor.submit(client.load_as_delta).result(
            timeout=10
        )

    assert second_handle is first_handle
    assert first_handle.version() == 1
    assert first_handle.to_pyarrow_table().num_rows == 2


@pytest.mark.parametrize('new_value', ['not an integer', '42'])
def test_raw_refresh_bypasses_polars_snapshot_lock(
    tmp_path, monkeypatch, new_value
):
    """Unsupported raw mutation can mix old schema with new data files."""
    original = pl.DataFrame({'id': [1], 'value': [7]})
    replacement = pl.DataFrame({'id': [2], 'value': [new_value]})
    write_deltalake(tmp_path, original)
    client = DeltaTableClient(str(tmp_path), lambda: {})
    raw_handle = client.load_as_delta()
    schema_captured = Event()
    resume_dataset = Event()
    real_schema = raw_handle.schema

    def capture_old_schema():
        schema = real_schema()
        schema_captured.set()
        assert resume_dataset.wait(timeout=10)
        return schema

    monkeypatch.setattr(raw_handle, 'schema', capture_old_schema)
    with ThreadPoolExecutor(max_workers=2) as executor:
        pending = executor.submit(client.load_as_polars)
        try:
            assert schema_captured.wait(timeout=5)
            assert client._refresh_lock.locked()
            write_deltalake(
                tmp_path,
                replacement,
                mode='overwrite',
                schema_mode='overwrite',
            )
            executor.submit(raw_handle.update_incremental).result(timeout=5)
            assert client._refresh_lock.locked()
            assert raw_handle.version() == 1
        finally:
            resume_dataset.set()
        mixed_snapshot = pending.result(timeout=10)

    # This asserts a documented unsupported boundary, not successful isolation.
    if new_value == 'not an integer':
        with pytest.raises(
            pl.exceptions.ComputeError,
            match="ArrowInvalid: Failed to parse string: 'not an integer'",
        ):
            mixed_snapshot.collect()
    else:
        # Coercible strings silently become integers under the stale schema.
        # This result matches neither committed snapshot's schema and data.
        assert_frame_equal(
            mixed_snapshot.collect(), pl.DataFrame({'id': [2], 'value': [42]})
        )
    monkeypatch.undo()
    assert_frame_equal(client.load_as_polars().collect(), replacement)


def test_captured_snapshot_requires_files_to_survive_vacuum(tmp_path):
    """Dataset isolation retains metadata but cannot preserve deleted files."""
    original = pl.DataFrame({'id': [1]})
    replacement = pl.DataFrame({'id': [2]})
    write_deltalake(tmp_path, original)
    client = DeltaTableClient(str(tmp_path), lambda: {})
    pending_read = client.load_as_polars()
    assert_frame_equal(pending_read.collect(), original)
    write_deltalake(tmp_path, replacement, mode='overwrite')

    # Unsafe zero retention is deliberate and confined to this temporary table.
    removed = DeltaTable(str(tmp_path)).vacuum(
        retention_hours=0, dry_run=False, enforce_retention_duration=False
    )
    assert removed
    with pytest.raises(pl.exceptions.ComputeError, match='FileNotFoundError'):
        pending_read.collect()
    assert_frame_equal(client.load_as_polars().collect(), replacement)
