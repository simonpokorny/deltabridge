"""Bounded real-Delta concurrency tests, with no mocked storage operations."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import polars as pl
from deltalake.writer import write_deltalake
from polars.testing import assert_frame_equal

from deltabridge.client import DeltaTableClient


def _generation_frame(generation):
    size = 12 + generation % 5
    values = list(range(size))
    # Changing column names and types catches mixed schema/file snapshots.
    payload = (
        [f'{generation}:{value}' for value in values]
        if generation % 2
        else [generation * 100 + value for value in values]
    )
    return pl.DataFrame(
        {
            'generation': [generation] * size,
            'id': values,
            'bucket': [value % 3 for value in values],
            f'payload_{generation % 3}': payload,
        }
    )


def _write_generation(path, generation):
    write_deltalake(
        path,
        _generation_frame(generation),
        mode='overwrite',
        schema_mode='overwrite',
        partition_by=['bucket'],
    )


def _reader_filter(reader):
    return [
        (None, [0, 1, 2]),
        ([('bucket', '=', '0')], [0]),
        ([('bucket', 'in', ['0', '2'])], [0, 2]),
        ([('bucket', '!=', '1')], [0, 2]),
    ][reader % 4]


def _assert_generation(frame, allowed_generations, buckets):
    generations = frame['generation'].unique().to_list()
    assert len(generations) == 1
    generation = generations[0]
    assert generation in allowed_generations
    expected = _generation_frame(generation).filter(
        pl.col('bucket').is_in(buckets)
    )
    assert_frame_equal(frame.sort('id'), expected.sort('id'))


def test_eight_readers_and_schema_changing_writer(tmp_path):
    """Validate 1,280 collections, including 160 delayed snapshots."""
    rounds = 20
    readers = 8
    live_reads_per_round = 7
    _write_generation(tmp_path, 0)
    client = DeltaTableClient(str(tmp_path), lambda: {})
    phase = Barrier(readers + 1, timeout=30)

    def read(reader):
        partition_filter, buckets = _reader_filter(reader)
        collections = 0
        try:
            for generation in range(1, rounds + 1):
                pending = client.load_as_polars(partition_filter)
                phase.wait()
                for _ in range(live_reads_per_round):
                    frame = client.load_as_polars(partition_filter).collect()
                    _assert_generation(
                        frame, {generation - 1, generation}, buckets
                    )
                    collections += 1
                # The writer has committed before delayed collection begins.
                phase.wait()
                _assert_generation(
                    pending.collect(), {generation - 1}, buckets
                )
                collections += 1
                phase.wait()
        except BaseException:
            phase.abort()
            raise
        return collections

    def write():
        try:
            for generation in range(1, rounds + 1):
                phase.wait()
                _write_generation(tmp_path, generation)
                phase.wait()
                phase.wait()
        except BaseException:
            phase.abort()
            raise

    with ThreadPoolExecutor(max_workers=readers + 1) as pool:
        writer = pool.submit(write)
        results = [pool.submit(read, reader) for reader in range(readers)]
        counts = [result.result(timeout=120) for result in results]
        writer.result(timeout=120)

    assert sum(counts) == rounds * readers * (live_reads_per_round + 1)
    _assert_generation(client.load_as_polars().collect(), {rounds}, [0, 1, 2])


def test_same_lazy_frame_can_be_collected_from_eight_threads(tmp_path):
    """Collect one retained frame 80 times after its client changes schema."""
    _write_generation(tmp_path, 0)
    client = DeltaTableClient(str(tmp_path), lambda: {})
    pending = client.load_as_polars()
    _write_generation(tmp_path, 1)
    _assert_generation(client.load_as_polars().collect(), {1}, [0, 1, 2])
    start = Barrier(8, timeout=30)

    def read():
        try:
            start.wait()
            for _ in range(10):
                _assert_generation(pending.collect(), {0}, [0, 1, 2])
        except BaseException:
            start.abort()
            raise

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = [pool.submit(read) for _ in range(8)]
        for result in results:
            result.result(timeout=120)
