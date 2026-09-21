# Concurrency audit, 2026-09-21

Tested on `codex/experiment-thread-safe-reads`, then transferred to
`feat/concurrent-table-refresh`. Results apply to the snapshot and token-lock
implementation added after baseline commit
`2e839328cc17d141912b542e02c20a07026c8d3a`.

## Conclusion

No incorrect results were observed for client-managed `load_as_polars()` reads
in the tested local-filesystem scenarios. This is bounded evidence, not proof
of all possible schedules or storage backends.

Direct mutation of the shared object returned by `load_as_delta()` is still
unsafe concurrently with a client read. It can cause exceptions or silent
mixed-version results. This path bypasses the client's lock.

Captured datasets retain metadata, not ownership of physical data files or
renewable credentials. Deleting old files can break pending reads.

## Environment and results

- macOS ARM64, Python 3.13.14.
- Installed stack: deltalake 1.6.0, Polars 1.41.2, PyArrow 24.0.0.
- Full suite: 30 tests passed; Ruff formatting and lint passed.
- Minimum supported Polars: isolated Polars 1.14.0, deltalake 1.6.0,
  PyArrow 24.0.0. All 29 non-smoke tests passed. The project environment and
  dependency declarations were not changed.
- Stress suite: 10 independent subprocess runs, each with a 60-second outer
  timeout, all passed in 39.35 seconds total.
- Across those 10 runs: 13,620 validated collections, including 1,600 delayed
  snapshots and 800 concurrent collections of an identical LazyFrame.
- Main stress scenario: 8 readers and one independent writer, 20 schema-changing
  overwrite commits per run, 200 in total. Another 10 overwrites occurred in
  the retained-LazyFrame scenario.
- Results were checked against exact generation, column names, types, row counts,
  values, and partition selection. Tests do not merely check for exceptions.

## Supported paths exercised

| Scenario | Evidence |
| --- | --- |
| Shared client, concurrent Polars reads and external commits | Eight readers, changing column names/types, real local Delta files |
| Delayed collection after a newer commit and refresh | Original schema and data preserved |
| Same LazyFrame collected from multiple threads | Eight threads, 80 collections per run |
| Partition filtering during writes | Unfiltered, equality, inequality, and `in` filters |
| Refresh while another caller prepares a dataset | Event-controlled lock contention preserves operation order |
| Shared Azure provider across four real local table clients | Eight consumers, one token renewal, one rebuild per table |
| Old frames retained through credential-driven rebuild | Original data still readable locally |
| Credential, rebuild, incremental update, dataset preparation failure | One injected failure, seven successful competing reads, successful retry |
| Token provider exception | Lock released and next request can refresh |

Azure credentials are fake. Local Delta tables exercise the actual table client
and rebuild paths, but storage does not validate those credentials.

## Confirmed unsafe boundaries

### Raw refresh can mix old schema and new files

`test_raw_refresh_bypasses_polars_snapshot_lock` pauses a real dataset build
immediately after schema capture. A second thread overwrites the table and
calls `raw_handle.update_incremental()` while the client's lock remains held.
The scheduling hook changes timing only; Delta I/O and mutation are real.

Two outcomes were reproduced:

1. Version 0 contains `id=1, value=7` with integer schema. Version 1 contains
   `id=2, value='not an integer'` with string schema. Collection raises a Polars
   `ComputeError` wrapping an Arrow parsing failure.
2. With version 1 containing `value='42'`, collection silently returns
   `id=2, value=42` with integer schema. This result matches neither committed
   snapshot's schema and data.

The lock cannot protect operations invoked directly on the returned raw object.
Two calls to `load_as_delta()` can return the exact same object, whose version
changes when another thread refreshes it.

Mitigation: isolate raw operations on a dedicated client, or require every user
of the shared client and handles to hold the same application-level lock.
A stronger library guarantee would require a different raw-handle API or an
independent snapshot rather than exposing the mutable cached table.

### Vacuum can invalidate a pending snapshot

`test_captured_snapshot_requires_files_to_survive_vacuum` creates a LazyFrame,
overwrites the table, then vacuums obsolete files with zero retention on a pytest
temporary table only. Collection raises `ComputeError`/`FileNotFoundError`.
A fresh read succeeds.

This is a file-retention boundary, not a failure of the thread lock. A metadata
snapshot cannot prevent deletion of its files. Production retention must exceed
read lifetime.

The four tests in `test_concurrency_boundaries.py` characterize these current
limitations and therefore pass when the unsafe behavior is reproduced. A green
suite must not be interpreted as full thread safety of raw handles. Revisit
these assertions if the public raw-handle contract changes.

## Regression sensitivity

Two regression tests were also run against a temporary copy of the original
HEAD implementation, without modifying the worktree:

- Delayed snapshot test failed: the old LazyFrame returned the replacement
  integer schema instead of its original string schema.
- Concurrent token test failed: the second caller did not use the provider lock.

Both pass with the experimental implementation.

## Reproduction

Run from the repository root:

```bash
uv run pytest -q
uv run pytest -q tests/test_concurrency_stress.py
uv run pytest -vv tests/test_concurrency_boundaries.py
uv run pytest -q tests/test_concurrency_credentials.py
```

Use an outer process timeout for repeated stress runs: a future timeout alone
cannot stop a native deadlock or prevent executor shutdown from waiting.

## Not established by this audit

- Live Azure Blob reads, real token expiry during delayed collection, network
  timeouts, retries, permission changes, and cloud throttling.
- Other Python versions, free-threaded Python, Windows/Linux, or a full matrix
  of supported dependency versions beyond the two tested Polars versions.
- Multiple concurrent writer processes, fork behavior, or distributed locking.
- Fairness, latency under production-sized metadata, or memory behavior during
  prolonged service operation.

The next useful integration test is a disposable Azure container with actual
credential rotation and delayed reads. Its data lifecycle and access need to be
specified before treating local results as cloud guarantees.
