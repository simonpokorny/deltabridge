# deltabridge
Thin Python wrapper for reading Delta tables from object storage (currently 
Azure Blob Storage) or a local filesystem, with low and stable latency. 
Optimized for repeated reads from long-running Python services.
A typical use case is exposing the final products of a data pipeline 
via a REST API, where request latency should stay predictable.

 > **Note**: The efficiency is achieved by using Rust-based loading of Delta tables through [delta-rs](https://github.com/delta-io/delta-rs)
 > and automatic incremental caching of Delta transaction logs.

## Installation

```bash
pip install deltabridge
```

Or, with [uv](https://docs.astral.sh/uv/):

```bash
uv add deltabridge
```

## Usage

### Client reuse and concurrency

Reuse the same table client for repeated reads to retain its metadata cache.
Each `get_table_client()` call creates a fresh instance. Each load refreshes
the table metadata; changed storage options cause the cached table to be rebuilt.

Threads can share a table client for `load_as_polars()` reads. A per-instance
lock covers metadata refresh and creation of a PyArrow dataset with a fixed
schema and file list. The returned LazyFrame reads that snapshot, even if a
later load refreshes the client. Data collection runs outside the lock, so
different reads can collect concurrently. Metadata preparation happens during
`load_as_polars()`; reading the data remains lazy. Snapshot files must remain
available until collection finishes, and cloud credentials must still be valid.

`load_as_delta()` returns a shared, mutable `DeltaTable`, not an independent
snapshot. Only its refresh and selection are protected. Direct use of that
object, including writes, must not overlap other uses or loads on the same
client without application-level synchronization. Use a dedicated client for
such operations, or an application lock that every caller sharing the client
and its handles observes for the entire operation.

An `AzureDeltaClient` also serializes token refresh across the table clients
created from it. These locks coordinate threads within one process; they do
not coordinate writes to storage from other processes or applications.

See the [concurrency audit](.knowledgebase/concurrency-audit.md) for tested
scenarios, reproduced unsafe boundaries, and limits of the local test coverage.

### Examples

#### Azure

```python
import os

import deltalake
import polars as pl

from deltabridge import PartitionFilterOperator
from deltabridge.azure import AzureDeltaClient

azure_delta_client = AzureDeltaClient()
table_client = azure_delta_client.get_table_client(
    table_uri=os.environ['MY_TABLE_STORAGE_URI'],
)

# Get a DeltaTable instance
delta_table: deltalake.DeltaTable = table_client.load_as_delta()

# Load the data as a Polars LazyFrame
table_ldf: pl.LazyFrame = table_client.load_as_polars()
# Collect to a Polars DataFrame
table_df: pl.DataFrame = table_ldf.filter(pl.col('x') > 3).collect()

# For partitioned tables, push filters down to the partition columns so that
# only matching partitions are read from storage (avoiding a full scan).
# Multiple partition filters are combined using the logical AND operator.
table_df = table_client.load_as_polars(
    partition_filter=[
        ('country', PartitionFilterOperator.IN, ['CZ', 'SK']),
        ('year', PartitionFilterOperator.EQUAL, '2024'),
    ],
).collect()
```

#### Local filesystem

```python
import polars as pl

from deltabridge.local import LocalDeltaClient

MY_TABLE_PATH = '/tmp/my_table'

# Write a table to a local filesystem
pl.DataFrame({'x': [1, 2, 3]}).write_delta(
    target=MY_TABLE_PATH
)

local_delta_client = LocalDeltaClient()
table_client = local_delta_client.get_table_client(
    table_uri=MY_TABLE_PATH  # File path can be used as table URI
)

# Load the data as a Polars LazyFrame and collect it into a DataFrame
table_df = table_client.load_as_polars().collect()
print(table_df)
```

### Databricks tables
If your Delta tables are managed by Databricks (Unity Catalog), they are 
still stored as ordinary Delta tables in object storage. Deltabridge can read 
them directly from the storage, so you can access them without a Databricks 
SQL warehouse or cluster:
* Use the table's storage location (in Azure Blob Storage) as the table URI.
    * You can find it in the Databricks Catalog Explorer UI under *Details* of the table.
* The reading identity needs at least the *Storage Blob Data Reader* permission on the storage location (storage account/container).

## Writing to Delta tables

deltabridge is **read-focused**: it provides no write API, and its optimizations don't apply to writes. This is deliberate:
* write use cases are more varied and harder to abstract well - appends, overwrites, merges/upserts, schema evolution and concurrency control all behave differently
* writes are typically handled upstream by the systems that produce the tables (often Spark/PySpark pipelines)

Writing is still possible: `load_as_delta()` returns a [`deltalake.DeltaTable`](https://delta-io.github.io/delta-rs/) with deltabridge's auth already configured, which you can pass to `deltalake`'s write API:

```python
import deltalake

deltalake.write_deltalake(table_client.load_as_delta(), df, mode='append')
```

## Cloud provider support
Object storage support currently covers Azure Blob Storage (plus the local 
filesystem).
