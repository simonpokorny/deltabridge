from __future__ import annotations

from enum import StrEnum
from typing import Any, Callable

import polars as pl
from deltalake import DeltaTable
from deltalake.exceptions import DeltaProtocolError
from deltalake.table import (
    MAX_SUPPORTED_READER_VERSION,
    SUPPORTED_READER_FEATURES,
)


class PartitionFilterOperator(StrEnum):
    EQUAL = '='
    NOT_EQUAL = '!='
    IN = 'in'
    NOT_IN = 'not in'


class DeltaTableClient:
    """
    Delta table client - used for accessing tables in Delta format
    stored in a local or cloud storage.

    The client is used to load the Delta table as a Polars LazyFrame
    or as a DeltaTable object.

    Notes
    -----
    The client should never be initialized directly. Instead, use the method
    `get_table_client` of a class derived from `BaseDeltaClient` to
    get a client for a specific storage provider.

    Parameters
    ----------
    table_uri: str
        URI of the Delta table.
    storage_options_fn: Callable[[], dict[str, str]]
        Function which returns the storage options for the Delta table.
        Storage options typically include an access token for the storage
        in which the Delta table is stored.
    """

    def __init__(
        self,
        table_uri: str,
        storage_options_fn: Callable[[], dict[str, str]],
    ):
        self._table_uri = table_uri
        self._storage_options_fn = storage_options_fn
        self._storage_options = self._storage_options_fn()
        self._delta_table = self._create_delta_table(self._storage_options)

    def _create_delta_table(
        self, storage_options: dict[str, str]
    ) -> DeltaTable:
        return DeltaTable(
            self._table_uri,
            storage_options=storage_options,
        )

    def _refresh_table(self) -> None:
        refreshed_storage_options = self._storage_options_fn()
        if self._storage_options != refreshed_storage_options:
            # The storage options have changed -> recreate DeltaTable instance
            delta_table = self._create_delta_table(refreshed_storage_options)
            self._storage_options = refreshed_storage_options
            self._delta_table = delta_table
        else:
            # Update table metadata using existing token
            self._delta_table.update_incremental()

    def load_as_delta(self) -> DeltaTable:
        """Load a Delta table.

        Returns
        -------
        DeltaTable
            A DeltaTable object representing the loaded table.
        """
        self._refresh_table()
        return self._delta_table

    def load_as_polars(
        self,
        partition_filter: list[tuple[str, PartitionFilterOperator, Any]]
        | None = None,
    ) -> pl.LazyFrame:
        """Load a Delta table, with optional partition filtering.

        Tables with deletion vectors can be read only without
        ``partition_filter``.

        Parameters
        ----------
        partition_filter
            Iterable of tuples containing the column name, operator and value
            to filter the table by partition columns.
            If multiple partition filters are provided, they are combined using
            the logical AND operator.
            If not provided, no partition filtering will be applied.

        Returns
        -------
        polars.LazyFrame
            A Polars LazyFrame representing the scanned Delta table.
            If partition filtering is applied, only matching rows
            are included.

        Raises
        ------
        ValueError
            If an invalid partition filter operator is provided.
        deltalake.exceptions.DeltaProtocolError
            If the table uses column mapping or another unsupported Delta
            feature, or deletion vectors with ``partition_filter``.
        """
        table = self.load_as_delta()
        protocol = table.protocol()

        # deltalake>=1.6.4 no longer rejects reader version 2 (column mapping)
        # and reads the columns as nulls (delta-io/delta-rs#4712)
        if protocol.min_reader_version == 2:
            raise DeltaProtocolError(
                'The table uses column mapping (reader version 2), '
                'which cannot be read.'
            )

        if partition_filter:
            for _, operator, _ in partition_filter:
                # Raises ValueError if invalid
                PartitionFilterOperator(operator)
            # pyarrow prunes partitions before listing files (the native
            # reader does not, pola-rs/polars#20998), but rejects tables with
            # deletion vectors
            return pl.scan_delta(
                source=table,
                use_pyarrow=True,
                pyarrow_options={'partitions': partition_filter},
            )

        # pl.scan_delta skips its protocol check when given a DeltaTable
        features = set(protocol.reader_features or [])
        unsupported = (
            features - SUPPORTED_READER_FEATURES - {'deletionVectors'}
        )
        if (
            protocol.min_reader_version > MAX_SUPPORTED_READER_VERSION
            or unsupported
        ):
            raise DeltaProtocolError(
                'The table requires reader version '
                f'{protocol.min_reader_version} with reader features '
                f'{sorted(features)}, which cannot be read.'
            )
        return pl.scan_delta(source=table)
