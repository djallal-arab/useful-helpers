import sqlite3
from typing import List, Tuple, Optional
from contextlib import contextmanager
import logging
import csv
import io
from queue import Queue
from threading import Lock


class SQLiteHelper:
    def __init__(self, db_path: str, pool_size: int = 5, enable_wal: bool = True):
        """
        Initialize the SQLiteHelper with the database file path.

        :param db_path: Path to the SQLite database file
        :param pool_size: Size of the connection pool
        :param enable_wal: Whether to enable Write-Ahead Logging
        """
        self.db_path = db_path
        self.pool_size = pool_size
        self.enable_wal = enable_wal
        self.logger = logging.getLogger(__name__)
        self.connection_pool = Queue(maxsize=pool_size)
        self.lock = Lock()
        self._initialize_pool()
        # sqlite3.sqlite3_config(sqlite3.SQLITE_CONFIG_SERIALIZED);

    def _initialize_pool(self):
        """Initialize the connection pool."""
        for _ in range(self.pool_size):
            conn = self._create_connection()
            self.connection_pool.put(conn)

    def _create_connection(self, busy_timeout: int = 5000):
        """Create a new database connection."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        # conn.execute(f"PRAGMA busy_timeout = {busy_timeout};")
        conn.row_factory = sqlite3.Row
        if self.enable_wal:
            conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    @contextmanager
    def get_connection(self, timeout: int = 10):
        """
        Context manager for getting a connection from the pool, with timeout handling.

        :param timeout: Time in seconds to wait for a connection before raising an error.
        """
        connection = None
        try:
            connection = self.connection_pool.get(timeout=timeout)
            yield connection
        finally:
            if connection:
                self.connection_pool.put(connection)

    def execute_query(self, query: str, params: Tuple = ()) -> List[sqlite3.Row]:
        """Execute a SELECT query and return the results."""
        self.logger.info(f"Executing query: {query} with params: {params}")
        with self.get_connection() as conn:
            try:
                cur = conn.cursor()
                cur.execute(query, params)
                return cur.fetchall()
            except sqlite3.Error as e:
                self.logger.error(f"Query execution error: {e}")
                raise

    def execute_write_query(self, query: str, params: Tuple = ()) -> int:
        """Execute an INSERT, UPDATE, or DELETE query."""
        self.logger.info(f"Executing write query: {query} with params: {params}")
        with self.get_connection() as conn:
            try:
                cur = conn.cursor()
                cur.execute(query, params)
                conn.commit()
                return cur.rowcount
            except sqlite3.Error as e:
                conn.rollback()
                self.logger.error(f"Write query execution error: {e}")
                raise

    def create_table(self, table_name: str, columns: List[str], constraints: List[str] = None) -> bool:
        """Create a new table in the database."""
        self._validate_identifier(table_name)
        for column in columns:
            self._validate_identifier(column.split()[0])

        query = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(columns)}"
        if constraints:
            query += f", {', '.join(constraints)}"
        query += ")"

        try:
            self.execute_write_query(query)
            return True
        except sqlite3.Error:
            return False

    def truncate_table(self, table_name: str) -> bool:
        self._validate_identifier(table_name)
        query = f"DELETE FROM {table_name}"
        try:
            self.execute_write_query(query)
            return True
        except sqlite3.Error:
            return False

    def insert(self, table_name: str, data: dict) -> int:
        """Insert a single row into a table."""
        self._validate_identifier(table_name)
        for key in data.keys():
            self._validate_identifier(key)

        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?' for _ in data])
        query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"

        return self.execute_write_query(query, tuple(data.values()))

    def insert_safe(self, table_name: str, data: dict, check_columns: List[str] | None = None) -> int:
        """Insert a single row into a table."""
        self._validate_identifier(table_name)
        for key in data.keys():
            self._validate_identifier(key)

        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?' for _ in data])
        if check_columns and len(check_columns) > 0:
            conflict_columns = ', '.join(check_columns)
            query = (f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders}) " +
                     f"ON CONFLICT({conflict_columns}) DO NOTHING;")
        else:
            query = (f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders}) " +
                     f"ON CONFLICT DO NOTHING;")
        return self.execute_write_query(query, tuple(data.values()))

    def bulk_insert(self, table_name: str, data: List[dict], batch_size: int = 1000) -> int:
        """Insert multiple rows into a table using batch processing."""
        if not data:
            return 0

        self._validate_identifier(table_name)
        for key in data[0].keys():
            self._validate_identifier(key)

        columns = ', '.join(data[0].keys())
        placeholders = ', '.join(['?' for _ in data[0]])
        query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"

        total_rows_inserted = 0
        with self.get_connection() as conn:
            try:
                cur = conn.cursor()
                for i in range(0, len(data), batch_size):
                    batch_data = data[i:i + batch_size]
                    cur.executemany(query, [tuple(row.values()) for row in batch_data])
                    total_rows_inserted += cur.rowcount
                conn.commit()
                return total_rows_inserted
            except sqlite3.Error as e:
                conn.rollback()
                self.logger.error(f"Bulk insert error: {e}")
                raise

    def update(self, table_name: str, data: dict, condition: str, condition_params: Tuple) -> int:
        """Update rows in a table."""
        self._validate_identifier(table_name)
        for key in data.keys():
            self._validate_identifier(key)

        set_clause = ', '.join([f"{k} = ?" for k in data.keys()])
        query = f"UPDATE {table_name} SET {set_clause} WHERE {condition}"
        params = tuple(data.values()) + condition_params

        return self.execute_write_query(query, params)

    def delete(self, table_name: str, condition: str, params: Tuple) -> int:
        """Delete rows from a table."""
        self._validate_identifier(table_name)
        query = f"DELETE FROM {table_name} WHERE {condition}"
        return self.execute_write_query(query, params)

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database."""
        self._validate_identifier(table_name)
        query = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
        result = self.execute_query(query, (table_name,))
        return len(result) > 0

    def get_table_info(self, table_name: str) -> List[sqlite3.Row]:
        """Get information about table columns."""
        self._validate_identifier(table_name)
        query = f"PRAGMA table_info({table_name})"
        return self.execute_query(query)

    def backup_database(self, backup_path: str) -> bool:
        """Create a backup of the database."""
        try:
            with self.get_connection() as conn:
                backup = sqlite3.connect(backup_path)
                conn.backup(backup)
                backup.close()
            return True
        except sqlite3.Error as e:
            self.logger.error(f"Backup error: {e}")
            return False

    def vacuum(self) -> bool:
        """Rebuild the database file, repacking it into a minimal amount of disk space."""
        try:
            with self.get_connection() as conn:
                conn.execute("VACUUM")
            return True
        except sqlite3.Error as e:
            self.logger.error(f"Vacuum error: {e}")
            return False

    def upsert(self, table_name: str, data: dict, conflict_columns: List[str]) -> int:
        """Perform an UPSERT operation (INSERT or UPDATE on conflict)."""
        self._validate_identifier(table_name)
        for key in data.keys():
            self._validate_identifier(key)
        for column in conflict_columns:
            self._validate_identifier(column)

        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?' for _ in data])
        conflict_clause = ', '.join(conflict_columns)
        update_clause = ', '.join([f"{k} = excluded.{k}" for k in data.keys() if k not in conflict_columns])

        query = f"""
        INSERT INTO {table_name} ({columns})
        VALUES ({placeholders})
        ON CONFLICT({conflict_clause})
        DO UPDATE SET {update_clause}
        """

        return self.execute_write_query(query, tuple(data.values()))

    def create_index(self, table_name: str, column_names: List[str], index_name: Optional[str] = None) -> bool:
        """Create an index on specified columns."""
        self._validate_identifier(table_name)
        for column in column_names:
            self._validate_identifier(column)
        if index_name:
            self._validate_identifier(index_name)

        if not index_name:
            index_name = f"idx_{table_name}_{'_'.join(column_names)}"

        columns = ', '.join(column_names)
        query = f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} ({columns})"

        try:
            self.execute_write_query(query)
            return True
        except sqlite3.Error:
            return False

    def json_extract(self, column: str, json_path: str) -> str:
        """Use SQLite's JSON1 extension to extract values from JSON columns."""
        self._validate_identifier(column)
        query = f"json_extract({column}, '{json_path}')"
        return query

    @staticmethod
    def _validate_identifier(identifier: str):
        """
        Validate if the provided identifier is a valid SQLite identifier, and prevent SQL injection.

        :param identifier: The identifier to validate.
        """
        if not identifier.isidentifier() or identifier.upper() in ('SELECT', 'INSERT', 'DELETE', 'UPDATE', 'DROP'):
            raise ValueError(f"Invalid identifier: {identifier}")

    @contextmanager
    def transaction(self):
        """Context manager for handling database transactions."""
        with self.get_connection() as conn:
            try:
                conn.execute("BEGIN")
                yield
                conn.commit()
            except sqlite3.Error as e:
                conn.rollback()
                self.logger.error(f"Transaction failed: {e}")
                raise

    # CSV handling functions (import/export) with improved error handling

    def import_from_csv(self, table_name: str, csv_data: str) -> int:
        """Import data from a CSV string into a table."""
        self._validate_identifier(table_name)

        reader = csv.DictReader(io.StringIO(csv_data))
        try:
            rows: List[dict] = [row for row in reader]
            return self.bulk_insert(table_name, rows)
        except csv.Error as e:
            self.logger.error(f"CSV import error: {e}")
            raise

    def export_to_csv(self, query: str, params: Tuple = ()) -> str:
        """Export the result of a query to CSV format."""
        with self.get_connection() as conn:
            try:
                cur = conn.cursor()
                cur.execute(query, params)
                rows = cur.fetchall()
                output = io.StringIO()
                writer = csv.writer(output)
                writer.writerow([desc[0] for desc in cur.description])  # Write header
                writer.writerows(rows)
                return output.getvalue()
            except sqlite3.Error as e:
                self.logger.error(f"CSV export error: {e}")
                raise
