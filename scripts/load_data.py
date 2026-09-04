"""Load the Olist Brazilian e-commerce dataset into DuckDB at data/canon.duckdb.

Idempotent by design, in two independent layers:

1.  **Download.** A CSV is re-fetched only when it is absent or its size does not match
    the mirror's ``Content-Length``. Downloads land in a ``.part`` file and are renamed
    atomically, so an interrupted run never leaves a half file that looks complete.
2.  **Load.** Every table is ``CREATE OR REPLACE``d from the CSV, and the SHA-256 of each
    source file is recorded in ``_canon_load_manifest``. A re-run whose hashes all match
    the manifest exits without touching the tables unless ``--force`` is passed.

The dataset lives on Kaggle, which requires credentials. We read it from a HuggingFace
mirror instead and verify what arrives: the header of every CSV is compared against the
column spec below, so a mirror that has silently altered, reordered, or renamed a column
fails the load rather than producing a warehouse that is subtly wrong.

Column types are declared, never sniffed. Money is ``DECIMAL(12, 2)`` because
``docs/01-evaluation.md`` compares money as DECIMAL, and zip prefixes are ``VARCHAR``
because they have leading zeros that an integer cast would eat.

Run:  uv run python scripts/load_data.py [--force] [--skip-geolocation]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Final

import duckdb
from pydantic import BaseModel, ConfigDict

REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
RAW_DIR: Final[Path] = REPO_ROOT / "data" / "raw"
DB_PATH: Final[Path] = REPO_ROOT / "data" / "canon.duckdb"

# Primary mirror, then a fallback with an identical file set. Both are read-only public
# HuggingFace dataset repos serving the unmodified Kaggle CSVs.
MIRRORS: Final[tuple[str, ...]] = (
    "https://huggingface.co/datasets/bulutttt/olist-raw-data/resolve/main",
    "https://huggingface.co/datasets/aviahYadler/Olist_Ecommerce_Dataset/resolve/main",
)

MANIFEST_TABLE: Final[str] = "_canon_load_manifest"


class TableSpec(BaseModel):
    """How one CSV becomes one DuckDB table. A bare dict would not survive review."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    table: str
    """Table name in DuckDB. The `olist_` prefix and `_dataset` suffix are dropped."""

    csv: str
    """File name on the mirror."""

    columns: tuple[tuple[str, str], ...]
    """(column name, DuckDB type), in file order. Declared, not sniffed."""

    published_rows: int
    """Row count published for the Kaggle original.

    Used *only* as a mirror-integrity check -- a mismatch means the mirror is not the
    dataset we think it is. It is never reported as a measured figure; the numbers this
    script prints are counted from the files that actually landed on disk.
    """

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.columns)


# Note: `product_name_lenght` / `product_description_lenght` are misspelled in the
# source data. They are kept verbatim -- renaming them here would mean the warehouse no
# longer matches the published dataset, and the semantic layer is the right place to
# give columns readable names.
SPECS: Final[tuple[TableSpec, ...]] = (
    TableSpec(
        table="orders",
        csv="olist_orders_dataset.csv",
        columns=(
            ("order_id", "VARCHAR"),
            ("customer_id", "VARCHAR"),
            ("order_status", "VARCHAR"),
            ("order_purchase_timestamp", "TIMESTAMP"),
            ("order_approved_at", "TIMESTAMP"),
            ("order_delivered_carrier_date", "TIMESTAMP"),
            ("order_delivered_customer_date", "TIMESTAMP"),
            ("order_estimated_delivery_date", "TIMESTAMP"),
        ),
        published_rows=99_441,
    ),
    TableSpec(
        table="order_items",
        csv="olist_order_items_dataset.csv",
        columns=(
            ("order_id", "VARCHAR"),
            ("order_item_id", "INTEGER"),
            ("product_id", "VARCHAR"),
            ("seller_id", "VARCHAR"),
            ("shipping_limit_date", "TIMESTAMP"),
            ("price", "DECIMAL(12, 2)"),
            ("freight_value", "DECIMAL(12, 2)"),
        ),
        published_rows=112_650,
    ),
    TableSpec(
        table="order_payments",
        csv="olist_order_payments_dataset.csv",
        columns=(
            ("order_id", "VARCHAR"),
            ("payment_sequential", "INTEGER"),
            ("payment_type", "VARCHAR"),
            ("payment_installments", "INTEGER"),
            ("payment_value", "DECIMAL(12, 2)"),
        ),
        published_rows=103_886,
    ),
    TableSpec(
        table="order_reviews",
        csv="olist_order_reviews_dataset.csv",
        columns=(
            ("review_id", "VARCHAR"),
            ("order_id", "VARCHAR"),
            ("review_score", "INTEGER"),
            ("review_comment_title", "VARCHAR"),
            ("review_comment_message", "VARCHAR"),
            ("review_creation_date", "TIMESTAMP"),
            ("review_answer_timestamp", "TIMESTAMP"),
        ),
        published_rows=99_224,
    ),
    TableSpec(
        table="customers",
        csv="olist_customers_dataset.csv",
        columns=(
            ("customer_id", "VARCHAR"),
            ("customer_unique_id", "VARCHAR"),
            ("customer_zip_code_prefix", "VARCHAR"),
            ("customer_city", "VARCHAR"),
            ("customer_state", "VARCHAR"),
        ),
        published_rows=99_441,
    ),
    TableSpec(
        table="products",
        csv="olist_products_dataset.csv",
        columns=(
            ("product_id", "VARCHAR"),
            ("product_category_name", "VARCHAR"),
            ("product_name_lenght", "INTEGER"),
            ("product_description_lenght", "INTEGER"),
            ("product_photos_qty", "INTEGER"),
            ("product_weight_g", "INTEGER"),
            ("product_length_cm", "INTEGER"),
            ("product_height_cm", "INTEGER"),
            ("product_width_cm", "INTEGER"),
        ),
        published_rows=32_951,
    ),
    TableSpec(
        table="sellers",
        csv="olist_sellers_dataset.csv",
        columns=(
            ("seller_id", "VARCHAR"),
            ("seller_zip_code_prefix", "VARCHAR"),
            ("seller_city", "VARCHAR"),
            ("seller_state", "VARCHAR"),
        ),
        published_rows=3_095,
    ),
    TableSpec(
        table="geolocation",
        csv="olist_geolocation_dataset.csv",
        columns=(
            ("geolocation_zip_code_prefix", "VARCHAR"),
            ("geolocation_lat", "DOUBLE"),
            ("geolocation_lng", "DOUBLE"),
            ("geolocation_city", "VARCHAR"),
            ("geolocation_state", "VARCHAR"),
        ),
        published_rows=1_000_163,
    ),
    TableSpec(
        table="product_category_translation",
        csv="product_category_name_translation.csv",
        columns=(
            ("product_category_name", "VARCHAR"),
            ("product_category_name_english", "VARCHAR"),
        ),
        published_rows=71,
    ),
)


class LoadError(RuntimeError):
    """Raised when the data on disk is not the dataset this project expects."""


def _remote_size(url: str) -> int | None:
    """Content-Length for a mirror URL, or None if the mirror will not say."""
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "canon-loader"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:  # https only
            length = response.headers.get("Content-Length")
    except (urllib.error.URLError, TimeoutError):
        return None
    return int(length) if length is not None else None


def _download(csv_name: str, destination: Path) -> None:
    """Fetch one CSV, trying each mirror in turn. Atomic: writes .part, then renames."""
    part = destination.with_suffix(destination.suffix + ".part")
    last_error: Exception | None = None

    for base in MIRRORS:
        url = f"{base}/{csv_name}"
        expected = _remote_size(url)

        if destination.exists() and expected is not None and destination.stat().st_size == expected:
            print(f"  {csv_name}: present and complete, skipping download")
            return

        try:
            request = urllib.request.Request(url, headers={"User-Agent": "canon-loader"})
            with (
                urllib.request.urlopen(request, timeout=600) as response,  # https only
                part.open("wb") as handle,
            ):
                downloaded = 0
                while chunk := response.read(1 << 20):
                    handle.write(chunk)
                    downloaded += len(chunk)
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last_error = error
            part.unlink(missing_ok=True)
            print(f"  {csv_name}: mirror failed ({error}); trying next")
            continue

        if expected is not None and downloaded != expected:
            part.unlink(missing_ok=True)
            last_error = LoadError(f"{csv_name}: got {downloaded} bytes, mirror said {expected}")
            continue

        part.replace(destination)
        print(f"  {csv_name}: downloaded {downloaded:,} bytes")
        return

    raise LoadError(f"could not fetch {csv_name} from any mirror") from last_error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _check_header(spec: TableSpec, path: Path) -> None:
    """Fail if the mirror's columns are not exactly what we declared, in order.

    This is the check that turns "a mirror quietly changed the data" from a silent
    wrong-numbers bug into a loud startup failure.
    """
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header_line = handle.readline().strip()

    found = tuple(field.strip().strip('"').strip() for field in header_line.split(","))
    if found != spec.column_names:
        raise LoadError(
            f"{spec.csv}: header does not match the declared spec.\n"
            f"  expected: {spec.column_names}\n"
            f"  found:    {found}\n"
            "The mirror is not serving the dataset this project was written against."
        )


def _read_manifest(connection: duckdb.DuckDBPyConnection) -> dict[str, str]:
    tables = {row[0] for row in connection.execute("SHOW TABLES").fetchall()}
    if MANIFEST_TABLE not in tables:
        return {}
    rows = connection.execute(f"SELECT csv, sha256 FROM {MANIFEST_TABLE}").fetchall()
    return {str(csv): str(sha) for csv, sha in rows}


def _load_table(connection: duckdb.DuckDBPyConnection, spec: TableSpec, path: Path) -> int:
    column_struct = ", ".join(f"'{name}': '{dtype}'" for name, dtype in spec.columns)
    # The CSV path is a parameter, not interpolated; only our own declared identifiers
    # and types reach the SQL text.
    connection.execute(
        f"CREATE OR REPLACE TABLE {spec.table} AS "  # identifiers are ours, not input
        "SELECT * FROM read_csv(?, header = true, columns = {" + column_struct + "})",
        [str(path)],
    )
    (count,) = connection.execute(f"SELECT count(*) FROM {spec.table}").fetchone()  # type: ignore[misc]
    return int(count)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="reload every table even if hashes are unchanged"
    )
    parser.add_argument(
        "--skip-geolocation",
        action="store_true",
        help="skip the ~1M-row lat/long lookup; no canonical metric depends on it",
    )
    args = parser.parse_args()

    specs = tuple(
        spec for spec in SPECS if not (args.skip_geolocation and spec.table == "geolocation")
    )

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"raw CSVs -> {RAW_DIR}")
    for spec in specs:
        _download(spec.csv, RAW_DIR / spec.csv)

    print("\nverifying headers against the declared spec")
    for spec in specs:
        _check_header(spec, RAW_DIR / spec.csv)
    print(f"  {len(specs)} headers match")

    hashes = {spec.csv: _sha256(RAW_DIR / spec.csv) for spec in specs}

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(DB_PATH))
    try:
        existing = _read_manifest(connection)
        tables = {row[0] for row in connection.execute("SHOW TABLES").fetchall()}
        unchanged = all(
            existing.get(spec.csv) == hashes[spec.csv] and spec.table in tables for spec in specs
        )
        if unchanged and not args.force:
            print(f"\n{DB_PATH.name} is already current for these CSVs; nothing to do.")
            print("(pass --force to reload anyway)")
            return 0

        print(f"\nloading -> {DB_PATH}")
        counts: dict[str, int] = {}
        drift: list[str] = []
        for spec in specs:
            counts[spec.table] = _load_table(connection, spec, RAW_DIR / spec.csv)
            if counts[spec.table] != spec.published_rows:
                drift.append(
                    f"{spec.table}: loaded {counts[spec.table]:,}, "
                    f"published shape is {spec.published_rows:,}"
                )

        connection.execute(
            f"CREATE OR REPLACE TABLE {MANIFEST_TABLE} "
            "(csv VARCHAR, sha256 VARCHAR, table_name VARCHAR, row_count BIGINT, "
            "loaded_at TIMESTAMP)"
        )
        connection.executemany(
            f"INSERT INTO {MANIFEST_TABLE} VALUES (?, ?, ?, ?, current_timestamp)",
            [[spec.csv, hashes[spec.csv], spec.table, counts[spec.table]] for spec in specs],
        )

        width = max(len(spec.table) for spec in specs)
        print("\nrow counts, counted from the files on disk:")
        for spec in specs:
            print(f"  {spec.table:<{width}}  {counts[spec.table]:>10,}")

        if drift:
            print("\nWARNING: row counts differ from the published Kaggle shape:")
            for line in drift:
                print(f"  {line}")
            print("  The mirror may not be the unmodified dataset. Investigate before trusting")
            print("  any number this warehouse produces.")
        else:
            print("\nall tables match the published Kaggle row counts")

        first, last = connection.execute(
            "SELECT min(order_purchase_timestamp), max(order_purchase_timestamp) FROM orders"
        ).fetchone()  # type: ignore[misc]
        print(f"orders span {first} .. {last}")
        print(f"\nmanifest written to {MANIFEST_TABLE}. Re-running is a no-op.")
    finally:
        connection.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
