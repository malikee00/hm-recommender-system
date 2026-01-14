import argparse
import logging
import os
from dataclasses import dataclass
from typing import List, Tuple

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T


# -----------------------------
# Logging
# -----------------------------
def _get_logger() -> logging.Logger:
    logger = logging.getLogger("transform")
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("[%(levelname)s] %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


# -----------------------------
# Spark
# -----------------------------
def create_spark(app_name: str = "hm_transform") -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.ui.enabled", "false")
        .config("spark.port.maxRetries", "32")
        .getOrCreate()
    )


# -----------------------------
# IO helpers
# -----------------------------
def assert_file_exists(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")


def read_csv(spark: SparkSession, path: str) -> DataFrame:
    return (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(path)
    )


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_parquet(df: DataFrame, out_path: str, mode: str = "overwrite") -> None:
    (df.write.mode(mode).parquet(out_path))


# -----------------------------
# Transform helpers
# -----------------------------
def standardize_columns(df: DataFrame) -> DataFrame:
    """
    Lowercase + strip spaces, replace spaces with underscore.
    """
    renamed = df
    for c in df.columns:
        new_c = c.strip().lower().replace(" ", "_")
        if new_c != c:
            renamed = renamed.withColumnRenamed(c, new_c)
    return renamed


def drop_null_keys(df: DataFrame, keys: List[str]) -> DataFrame:
    cond = None
    for k in keys:
        kcond = F.col(k).isNotNull()
        cond = kcond if cond is None else (cond & kcond)
    return df.filter(cond) if cond is not None else df


def dedup_by_key(df: DataFrame, key: str) -> DataFrame:
    """
    Keep 1 row per key (first occurrence). For canonical tables, this is enough.
    """
    return df.dropDuplicates([key])


def cast_transactions_types(tx: DataFrame) -> DataFrame:
    tx2 = tx

    # cast columns (safe)
    tx2 = tx2.withColumn("t_dat", F.to_date(F.col("t_dat")))
    tx2 = tx2.withColumn("price", F.col("price").cast(T.DoubleType()))
    tx2 = tx2.withColumn("sales_channel_id", F.col("sales_channel_id").cast(T.IntegerType()))

    # drop null keys
    tx2 = drop_null_keys(tx2, ["customer_id", "article_id", "t_dat"])

    # filter bad price
    tx2 = tx2.filter(F.col("price").isNull() | (F.col("price") >= F.lit(0.0)))
    # if "price must be non-null", uncomment:
    # tx2 = tx2.filter(F.col("price").isNotNull() & (F.col("price") >= F.lit(0.0)))

    return tx2


@dataclass(frozen=True)
class TransformOutputs:
    customers_out: str
    articles_out: str
    transactions_out: str


def transform_and_write(
    spark: SparkSession,
    raw_dir: str,
    processed_dir: str,
    mode: str = "overwrite",
) -> TransformOutputs:
    logger = _get_logger()

    customers_path = os.path.join(raw_dir, "customers.csv")
    articles_path = os.path.join(raw_dir, "articles.csv")
    transactions_path = os.path.join(raw_dir, "transactions_train.csv")

    assert_file_exists(customers_path)
    assert_file_exists(articles_path)
    assert_file_exists(transactions_path)

    ensure_dir(processed_dir)

    # Read
    customers = read_csv(spark, customers_path)
    articles = read_csv(spark, articles_path)
    tx = read_csv(spark, transactions_path)

    # Standardize columns
    customers = standardize_columns(customers)
    articles = standardize_columns(articles)
    tx = standardize_columns(tx)

    # customers: dedup + drop null key
    customers = drop_null_keys(customers, ["customer_id"])
    customers = dedup_by_key(customers, "customer_id")

    # articles: dedup + drop null key
    articles = drop_null_keys(articles, ["article_id"])
    articles = dedup_by_key(articles, "article_id")

    # transactions: cast + filter
    tx = cast_transactions_types(tx)

    canonical_tx_cols = ["t_dat", "customer_id", "article_id", "price", "sales_channel_id"]
    existing_tx_cols = [c for c in canonical_tx_cols if c in tx.columns]
    tx = tx.select(*existing_tx_cols)

    # Output paths
    customers_out = os.path.join(processed_dir, "customers_clean.parquet")
    articles_out = os.path.join(processed_dir, "articles_clean.parquet")
    transactions_out = os.path.join(processed_dir, "transactions_clean.parquet")

    # Write
    logger.info(f"Writing customers -> {customers_out}")
    write_parquet(customers, customers_out, mode=mode)

    logger.info(f"Writing articles  -> {articles_out}")
    write_parquet(articles, articles_out, mode=mode)

    logger.info(f"Writing tx        -> {transactions_out}")
    write_parquet(tx, transactions_out, mode=mode)

    # Quick sanity logs
    logger.info(f"customers_clean rows: {customers.count():,}")
    logger.info(f"articles_clean rows : {articles.count():,}")
    logger.info(f"transactions_clean rows: {tx.count():,}")

    logger.info("[DONE] Transform layer completed (canonical parquet written).")

    return TransformOutputs(
        customers_out=customers_out,
        articles_out=articles_out,
        transactions_out=transactions_out,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="H&M transform layer (Phase 2.3)")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw/hm/reference",
        help="Raw CSV folder (default: data/raw/hm/reference)",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="data/processed",
        help="Processed output folder for canonical parquet (default: data/processed)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="overwrite",
        choices=["overwrite", "errorifexists"],
        help="Write mode for parquet outputs (default: overwrite)",
    )
    args = parser.parse_args()

    logger = _get_logger()
    logger.info(f"Raw dir       : {args.raw_dir}")
    logger.info(f"Processed dir : {args.processed_dir}")
    logger.info(f"Write mode    : {args.mode}")

    spark = create_spark()

    try:
        transform_and_write(
            spark=spark,
            raw_dir=args.raw_dir,
            processed_dir=args.processed_dir,
            mode=args.mode,
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()