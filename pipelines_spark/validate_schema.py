
import argparse
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F


# -----------------------------
# Config / thresholds
# -----------------------------
REQUIRED_CUSTOMERS_COLS = ["customer_id"]
REQUIRED_ARTICLES_COLS = ["article_id"]
REQUIRED_TRANSACTIONS_COLS = ["t_dat", "customer_id", "article_id", "price", "sales_channel_id"]


@dataclass(frozen=True)
class CoverageThresholds:
    """
    Join coverage thresholds. You can tune these if needed.
    For H&M Kaggle data, coverage should be near 1.0.
    """
    tx_to_customers_min: float = 0.98
    tx_to_articles_min: float = 0.98


# -----------------------------
# Helpers
# -----------------------------
def _get_logger() -> logging.Logger:
    logger = logging.getLogger("validate_schema")
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("[%(levelname)s] %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def create_spark(app_name: str = "hm_validate_schema") -> SparkSession:
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


def require_columns(df: DataFrame, required: List[str], table_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise AssertionError(f"[{table_name}] Missing required columns: {missing}")


def non_null_rate(df: DataFrame, col_name: str) -> float:
    total = df.count()
    if total == 0:
        return 0.0
    non_null = df.filter(F.col(col_name).isNotNull()).count()
    return non_null / total


def validate_transactions_quality(tx: DataFrame) -> Dict[str, float]:
    for c in ["customer_id", "article_id", "t_dat"]:
        null_cnt = tx.filter(F.col(c).isNull()).count()
        if null_cnt > 0:
            raise AssertionError(f"[transactions_train] Found {null_cnt} null values in key column '{c}'")

    # price >= 0
    neg_price_cnt = tx.filter(F.col("price") < F.lit(0)).count()
    if neg_price_cnt > 0:
        raise AssertionError(f"[transactions_train] Found {neg_price_cnt} rows with price < 0")

    tx2 = tx.withColumn("t_dat_parsed", F.to_date(F.col("t_dat")))
    bad_date_cnt = tx2.filter(F.col("t_dat").isNotNull() & F.col("t_dat_parsed").isNull()).count()
    if bad_date_cnt > 0:
        raise AssertionError(f"[transactions_train] Found {bad_date_cnt} rows with non-parseable t_dat")

    total = tx.count()
    return {
        "transactions_rows": float(total),
        "transactions_bad_date_rows": float(bad_date_cnt),
        "transactions_negative_price_rows": float(neg_price_cnt),
    }


def compute_join_coverage(
    tx: DataFrame,
    customers: DataFrame,
    articles: DataFrame,
) -> Tuple[float, float]:
    tx_count = tx.count()
    if tx_count == 0:
        raise AssertionError("[transactions_train] Table has 0 rows; cannot validate join coverage.")

    # Distinct keys in dimension tables
    cust_keys = customers.select("customer_id").dropna().dropDuplicates()
    art_keys = articles.select("article_id").dropna().dropDuplicates()

    # Coverage to customers
    tx_cust_matched = (
        tx.select("customer_id")
        .dropna()
        .join(cust_keys, on="customer_id", how="inner")
        .count()
    )
    tx_customer_coverage = tx_cust_matched / tx_count

    # Coverage to articles
    tx_art_matched = (
        tx.select("article_id")
        .dropna()
        .join(art_keys, on="article_id", how="inner")
        .count()
    )
    tx_article_coverage = tx_art_matched / tx_count

    return tx_customer_coverage, tx_article_coverage


def validate_schema_and_quality(
    spark: SparkSession,
    raw_dir: str,
    thresholds: CoverageThresholds = CoverageThresholds(),
) -> Dict[str, float]:
    logger = _get_logger()

    customers_path = os.path.join(raw_dir, "customers.csv")
    articles_path = os.path.join(raw_dir, "articles.csv")
    transactions_path = os.path.join(raw_dir, "transactions_train.csv")

    # Ensure files exist
    assert_file_exists(customers_path)
    assert_file_exists(articles_path)
    assert_file_exists(transactions_path)

    customers = read_csv(spark, customers_path)
    articles = read_csv(spark, articles_path)
    tx = read_csv(spark, transactions_path)

    # Required columns
    require_columns(customers, REQUIRED_CUSTOMERS_COLS, "customers")
    require_columns(articles, REQUIRED_ARTICLES_COLS, "articles")
    require_columns(tx, REQUIRED_TRANSACTIONS_COLS, "transactions_train")

    # Basic quality checks (hard fails)
    stats = {}
    stats.update(validate_transactions_quality(tx))

    # Join coverage checks
    cust_cov, art_cov = compute_join_coverage(tx, customers, articles)
    stats["tx_to_customers_coverage"] = float(cust_cov)
    stats["tx_to_articles_coverage"] = float(art_cov)

    logger.info(f"Join coverage transactions -> customers: {cust_cov:.4f}")
    logger.info(f"Join coverage transactions -> articles : {art_cov:.4f}")

    if cust_cov < thresholds.tx_to_customers_min:
        raise AssertionError(
            f"Join coverage too low (tx->customers): {cust_cov:.4f} < {thresholds.tx_to_customers_min:.2f}"
        )
    if art_cov < thresholds.tx_to_articles_min:
        raise AssertionError(
            f"Join coverage too low (tx->articles): {art_cov:.4f} < {thresholds.tx_to_articles_min:.2f}"
        )

    logger.info("[DONE] Schema + quality validation passed.")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="H&M schema validation (Phase 2.2)")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw/hm/reference",
        help="Path to raw H&M CSV folder (default: data/raw/hm/reference)",
    )
    parser.add_argument(
        "--tx-to-customers-min",
        type=float,
        default=CoverageThresholds().tx_to_customers_min,
        help="Minimum join coverage from transactions to customers (default: 0.98)",
    )
    parser.add_argument(
        "--tx-to-articles-min",
        type=float,
        default=CoverageThresholds().tx_to_articles_min,
        help="Minimum join coverage from transactions to articles (default: 0.98)",
    )
    args = parser.parse_args()

    logger = _get_logger()
    logger.info(f"Raw dir: {args.raw_dir}")

    spark = create_spark()

    try:
        thresholds = CoverageThresholds(
            tx_to_customers_min=float(args.tx_to_customers_min),
            tx_to_articles_min=float(args.tx_to_articles_min),
        )
        validate_schema_and_quality(spark=spark, raw_dir=args.raw_dir, thresholds=thresholds)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
