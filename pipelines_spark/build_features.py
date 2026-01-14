import argparse
import logging
import os

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from features.feature_spec import (
    INTERACTIONS_COLS,
    USER_FEATURES_COLS,
    ITEM_FEATURES_COLS,
    USER_HISTORY_AGG_COLS,
    ITEM_POPULARITY_COLS,
)
from features.helpers import (
    require_columns,
    safe_select,
    enforce_unique_key,
    drop_null_keys,
    top1_category_per_user,
    popularity_rank,
)


def _get_logger() -> logging.Logger:
    logger = logging.getLogger("build_features")
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("[%(levelname)s] %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def create_spark(app_name: str = "hm_build_features") -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.ui.enabled", "false")
        .config("spark.port.maxRetries", "32")
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
    )


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="H&M feature building (Phase 2.4)")
    parser.add_argument("--processed-dir", type=str, default="data/processed", help="Canonical parquet folder")
    parser.add_argument("--feature-store-dir", type=str, default="data/feature_store", help="Feature store output folder")
    parser.add_argument("--mode", type=str, default="overwrite", choices=["overwrite", "errorifexists"], help="Write mode")
    args = parser.parse_args()

    logger = _get_logger()
    logger.info(f"Processed dir     : {args.processed_dir}")
    logger.info(f"Feature store dir : {args.feature_store_dir}")
    logger.info(f"Write mode        : {args.mode}")

    customers_path = os.path.join(args.processed_dir, "customers_clean.parquet")
    articles_path = os.path.join(args.processed_dir, "articles_clean.parquet")
    tx_path = os.path.join(args.processed_dir, "transactions_clean.parquet")

    if not os.path.exists(customers_path):
        raise FileNotFoundError(f"Missing: {customers_path}")
    if not os.path.exists(articles_path):
        raise FileNotFoundError(f"Missing: {articles_path}")
    if not os.path.exists(tx_path):
        raise FileNotFoundError(f"Missing: {tx_path}")

    ensure_dir(args.feature_store_dir)

    spark = create_spark()
    try:
        customers = spark.read.parquet(customers_path)
        articles = spark.read.parquet(articles_path)
        tx = spark.read.parquet(tx_path)

        # -----------------------------
        # Output 1 — interactions.parquet
        # -----------------------------
        require_columns(tx, ["customer_id", "article_id", "t_dat", "price", "sales_channel_id"], "transactions_clean")

        interactions = safe_select(tx, INTERACTIONS_COLS)
        interactions = drop_null_keys(interactions, ["customer_id", "article_id", "t_dat"])
        interactions = interactions.filter(F.col("price").isNull() | (F.col("price") >= F.lit(0.0)))

        out_interactions = os.path.join(args.feature_store_dir, "interactions.parquet")
        logger.info(f"Writing interactions -> {out_interactions}")
        interactions.write.mode(args.mode).parquet(out_interactions)

        # -----------------------------
        # Output 2 — user_features.parquet
        # -----------------------------
        require_columns(customers, ["customer_id"], "customers_clean")

        user_features = safe_select(customers, USER_FEATURES_COLS)

        # normalize column name: in raw it's "active" maybe int/bool/str; keep as-is for now
        user_features = drop_null_keys(user_features, ["customer_id"])
        user_features = user_features.dropDuplicates(["customer_id"])
        enforce_unique_key(user_features, "customer_id", "user_features")

        out_user = os.path.join(args.feature_store_dir, "user_features.parquet")
        logger.info(f"Writing user_features -> {out_user}")
        user_features.write.mode(args.mode).parquet(out_user)

        # -----------------------------
        # Output 3 — item_features.parquet
        # -----------------------------
        require_columns(articles, ["article_id"], "articles_clean")

        item_features = safe_select(articles, ITEM_FEATURES_COLS)
        item_features = drop_null_keys(item_features, ["article_id"])
        item_features = item_features.dropDuplicates(["article_id"])
        enforce_unique_key(item_features, "article_id", "item_features")

        out_item = os.path.join(args.feature_store_dir, "item_features.parquet")
        logger.info(f"Writing item_features -> {out_item}")
        item_features.write.mode(args.mode).parquet(out_item)

        # -----------------------------
        # Output 4 — user_history_agg.parquet
        # -----------------------------
        # total_purchases, last_purchase_date, avg_price, top_product_group_name
        user_basic_agg = (
            interactions.groupBy("customer_id")
            .agg(
                F.count(F.lit(1)).alias("total_purchases"),
                F.max("t_dat").alias("last_purchase_date"),
                F.avg("price").alias("avg_price"),
            )
        )

        # top category from interactions + articles
        require_columns(articles, ["article_id", "product_group_name"], "articles_clean (for top category)")
        top_cat = top1_category_per_user(
            interactions=interactions,
            articles=articles.select("article_id", "product_group_name"),
            category_col="product_group_name",
        )

        user_history_agg = (
            user_basic_agg.join(top_cat, on="customer_id", how="left")
        )

        # select final columns
        user_history_agg = safe_select(user_history_agg, USER_HISTORY_AGG_COLS)

        out_user_hist = os.path.join(args.feature_store_dir, "user_history_agg.parquet")
        logger.info(f"Writing user_history_agg -> {out_user_hist}")
        user_history_agg.write.mode(args.mode).parquet(out_user_hist)

        # -----------------------------
        # Output 5 — item_popularity.parquet
        # -----------------------------
        item_pop = popularity_rank(interactions, item_key="article_id")
        item_pop = safe_select(item_pop, ITEM_POPULARITY_COLS)

        out_pop = os.path.join(args.feature_store_dir, "item_popularity.parquet")
        logger.info(f"Writing item_popularity -> {out_pop}")
        item_pop.write.mode(args.mode).parquet(out_pop)

        # -----------------------------
        # Quick log counts
        # -----------------------------
        logger.info(f"interactions rows     : {interactions.count():,}")
        logger.info(f"user_features rows    : {user_features.count():,}")
        logger.info(f"item_features rows    : {item_features.count():,}")
        logger.info(f"user_history_agg rows : {user_history_agg.count():,}")
        logger.info(f"item_popularity rows  : {item_pop.count():,}")

        logger.info("[DONE] Feature store tables written.")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
