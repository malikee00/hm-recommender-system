from typing import List

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


def require_columns(df: DataFrame, required: List[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise AssertionError(f"[{name}] Missing required columns: {missing}")


def safe_select(df: DataFrame, cols: List[str]) -> DataFrame:
    existing = [c for c in cols if c in df.columns]
    return df.select(*existing)


def enforce_unique_key(df: DataFrame, key: str, name: str) -> DataFrame:
    dup = df.groupBy(key).count().filter(F.col("count") > 1).limit(1).count()
    if dup > 0:
        raise AssertionError(f"[{name}] Not unique by key '{key}'")
    return df


def drop_null_keys(df: DataFrame, keys: List[str]) -> DataFrame:
    cond = None
    for k in keys:
        kcond = F.col(k).isNotNull()
        cond = kcond if cond is None else (cond & kcond)
    return df.filter(cond) if cond is not None else df


def top1_category_per_user(
    interactions: DataFrame,
    articles: DataFrame,
    user_key: str = "customer_id",
    item_key: str = "article_id",
    category_col: str = "product_group_name",
) -> DataFrame:
    """
    Return per-user top-1 category based on purchase counts.
    Output cols: customer_id, top_product_group_name
    """
    joined = interactions.join(
        articles.select(item_key, category_col),
        on=item_key,
        how="left",
    )

    # If some items don't match articles, category can be null; keep but handle later.
    agg = (
        joined.groupBy(user_key, category_col)
        .agg(F.count(F.lit(1)).alias("cnt"))
    )

    w = Window.partitionBy(user_key).orderBy(F.col("cnt").desc(), F.col(category_col).asc_nulls_last())
    ranked = agg.withColumn("rn", F.row_number().over(w)).filter(F.col("rn") == 1)

    return ranked.select(
        F.col(user_key),
        F.col(category_col).alias("top_product_group_name"),
    )


def popularity_rank(interactions: DataFrame, item_key: str = "article_id") -> DataFrame:
    pop = interactions.groupBy(item_key).agg(F.count(F.lit(1)).alias("purchase_count"))
    w = Window.orderBy(F.col("purchase_count").desc(), F.col(item_key).asc())
    return pop.withColumn("popularity_rank", F.dense_rank().over(w))
