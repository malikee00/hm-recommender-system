import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


@pytest.fixture(scope="session")
def spark():
    spark = (
        SparkSession.builder
        .appName("hm_test_interactions_sanity")
        .master("local[*]")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.ui.enabled", "false")
        .config("spark.port.maxRetries", "32")
        .getOrCreate()
    )
    yield spark
    spark.stop()


def test_interactions_no_null_keys(spark):
    df = spark.read.parquet("data/feature_store/interactions.parquet")
    null_keys = df.filter(
        F.col("customer_id").isNull() |
        F.col("article_id").isNull() |
        F.col("t_dat").isNull()
    ).count()
    assert null_keys == 0


def test_interactions_price_non_negative(spark):
    df = spark.read.parquet("data/feature_store/interactions.parquet")
    bad = df.filter(F.col("price").isNotNull() & (F.col("price") < 0)).count()
    assert bad == 0


def test_interactions_date_valid_type(spark):
    df = spark.read.parquet("data/feature_store/interactions.parquet")
    # ensure t_dat is date and not null (null already checked)
    # if t_dat isn't DateType, to_date will produce nulls -> count would be >0
    bad = df.withColumn("t_dat_parsed", F.to_date(F.col("t_dat"))).filter(F.col("t_dat_parsed").isNull()).count()
    assert bad == 0


def test_interactions_strength_sane(spark):
    """
    Strength = count of interactions per user/item. Minimal sanity: max >= 1 and no negative counts (impossible).
    """
    df = spark.read.parquet("data/feature_store/interactions.parquet")
    user_cnt = df.groupBy("customer_id").count()
    assert user_cnt.agg(F.min("count")).first()[0] >= 1

    item_cnt = df.groupBy("article_id").count()
    assert item_cnt.agg(F.min("count")).first()[0] >= 1
