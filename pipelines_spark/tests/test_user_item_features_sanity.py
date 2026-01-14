import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


@pytest.fixture(scope="session")
def spark():
    spark = (
        SparkSession.builder
        .appName("hm_test_user_item_features_sanity")
        .master("local[*]")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.ui.enabled", "false")
        .config("spark.port.maxRetries", "32")
        .getOrCreate()
    )
    yield spark
    spark.stop()


def test_user_features_unique_by_customer_id(spark):
    df = spark.read.parquet("data/feature_store/user_features.parquet")
    dup = df.groupBy("customer_id").count().filter(F.col("count") > 1).count()
    assert dup == 0
    nulls = df.filter(F.col("customer_id").isNull()).count()
    assert nulls == 0


def test_item_features_unique_by_article_id(spark):
    df = spark.read.parquet("data/feature_store/item_features.parquet")
    dup = df.groupBy("article_id").count().filter(F.col("count") > 1).count()
    assert dup == 0
    nulls = df.filter(F.col("article_id").isNull()).count()
    assert nulls == 0
