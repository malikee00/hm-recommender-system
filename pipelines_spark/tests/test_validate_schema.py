import os
import tempfile

import pytest
from pyspark.sql import SparkSession

from pipelines_spark.validate_schema import (
    CoverageThresholds,
    validate_schema_and_quality,
)


@pytest.fixture(scope="session")
def spark():
    spark = (
        SparkSession.builder
        .appName("hm_validate_schema_tests")
        .master("local[*]")
        .getOrCreate()
    )
    yield spark
    spark.stop()


def _write_text(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def test_validate_schema_happy_path(spark):
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_dir = tmpdir

        _write_text(
            os.path.join(raw_dir, "customers.csv"),
            "customer_id,age\nc1,25\nc2,30\n",
        )
        _write_text(
            os.path.join(raw_dir, "articles.csv"),
            "article_id,product_type_name\n1001,T-shirt\n1002,Pants\n",
        )
        _write_text(
            os.path.join(raw_dir, "transactions_train.csv"),
            "t_dat,customer_id,article_id,price,sales_channel_id\n"
            "2020-09-01,c1,1001,10.0,1\n"
            "2020-09-02,c2,1002,20.0,2\n",
        )

        stats = validate_schema_and_quality(
            spark=spark,
            raw_dir=raw_dir,
            thresholds=CoverageThresholds(tx_to_customers_min=1.0, tx_to_articles_min=1.0),
        )
        assert stats["transactions_rows"] == 2.0
        assert stats["tx_to_customers_coverage"] == 1.0
        assert stats["tx_to_articles_coverage"] == 1.0


def test_validate_schema_fails_on_negative_price(spark):
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_dir = tmpdir

        _write_text(
            os.path.join(raw_dir, "customers.csv"),
            "customer_id\nc1\n",
        )
        _write_text(
            os.path.join(raw_dir, "articles.csv"),
            "article_id\n1001\n",
        )
        _write_text(
            os.path.join(raw_dir, "transactions_train.csv"),
            "t_dat,customer_id,article_id,price,sales_channel_id\n"
            "2020-09-01,c1,1001,-1.0,1\n",
        )

        with pytest.raises(AssertionError, match="price < 0"):
            validate_schema_and_quality(spark=spark, raw_dir=raw_dir)


def test_validate_schema_fails_on_bad_date(spark):
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_dir = tmpdir

        _write_text(os.path.join(raw_dir, "customers.csv"), "customer_id\nc1\n")
        _write_text(os.path.join(raw_dir, "articles.csv"), "article_id\n1001\n")
        _write_text(
            os.path.join(raw_dir, "transactions_train.csv"),
            "t_dat,customer_id,article_id,price,sales_channel_id\n"
            "not-a-date,c1,1001,10.0,1\n",
        )

        with pytest.raises(AssertionError, match="non-parseable t_dat"):
            validate_schema_and_quality(spark=spark, raw_dir=raw_dir)


def test_validate_schema_fails_on_low_join_coverage(spark):
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_dir = tmpdir

        _write_text(os.path.join(raw_dir, "customers.csv"), "customer_id\nc999\n")
        _write_text(os.path.join(raw_dir, "articles.csv"), "article_id\n9999\n")
        _write_text(
            os.path.join(raw_dir, "transactions_train.csv"),
            "t_dat,customer_id,article_id,price,sales_channel_id\n"
            "2020-09-01,c1,1001,10.0,1\n",
        )

        with pytest.raises(AssertionError, match="Join coverage too low"):
            validate_schema_and_quality(
                spark=spark,
                raw_dir=raw_dir,
                thresholds=CoverageThresholds(tx_to_customers_min=0.98, tx_to_articles_min=0.98),
            )
