import os
import sys

import pytest
from pyspark.sql import SparkSession

os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"


@pytest.fixture
def spark(scope="session"):
    """
    Creates local spark object for running pytest
    """

    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    spark = (
        SparkSession.builder.master("local[1]")
        .appName("local-tests")
        .config("spark.executor.cores", "1")
        .config("spark.executor.instances", "1")
        .config("spark.sql.shuffle.partitions", "1")
        .getOrCreate()
    )

    yield spark
    spark.stop()


def pytest_configure(config):
    """
    Disable component logger logging for all tests.
    """
    os.environ["COMPONENT_LOGGER_DUMMY_MODE"] = "yes"
