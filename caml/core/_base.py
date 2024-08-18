import abc

import ibis

try:
    import pandas
except ImportError:
    pandas = None

try:
    import polars
except ImportError:
    polars = None

try:
    import pyspark
    from pyspark.sql import SparkSession
except ImportError:
    pyspark = None

import logging

from ..utils import generate_random_string

logger = logging.getLogger(__name__)


class CamlBase(metaclass=abc.ABCMeta):
    """
    Base ABC class for core classes.
    """

    @property
    def dataframe(self):
        return self._return_ibis_dataframe_to_original_backend(ibis_df=self._ibis_df)

    @property
    def final_estimator(self):
        if self._final_estimator is not None:
            logger.info(
                "The best estimator has been fit on the entire dataset and will be returned."
            )
            return self._final_estimator
        elif self._best_estimator is not None:
            logger.info(
                "The best estimator has NOT been fit on the entire dataset. This is returning the estimator fit on the training dataset. Please run fit() method with final_estimator=True to fit the best estimator on the entire dataset, once validated."
            )
            return self._best_estimator
        else:
            raise ValueError(
                "No estimator has been fit yet. Please run fit() method first."
            )

    @abc.abstractmethod
    def fit(self):
        pass

    @abc.abstractmethod
    def validate(self):
        pass

    @abc.abstractmethod
    def predict(self):
        pass

    @abc.abstractmethod
    def rank_order(self):
        pass

    @abc.abstractmethod
    def summarize(self):
        pass

    def _ibis_connector(
        self,
        custom_table_name: str | None = None,
    ):
        """
        Connects the DataFrame to the Ibis backend based on the type of DataFrame.

        If the DataFrame is a pyspark.sql.DataFrame, it creates a temporary view and connects to Ibis using the PySpark session.
        If the DataFrame is a pandas.DataFrame, it connects to Ibis using the pandas DataFrame.
        If the DataFrame is a polars.DataFrame, it connects to Ibis using the polars DataFrame.
        If the DataFrame is an ibis.expr.types.Table, it creates a new table (copy of df) on Ibis using the current Ibis connection.

        This method sets the '_table_name`, '_ibis_df` and `_ibis_connection` internal attributes of the class when nonbase_df is None.

        Parameters
        ----------
        custom_table_name:
            The custom table name to use for the DataFrame in Ibis, by default None

        Returns
        -------
        str | None
            The table name of the DataFrame in Ibis if nonbase_df is not None, else None
        ibis.expr.types.Table | None
            The Ibis table expression of the DataFrame if nonbase_df is not None, else None
        ibis.client.Client | None
            The Ibis client object if nonbase_df is not None, else None
        """
        if custom_table_name is None:
            table_name = generate_random_string(10)
        else:
            table_name = custom_table_name

        if pyspark and isinstance(self.df, pyspark.sql.DataFrame):
            self._spark = SparkSession.builder.getOrCreate()
            self.df.createOrReplaceTempView(table_name)
            ibis_connection = ibis.pyspark.connect(session=self._spark)
            ibis_df = ibis_connection.table(table_name)
        elif pandas and isinstance(self.df, pandas.DataFrame):
            ibis_connection = ibis.pandas.connect({table_name: self.df})
            ibis_df = ibis_connection.table(table_name)
        elif polars and isinstance(self.df, polars.DataFrame):
            ibis_connection = ibis.polars.connect({table_name: self.df})
            ibis_df = ibis_connection.table(table_name)
        elif isinstance(self.df, ibis.expr.types.Table):
            ibis_connection = self.df._find_backend()
            if isinstance(ibis_connection, ibis.backends.pyspark.Backend):
                obj = self.df
            else:
                obj = self.df.execute()
            ibis_df = ibis_connection.create_view(name=table_name, obj=obj)

        self._table_name = table_name
        self._ibis_df = ibis_df
        self._ibis_connection = ibis_connection

    def _create_internal_ibis_table(
        self,
        data_dict: dict | None = None,
        df: ibis.expr.types.Table
        | pyspark.sql.DataFrame
        | pandas.DataFrame
        | polars.DataFrame
        | None = None,
    ):
        """
        Create an internal Ibis DataFrame based on the provided data dictionary.

        Args:
            data_dict (dict): A dictionary containing the data for the DataFrame.

        Returns:
            ibis_results_df: The created Ibis DataFrame.
        """

        table_name = generate_random_string(10)

        backend = self._ibis_connection.name

        if backend == "pandas":
            if data_dict is not None:
                df = pandas.DataFrame(data_dict)
            ibis_df = self._ibis_connection.create_table(name=table_name, obj=df)

        elif backend == "polars":
            if data_dict is not None:
                df = polars.from_dict(data_dict)
            ibis_df = self._ibis_connection.create_table(name=table_name, obj=df)
        elif backend == "pyspark":
            if data_dict is not None:
                df = self._spark.createDataFrame(pandas.DataFrame(data_dict))

            df.createOrReplaceTempView(table_name)
            ibis_df = self._ibis_connection.table(table_name)

        return ibis_df

    @staticmethod
    def _return_ibis_dataframe_to_original_backend(
        *, ibis_df: ibis.expr.types.Table, backend: str | None = None
    ):
        """
        Return the Ibis DataFrame to the original backend.

        Args:
            ibis_df: The Ibis DataFrame to return to the original backend.

        Returns:
            df: The DataFrame in the original backend.
        """

        if backend is None:
            backend = ibis_df._find_backend().name

        if backend == "pandas":
            df = ibis_df.to_pandas()
        elif backend == "polars":
            df = ibis_df.to_polars()
        elif backend == "pyspark":
            spark = SparkSession.builder.getOrCreate()
            df = spark.sql(ibis_df.compile())

        return df
