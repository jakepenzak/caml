import abc

import ibis
import pandas
import polars
import pyspark

from ..utils import generate_random_string


class CamlBase(metaclass=abc.ABCMeta):
    """
    Base ABC class for core classes.
    """

    @abc.abstractmethod
    def get_features(self):
        pass

    @abc.abstractmethod
    def fit(self):
        pass

    @abc.abstractmethod
    def predict(self):
        pass

    @abc.abstractmethod
    def optimize(self):
        pass

    @abc.abstractmethod
    def summarize(self):
        pass

    @abc.abstractmethod
    def rank(self):
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

        if isinstance(self.df, pyspark.sql.DataFrame):
            self.df.createOrReplaceTempView(table_name)
            ibis_connection = ibis.pyspark.connect(session=self.spark)
            ibis_df = ibis_connection.table(table_name)
        elif isinstance(self.df, pandas.DataFrame):
            ibis_connection = ibis.pandas.connect({table_name: self.df})
            ibis_df = ibis_connection.table(table_name)
        elif isinstance(self.df, polars.DataFrame):
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

    def _create_internal_ibis_table(self, data_dict: dict):
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
            results_df = pandas.DataFrame(data_dict)
            ibis_results_df = self._ibis_connection.create_table(
                name=table_name, obj=results_df
            )
        elif backend == "polars":
            results_df = polars.from_dict(data_dict)
            ibis_results_df = self._ibis_connection.create_table(
                name=table_name, obj=results_df
            )
        elif backend == "pyspark":
            results_df = self.spark.createDataFrame(pandas.DataFrame(data_dict))
            results_df.createOrReplaceTempView(table_name)
            ibis_results_df = self._ibis_connection.table(table_name)

        return ibis_results_df
