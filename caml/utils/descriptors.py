import ibis
import pandas
import polars
import pyspark
import sklearn


class ValidDataFrame:
    """
    Descriptor class that allows setting and getting a value that must be a valid DataFrame
    of type pandas.DataFrame, polars.DataFrame, pyspark.sql.DataFrame, or ibis.expr.types.Table.

    Parameters
    ----------
    strict:
        If True, the value must be a pandas.DataFrame, polars.DataFrame, pyspark.sql.DataFrame, or ibis.expr.types.Table.
        If False, None is an acceptable value.

    Raises
    ------
    ValueError
        If the assigned value is not a valid data type.
    """

    def __init__(self, strict=False):
        self.strict = strict

    def __get__(self, obj, objtype=None):
        return self.value

    def __set__(self, obj, value):
        if not self.strict:
            if value is None:
                self.value = None
                return

        if not isinstance(
            value,
            (
                pyspark.sql.DataFrame,
                pandas.DataFrame,
                polars.DataFrame,
                ibis.expr.types.Table,
            ),
        ):
            raise ValueError(
                "Value must be a valid DataFrame of type pyspark.sql.DataFrame, pandas.DataFrame, polars.DataFrame, or ibis.expr.types.Table."
            )

        self.value = value


class ValidSparkSession:
    """
    Descriptor class for a valid SparkSession object.

    Parameters
    ----------
    strict:
        If True, only a non-None SparkSession object is considered valid.
        If False, None or a valid SparkSession object is considered valid.

    Raises
    ------
    ValueError
        If the assigned value is not a valid data type.
    """

    def __init__(self, strict=False):
        self.strict = strict

    def __get__(self, obj, objtype=None):
        return self.value

    def __set__(self, obj, value):
        if not self.strict:
            if value is None:
                self.value = None
                return

        if not isinstance(value, pyspark.sql.SparkSession):
            raise ValueError("Value must be a valid SparkSession.")

        self.value = value


class ValidSklearnModel:
    """
    Descriptor class for a valid nuissance Sklearn model object.

    Parameters
    ----------
    strict:
        If True, only a non-None Sklearn model object is considered valid.
        If False, None or a valid Sklearn model object is considered valid.

    Raises
    ------
    ValueError
        If the assigned value is not a valid data type.
    """

    def __init__(self, strict=False):
        self.strict = strict

    def __get__(self, obj, objtype=None):
        return self.value

    def __set__(self, obj, value):
        if not self.strict:
            if value is None:
                self.value = None
                return

        if not (
            isinstance(value, sklearn.base.RegressorMixin)
            or isinstance(value, sklearn.base.ClassifierMixin)
        ):
            raise ValueError("Value must be a valid Sklearn model object.")

        self.value = value


class ValidBoolean:
    """
    Descriptor class for a valid boolean value.

    Parameters
    ----------
    strict:
        If True, only a non-None boolean is considered valid.
        If False, None or a valid boolean is considered valid.

    Raises
    ------
    ValueError
        If the assigned value is not a valid data type.
    """

    def __init__(self, strict=False):
        self.strict = strict

    def __get__(self, obj, objtype=None):
        return self.value

    def __set__(self, obj, value):
        if not self.strict:
            if value is None:
                self.value = None
                return

        if not isinstance(value, bool):
            raise ValueError("Value must be a boolean.")

        self.value = value


class ValidFeatureList:
    """
    Descriptor class for a valid str, list, or tuple feature names.

    Parameters
    ----------
    strict:
        If True, only a non-None string, list, or tuple of strings is considered valid.
        If False, None or a valid string, list, or tuple of strings is considered valid.

    Raises
    ------
    ValueError
        If the assigned value is not a valid data type.
    """

    def __init__(self, strict=False):
        self.strict = strict

    def __get__(self, obj, objtype=None):
        return self.value

    def __set__(self, obj, value):
        if not self.strict:
            if value is None:
                self.value = None
                return

        if not (
            isinstance(value, list)
            or isinstance(value, str)
            or isinstance(value, tuple)
        ):
            raise ValueError("Value must be a string, list, or tuple.")
        if isinstance(value, list) or isinstance(value, tuple):
            if not all(isinstance(x, str) for x in value):
                raise ValueError("List/tuple must contain only strings.")

        self.value = value


class ValidString:
    """
    Descriptor class for a valid string value.

    Parameters
    ----------
    strict:
        If True, only a non-None string is considered valid.
        If False, None or a valid string is considered valid.

    Raises
    ------
    ValueError
        If the assigned value is not a valid data type.
    """

    def __init__(self, strict=False):
        self.strict = strict

    def __get__(self, obj, objtype=None):
        return self.value

    def __set__(self, obj, value):
        if not self.strict:
            if value is None:
                self.value = None
                return

        if not isinstance(value, str):
            raise ValueError("Value must be a string.")

        self.value = value
