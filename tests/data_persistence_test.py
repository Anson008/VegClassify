import pytest
import numpy as np
from vegetation_index.data_persistence import DataArray2D


def test_data_array_2d_constructor():
    n_rows = 12
    col_names = ("col1", "col2", "col3")

    data_array = DataArray2D(n_rows, col_names)
    assert isinstance(data_array, DataArray2D)


def test_data_array_2d_constructor_handles_invalid_column_name():
    with pytest.raises(Exception) as exp:
        DataArray2D(12, "names")
    assert str(exp.value) == "Column names must be a tuple."

    with pytest.raises(Exception) as exp:
        DataArray2D(12, ())
    assert str(exp.value) == "Number of column names must be greater than 0."

    with pytest.raises(Exception) as exp:
        DataArray2D(12, (1,))
    assert str(exp.value) == "A column name must be a string."


def test_data_array_2d_get_shape():
    n_rows = 12
    col_names = ("col1", "col2", "col3")

    data_array = DataArray2D(n_rows, col_names)
    data_array_shape = data_array.get_shape()

    assert data_array_shape[0] == 12
    assert data_array_shape[1] == 3


def test_data_array_2d_update():
    n_rows = 3
    col_names = ("col1", "col2")
    new_value = 16

    data_array = DataArray2D(n_rows, col_names)
    data_array[0, 0] = new_value

    assert data_array[0, 0] == new_value
