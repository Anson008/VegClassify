import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Tuple


class DataArray2D:
    def __init__(self, n_rows: int, col_names: Tuple):
        if not isinstance(col_names, Tuple):
            raise TypeError("Column names must be a tuple.")
        if len(col_names) == 0:
            raise ValueError("Number of column names must be greater than 0.")
        if not isinstance(col_names[0], str):
            raise TypeError("A column name must be a string.")

        self._n_rows = n_rows
        self._n_cols = len(col_names)
        self._col_names = list(col_names)
        self._data_array = np.zeros((self._n_rows, self._n_cols))

    @property
    def data_array(self):
        return self._data_array

    def __getitem__(self, index):
        return self._data_array[index]

    def __setitem__(self, index, value):
        self._data_array[index] = value

    def get_shape(self):
        return self._data_array.shape

    def save_to_csv(self, file_path: str):
        df = pd.DataFrame(self._data_array, columns=self._col_names)
        df.to_csv(file_path, index=False)



