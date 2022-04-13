""" Matrix implementation.
Just a naive version :)
"""
from __future__ import annotations
from typing import List, Tuple
import pprint


class PyMatrix:
    """ Implement matrix and support some basic operations. """
    def __init__(self, data: List[List[float]]) -> None:
        # init data
        self.__data = data
        # matrix properties
        self.__row = len(data)
        self.__column = len(data[0]) if len(data) > 0 else 0

    def to_list(self) -> List[List[float]]:
        return self.__data

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.__row, self.__column)

    @property
    def T(self) -> PyMatrix:
        return PyMatrix([list(i) for i in zip(*self.__data)])

    def __repr__(self) -> str:
        # TODO: better print
        return pprint.saferepr(self.__data)

    def __getitem__(self, index: tuple | int) -> List[float] | float:
        # more safty checking at interface.py.Matrix()
        if isinstance(index, int):
            # get a row
            return self.__data[index]
            # TODO: implement getting a column
        elif isinstance(index, tuple):
            # get a number
            if len(index) != 2:
                raise Exception(f"index length must <= 2, now: {index}")
            return self.__data[index[0]][index[1]]
        else:
            raise Exception(f"use subscript as [1] or [1, 2], now: {index}")

    def __setitem__(self, index: tuple | int, val: float | List[float]) -> None:
        # more safty checking at interface.py.Matrix()
        if isinstance(index, int) and isinstance(val, (tuple, list)):
            # set a row
            self.__data[index] = val
            # TODO: implement setting columns
        elif isinstance(index, tuple) and isinstance(val, (float, int)):
            # set a number
            self.__data[index[0]][index[1]] = val
        else:
            raise Exception(f"use subscript as [1] or [1, 2], val as float or int, now: {index}, {type(val)}")

    def __add__(self, matrix: PyMatrix) -> PyMatrix:
        new_data = list(map(list, self.__data))  # make a copy
        # point wise add
        for r in range(self.__row):
            for c in range(self.__column):
                new_data[r][c] += matrix.__data[r][c]

        return PyMatrix(new_data)

    def __sub__(self, matrix: PyMatrix) -> PyMatrix:
        new_data = list(map(list, self.__data))  # make a copy
        # point wise substract
        for r in range(self.__row):
            for c in range(self.__column):
                new_data[r][c] -= matrix.__data[r][c]

        return PyMatrix(new_data)

    def __mul__(self, matrix: PyMatrix) -> PyMatrix:
        new_data = list(map(list, self.__data))  # make a copy
        # point wise substract
        for r in range(self.__row):
            for c in range(self.__column):
                new_data[r][c] *= matrix.__data[r][c]

        return PyMatrix(new_data)

    def rows(self) -> List[float]:
        for r in self.__data:
            yield r

    def columns(self) -> List[float]:
        for c in self.T.__data:
            yield c

    def sum(self, key=lambda x: x) -> float:
        summary = 0.0
        for row in self.rows():
            for num in row:
                summary += key(num)
        return summary

    def __iter__(self) -> float:
        for row in self.rows():
            for num in row:
                yield num

    def apply(self, operate: function) -> PyMatrix:
        row, column = self.shape
        new_data = list(map(list, self.__data))  # make a copy
        for r in range(row):
            for c in range(column):
                new_data[r][c] = operate(self.__data[r][c])
        return PyMatrix(new_data)

    @staticmethod
    def zeros(row: int, column: int) -> PyMatrix:
        """ new zeros matrix """
        data = []
        for _ in range(row):
            data.append([0.0]*column)
        return PyMatrix(data)

    @staticmethod
    def dot(mat1: PyMatrix, mat2: PyMatrix) -> PyMatrix:
        """ Calculate dot product of 2 matrices. """
        row1, _ = mat1.shape
        _, column2 = mat2.shape
        result = PyMatrix.zeros(row1, column2)
        for idx_r, r in enumerate(mat1.rows()):
            for idx_c, c in enumerate(mat2.columns()):
                result[idx_r][idx_c] = sum([a*b for a, b in zip(r, c)])

        return result

    @staticmethod
    def mul(mat: PyMatrix, val: float | int) -> PyMatrix:
        row, column = mat.shape
        result = PyMatrix.zeros(row, column)
        for r in range(row):
            for c in range(column):
                result[r][c] = mat[r][c] * val

        return result
