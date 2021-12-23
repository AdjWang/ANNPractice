""" Matrix implementation.
Just a naieve version :)
"""
from __future__ import annotations
from typing import List, Tuple
import pprint


class Verifier:
    """ Verify the matrix before do some operations. """
    @staticmethod
    def instance(mat):
        if not isinstance(mat, Matrix):
            raise Exception(f"invlid matrix instance type: {type(mat)}")

    @staticmethod
    def min_dim(mat):
        row, column = mat.shape
        if row == 0:
            raise Exception(f"invalid row: {row}")
        if column == 0:
            raise Exception(f"invalid column: {column}")

    @staticmethod
    def identical_shape(mat1, mat2):
        if mat1.shape != mat2.shape:
            raise Exception(
                f"matrix shape mismatch, mat1: {mat1.shape}, mat2: {mat2.shape}")

    @staticmethod
    def square(mat: Matrix):
        if mat.shape[0] != mat.shape[1]:
            raise Exception(f"not square: {mat.shape}")

    @staticmethod
    def dot_dim(mat1, mat2):
        _, column1 = mat1.shape
        row2, _ = mat2.shape
        if column1 != row2:
            raise Exception(
                f"matrix dot shape mismatch, mat1: {mat1.shape}, mat2: {mat2.shape}")


class Matrix:
    """ Implement matrix and support some basic operations. """
    @staticmethod
    def zeros(row: int, column: int) -> Matrix:
        """ new zeros matrix """
        data = []
        for _ in range(row):
            data.append([0.0]*column)
        return Matrix(data)

    @staticmethod
    def by_const(row: int, column: int, val: float) -> Matrix:
        data = []
        for _ in range(row):
            data.append([val]*column)
        return Matrix(data)

    @staticmethod
    def by_generator(row: int, column: int, val_generator: function) -> Matrix:
        data = []
        for _ in range(row):
            data.append([val_generator() for _ in range(column)])
        return Matrix(data)

    @staticmethod
    def by_list(data: List[List[float]]) -> Matrix:
        return Matrix(data)

    @staticmethod
    def by_diag(data: List[float]) -> Matrix:
        result = []
        for _ in range(len(data)):
            result.append([0]*len(data))

        for idx, i in enumerate(data):
            result[idx][idx] = i
        return Matrix(result)

    def __init__(self, data: List[List[float]]) -> None:
        # init data
        self.__data = data
        # matrix properties
        self.__row = len(data)
        self.__column = len(data[0]) if len(data) > 0 else 0

        # verify args
        Verifier.min_dim(self)

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.__row, self.__column)

    @property
    def T(self) -> Matrix:
        return Matrix([list(i) for i in zip(*self.__data)])

    def __repr__(self) -> str:
        # TODO: better print
        return pprint.saferepr(self.__data)

    def __getitem__(self, index: tuple | int) -> List[float] | float:
        if isinstance(index, int):
            # get a row
            return self.__data[index]
            # TODO: implement getting a column
        elif isinstance(index, tuple):
            # get a number
            if len(tuple) != 2:
                raise Exception("index length must <= 2")
            return self.__data[index[0]][index[1]]
        else:
            raise Exception("use subscript as [1] or [1, 2]")

    def __setitem__(self, index: tuple | int, val: float | List[float]) -> None:
        if isinstance(index, int) and isinstance(val, list):
            # set a row
            if len(val) != self.columns:
                raise Exception(
                    f"column num mismatch, expect: {self.columns}, input: {len(val)}")
            self.__data[index] = val
            # TODO: implement setting columns
        elif isinstance(index, tuple) and isinstance(val, float):
            # set a number
            if len(tuple) != 2:
                raise Exception("index length must <= 2")
            self.__data[index[0]][index[1]] = val
        else:
            raise Exception("use subscript index as [1] or [1, 2]")

    def __add__(self, matrix: Matrix) -> Matrix:
        # verify
        Verifier.instance(matrix)
        Verifier.identical_shape(self, matrix)

        new_data = list(map(list, self.__data))  # make a copy
        # point wise add
        for r in range(self.__row):
            for c in range(self.__column):
                new_data[r][c] += matrix.__data[r][c]

        return Matrix(new_data)

    def __sub__(self, matrix: Matrix) -> Matrix:
        # verify
        Verifier.instance(matrix)
        Verifier.identical_shape(self, matrix)

        new_data = list(map(list, self.__data))  # make a copy
        # point wise substract
        for r in range(self.__row):
            for c in range(self.__column):
                new_data[r][c] -= matrix.__data[r][c]

        return Matrix(new_data)

    def __mul__(self, matrix: Matrix) -> Matrix:
        # verify
        Verifier.instance(matrix)
        Verifier.identical_shape(self, matrix)

        new_data = list(map(list, self.__data))  # make a copy
        # point wise substract
        for r in range(self.__row):
            for c in range(self.__column):
                new_data[r][c] *= matrix.__data[r][c]

        return Matrix(new_data)

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

    def apply(self, operate: function) -> Matrix:
        row, column = self.shape
        new_data = list(map(list, self.__data))  # make a copy
        for r in range(row):
            for c in range(column):
                new_data[r][c] = operate(self.__data[r][c])
        return Matrix(new_data)

    @staticmethod
    def dot(mat1: Matrix, mat2: Matrix) -> Matrix:
        """ Calculate dot product of 2 matrices. """
        # verify
        Verifier.instance(mat1)
        Verifier.instance(mat2)
        Verifier.dot_dim(mat1, mat2)

        row1, _ = mat1.shape
        _, column2 = mat2.shape
        result = Matrix.zeros(row1, column2)
        for idx_r, r in enumerate(mat1.rows()):
            for idx_c, c in enumerate(mat2.columns()):
                result[idx_r][idx_c] = sum([a*b for a, b in zip(r, c)])

        return result

    @staticmethod
    def mul(mat: Matrix, val: float | int) -> Matrix:
        # verify
        Verifier.instance(mat)

        row, column = mat.shape
        result = Matrix.zeros(row, column)
        for r in range(row):
            for c in range(column):
                result[r][c] = mat[r][c] * val

        return result
