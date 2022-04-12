from __future__ import annotations
from typing import List, Tuple

from .PyMatrix import PyMatrix as MatrixImpl


class Verifier:
    """ Verify the matrix before do some operations. """
    @staticmethod
    def instance(mat):
        if not isinstance(mat, MatrixImpl):
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
    def square(mat: MatrixImpl):
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
    @staticmethod
    def zeros(row: int, column: int) -> MatrixImpl:
        """ new zeros matrix """
        data = []
        for _ in range(row):
            data.append([0.0]*column)
        return MatrixImpl(data)

    @staticmethod
    def by_const(row: int, column: int, val: float) -> MatrixImpl:
        data = []
        for _ in range(row):
            data.append([val]*column)
        return MatrixImpl(data)

    @staticmethod
    def by_generator(row: int, column: int, val_generator: function) -> MatrixImpl:
        data = []
        for _ in range(row):
            data.append([val_generator() for _ in range(column)])
        return MatrixImpl(data)

    @staticmethod
    def by_list(data: List[List[float]]) -> MatrixImpl:
        return MatrixImpl(data)

    @staticmethod
    def by_diag(data: List[float]) -> MatrixImpl:
        result = []
        for _ in range(len(data)):
            result.append([0]*len(data))

        for idx, i in enumerate(data):
            result[idx][idx] = i
        return MatrixImpl(result)

    """ Implement matrix and support some basic operations. """
    def __init__(self, data: List[List[float]]) -> None:
        self.__mat_impl = MatrixImpl(data)

        # verify args
        Verifier.min_dim(self)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.__mat_impl.shape

    @property
    def T(self) -> MatrixImpl:
        return self.__mat_impl.T

    def __repr__(self) -> str:
        return self.__mat_impl.__repr__()

    def __getitem__(self, index: tuple | int) -> List[float] | float:
        return self.__mat_impl.__getitem__(index)

    def __setitem__(self, index: tuple | int, val: float | List[float]) -> None:
        self.__mat_impl.__setitem__(index, val)

    def __add__(self, matrix: MatrixImpl) -> MatrixImpl:
        # verify
        Verifier.instance(matrix)
        Verifier.identical_shape(self, matrix)

        return self.__mat_impl.__add__(matrix)

    def __sub__(self, matrix: MatrixImpl) -> MatrixImpl:
        # verify
        Verifier.instance(matrix)
        Verifier.identical_shape(self, matrix)

        return self.__mat_impl.__sub__(matrix)

    def __mul__(self, matrix: MatrixImpl) -> MatrixImpl:
        # verify
        Verifier.instance(matrix)
        Verifier.identical_shape(self, matrix)

        return self.__mat_impl.__mul__(matrix)

    def rows(self) -> List[float]:
        return self.__mat_impl.rows()

    def columns(self) -> List[float]:
        return self.__mat_impl.columns()

    def sum(self, key=lambda x: x) -> float:
        return self.__mat_impl.sum(key)

    def __iter__(self) -> float:
        return self.__mat_impl.__iter__()

    def apply(self, operate: function) -> MatrixImpl:
        return self.__mat_impl.apply(operate)

    @staticmethod
    def dot(mat1: MatrixImpl, mat2: MatrixImpl) -> MatrixImpl:
        """ Calculate dot product of 2 matrices. """
        # verify
        Verifier.instance(mat1)
        Verifier.instance(mat2)
        Verifier.dot_dim(mat1, mat2)

        return MatrixImpl.dot(mat1, mat2)

    @staticmethod
    def mul(mat: MatrixImpl, val: float | int) -> MatrixImpl:
        # verify
        Verifier.instance(mat)

        return MatrixImpl.dot(mat, val)


