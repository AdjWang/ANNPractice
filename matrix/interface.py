from __future__ import annotations
from typing import List, Tuple

# from .PyMatrix import PyMatrix as MatrixImpl
from .CMatrix import CMatrix as MatrixImpl

ENABLE_VERIFIER = True

def verifier_wrapper(vfunc):
    if ENABLE_VERIFIER:
        return vfunc
    else:
        def _empty(*args, **kwargs):
            pass
        return _empty


class Verifier:
    """ Verify the matrix before do some operations. """
    @verifier_wrapper
    @staticmethod
    def instance(mat):
        if not isinstance(mat, Matrix):
            raise Exception(f"invlid matrix instance type: {type(mat)}")

    @verifier_wrapper
    @staticmethod
    def min_dim(mat):
        row, column = mat.shape
        if row == 0:
            raise Exception(f"invalid row: {row}")
        if column == 0:
            raise Exception(f"invalid column: {column}")

    @verifier_wrapper
    @staticmethod
    def identical_shape(mat1, mat2):
        if mat1.shape != mat2.shape:
            raise Exception(
                f"matrix shape mismatch, mat1: {mat1.shape}, mat2: {mat2.shape}")

    @verifier_wrapper
    @staticmethod
    def square(mat: MatrixImpl):
        if mat.shape[0] != mat.shape[1]:
            raise Exception(f"not square: {mat.shape}")

    @verifier_wrapper
    @staticmethod
    def dot_dim(mat1, mat2):
        _, column1 = mat1.shape
        row2, _ = mat2.shape
        if column1 != row2:
            raise Exception(
                f"matrix dot shape mismatch, mat1: {mat1.shape}, mat2: {mat2.shape}")


class Matrix:
    @staticmethod
    def zeros(row: int, column: int) -> Matrix:
        """ new zeros matrix """
        data = []
        for _ in range(row):
            data.append([0.0]*column)
        return Matrix(MatrixImpl(data))

    @staticmethod
    def from_const(row: int, column: int, val: float) -> Matrix:
        data = []
        for _ in range(row):
            data.append([val]*column)
        return Matrix(MatrixImpl(data))

    @staticmethod
    def from_generator(row: int, column: int, val_generator: function) -> Matrix:
        data = []
        for _ in range(row):
            data.append([val_generator() for _ in range(column)])
        return Matrix(MatrixImpl(data))

    @staticmethod
    def from_list(data: List[List[float]]) -> Matrix:
        return Matrix(MatrixImpl(data))

    @staticmethod
    def from_diag(data: List[float]) -> Matrix:
        result = []
        for _ in range(len(data)):
            result.append([0]*len(data))

        for idx, i in enumerate(data):
            result[idx][idx] = i
        return Matrix(MatrixImpl(result))

    """ Implement matrix and support some basic operations. """
    def __init__(self, mat_impl: MatrixImpl) -> None:
        self._mat_impl = mat_impl

        # verify args
        Verifier.min_dim(self)

    def to_list(self) -> List[List[float]]:
        return self._mat_impl.to_list()

    @property
    def shape(self) -> Tuple[int, int]:
        return self._mat_impl.shape

    @property
    def T(self) -> Matrix:
        return Matrix(self._mat_impl.T)

    def __repr__(self) -> str:
        return self._mat_impl.__repr__()

    def __getitem__(self, index: tuple | int) -> List[float] | float:
        if isinstance(index, int):
            # get a row
            return self._mat_impl.__getitem__(index)
            # TODO: implement getting a column
        elif isinstance(index, tuple):
            # get a number
            if len(index) != 2:
                raise Exception(f"index length must <= 2, now: {index}")
            return self._mat_impl.__getitem__(index)
        else:
            raise Exception(f"use subscript as [1] or [1, 2], now: {index}")

    def __setitem__(self, index: tuple | int, val: float | List[float]) -> None:
        row, column = self.shape
        if isinstance(index, int) and isinstance(val, (tuple, list)):
            # set a row
            assert len(val) == column,\
                f"column num mismatch, expect: {column}, input: {len(val)}"
            assert index < row,\
                f"index out of range: [0, {row})"
            self._mat_impl.__setitem__(index, val)
        elif isinstance(index, tuple) and isinstance(val, (float, int)):
            # set a number
            if len(index) != 2:
                raise Exception(f"index length must <= 2, now: {index}")
            if index[0]*column + index[1] >= row * column:
                raise Exception(f"index out of range: {index}, max: {(row, column)}")
            self._mat_impl.__setitem__(index, val)
        else:
            raise Exception(f"use subscript as [1] or [1, 2], val as float or int, now: {index}, {type(val)}")

    def __add__(self, matrix: Matrix) -> Matrix:
        # verify
        Verifier.instance(matrix)
        Verifier.identical_shape(self, matrix)

        return Matrix(self._mat_impl.__add__(matrix._mat_impl))

    def __sub__(self, matrix: Matrix) -> Matrix:
        # verify
        Verifier.instance(matrix)
        Verifier.identical_shape(self, matrix)

        return Matrix(self._mat_impl.__sub__(matrix._mat_impl))

    def __mul__(self, matrix: Matrix) -> Matrix:
        # verify
        Verifier.instance(matrix)
        Verifier.identical_shape(self, matrix)

        return Matrix(self._mat_impl.__mul__(matrix._mat_impl))

    def rows(self) -> List[float]:
        for row in self._mat_impl.rows():
            yield row

    def columns(self) -> List[float]:
        for column in self._mat_impl.columns():
            yield column

    def sum(self, key=lambda x: x) -> float:
        return self._mat_impl.sum(key)

    def __iter__(self) -> float:
        for num in self._mat_impl:
            yield num

    def apply(self, operate: function) -> Matrix:
        return Matrix(self._mat_impl.apply(operate))

    @staticmethod
    def dot(mat1: Matrix, mat2: Matrix) -> Matrix:
        """ Calculate dot product of 2 matrices. """
        # verify
        Verifier.instance(mat1)
        Verifier.instance(mat2)
        Verifier.dot_dim(mat1, mat2)

        return Matrix(MatrixImpl.dot(mat1._mat_impl, mat2._mat_impl))

    @staticmethod
    def mul(mat: Matrix, val: float | int) -> Matrix:
        # verify
        Verifier.instance(mat)

        return Matrix(MatrixImpl.mul(mat._mat_impl, val))

    # for pickle
    def __getstate__(self):
        return self.to_list()

    # for pickle
    def __setstate__(self, data):
        self._mat_impl = MatrixImpl(data)
