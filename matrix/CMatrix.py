from __future__ import annotations
from typing import List, Tuple
from ctypes import CDLL, Structure, POINTER, CFUNCTYPE, byref, c_int, c_double
import pprint

from pathlib import Path
libcmatrix = CDLL(Path(__file__).parent.absolute().joinpath('CMatrix/libcmatrix.so'))
libcmatrix.cmat_sum.restype = c_double

class CMatrix(Structure):
    _fields_ = [
        ("row", c_int),
        ("column", c_int),
        ("data", POINTER(POINTER(c_double))),
    ]

    @staticmethod
    def __sublist_as_tuple(l):
        return [tuple(i) for i in l]

    def __init__(self, data: List[Tuple[float]]) -> None:
        self.row = len(data)
        self.column = len(data[0])

        data = CMatrix.__sublist_as_tuple(data)

        cdata2d = (c_double*self.column*self.row)(*data)
        LP_C_DOUBLE = POINTER(c_double)
        self.data = (LP_C_DOUBLE*self.row)(*cdata2d)
        pass

    def to_list(self):
        pydata2d = []
        for r in range(self.row):
            datarow = [self.data.contents[r*self.column + c]
                        for c in range(self.column)]
            pydata2d.append(datarow)
        return pydata2d

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.row, self.column)

    @property
    def T(self) -> CMatrix:
        ret = CMatrix.zeros(self.column, self.row)
        libcmatrix.cmat_transpose(byref(self), byref(ret))
        return ret

    def __repr__(self) -> str:
        # TODO: better print
        return pprint.saferepr(self.to_list())

    def __getitem__(self, index: tuple | int) -> List[float] | float:
        list_data = self.to_list()

        if isinstance(index, int):
            # get a row
            return list_data[index]
            # TODO: implement getting a column
        elif isinstance(index, tuple):
            # get a number
            if len(index) != 2:
                raise Exception("index length must <= 2")
            return list_data[index[0]][index[1]]
        else:
            raise Exception("use subscript as [1] or [1, 2]")

    def __setitem__(self, index: tuple | int, val: float | List[float]) -> None:
        if isinstance(index, int) and isinstance(val, (tuple, list)):
            # set a row
            assert len(val) == self.column,\
                f"column num mismatch, expect: {self.column}, input: {len(val)}"
            assert index < self.row,\
                f"index out of range: [0, {self.row})"
            for i in range(index*self.column, (index+1)*self.column):
                self.data.contents[i] = val[i]
            # TODO: implement setting columns
        elif isinstance(index, tuple) and isinstance(val, (float, int)):
            # set a number
            if len(index) != 2:
                raise Exception("index length must <= 2")
            self.data.contents[index[0]*self.column + index[1]] = float(val)
        else:
            raise Exception("use subscript index as [1] or [1, 2]")

    def __add__(self, matrix: CMatrix) -> CMatrix:
        ret = CMatrix.zeros(self.row, self.column)
        libcmatrix.cmat_add(byref(self), byref(matrix), byref(ret))
        return ret

    def __sub__(self, matrix: CMatrix) -> CMatrix:
        ret = CMatrix.zeros(self.row, self.column)
        libcmatrix.cmat_sub(byref(self), byref(matrix), byref(ret))
        return ret

    def __mul__(self, matrix: CMatrix) -> CMatrix:
        ret = CMatrix.zeros(self.row, self.column)
        libcmatrix.cmat_mul_mat(byref(self), byref(matrix), byref(ret))
        return ret

    def rows(self) -> List[float]:
        for r in self.to_list():
            yield r

    def columns(self) -> List[float]:
        for c in self.T.to_list():
            yield c

    def sum(self, key=lambda x: x) -> float:
        KEY_FUNC = CFUNCTYPE(c_double, c_double)
        return libcmatrix.cmat_sum(byref(self), KEY_FUNC(key))

    def __iter__(self) -> float:
        for i in range(self.row*self.column):
            yield self.data.contents[i]

    def apply(self, operate: function) -> CMatrix:
        KEY_FUNC = CFUNCTYPE(c_double, c_double)
        ret = CMatrix.zeros(self.row, self.column)
        libcmatrix.cmat_apply(byref(self), byref(ret), KEY_FUNC(operate))
        return ret

    @staticmethod
    def zeros(row: int, column: int) -> CMatrix:
        """ new zeros matrix """
        data = []
        for _ in range(row):
            data.append([0.0]*column)
        return CMatrix(data)

    @staticmethod
    def dot(mat1: CMatrix, mat2: CMatrix) -> CMatrix:
        """ Calculate dot product of 2 matrices. """
        ret = CMatrix.zeros(mat1.row, mat2.column)
        libcmatrix.cmat_dot(byref(mat1), byref(mat2), byref(ret))
        return ret

    @staticmethod
    def mul(mat: CMatrix, val: float | int) -> CMatrix:
        ret = CMatrix.zeros(mat.row, mat.column)
        libcmatrix.cmat_mul_val(byref(mat), c_double(val), byref(ret))
        return ret

# test CMatrix
if __name__ == '__main__':

    mat1 = CMatrix([[10, 2.0], [3.0, 4.0], [5.0, 6.0]])
    mat2 = CMatrix([(10, 2.0), (3.0, 4.0), (5.0, 6.0)])
    mat3 = CMatrix([(10, 2.0), (3.0, 4.0), (5.0, 6.0)])
    libcmatrix.cmat_add(byref(mat1), byref(mat2), byref(mat3))
    # ret = pointer(new_mat)
    # print(ret.row, ret.data)

    print(mat3.to_list())

    mat4 = CMatrix([(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)])
    libcmatrix.cmat_transpose(byref(mat3), byref(mat4))
    print(mat4.to_list())

    mat5 = CMatrix([(1, 0, -1), (2, 2, 3), (0, 1, -5)])
    mat6 = CMatrix([(2, 1, 4), (0, 2, 0), (0, 3, 1)])
    mat7 = CMatrix([(2, 1, 4), (0, 2, 0), (0, 3, 1)])
    libcmatrix.cmat_dot(byref(mat5), byref(mat6), byref(mat7))
    # [[2, -2, 3], [4, 15, 11], [0, -13, -5]]
    print(mat7.to_list())
