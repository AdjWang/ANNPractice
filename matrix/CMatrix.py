import ctypes
from ctypes import CDLL


class CMatrix(ctypes.Structure):
    _fields_ = [
        ("row", ctypes.c_int),
        ("column", ctypes.c_int),
        ("data", ctypes.POINTER(ctypes.POINTER(ctypes.c_double))),
    ]

    @staticmethod
    def from_list(data2d):
        row = len(data2d)
        column = len(data2d[0])
        cdata2d = (ctypes.c_double*column*row)(*data2d)
        LP_C_DOUBLE = ctypes.POINTER(ctypes.c_double)
        cdata2d = (LP_C_DOUBLE*row)(*cdata2d)
        return CMatrix(row, column, cdata2d)

    def __init__(self, row, column, cdata2d):
        self.row = row
        self.column = column
        self.data = cdata2d

    def to_list(self):
        pydata2d = []
        for r in range(self.row):
            datarow = [self.data.contents[r*self.column + c]
                        for c in range(self.column)]
            pydata2d.append(datarow)
        return pydata2d

# test CMatrix
if __name__ == '__main__':
    libcmatrix = CDLL('./CMatrix/libcmatrix.so')

    mat1 = CMatrix.from_list([(10, 2.0), (3.0, 4.0), (5.0, 6.0)])
    mat2 = CMatrix.from_list([(10, 2.0), (3.0, 4.0), (5.0, 6.0)])
    mat3 = CMatrix.from_list([(10, 2.0), (3.0, 4.0), (5.0, 6.0)])
    libcmatrix.cmat_add(ctypes.byref(mat1), ctypes.byref(mat2), ctypes.byref(mat3))
    # ret = ctypes.pointer(new_mat)
    # print(ret.row, ret.data)

    print(mat3.to_list())

    mat4 = CMatrix.from_list([(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)])
    libcmatrix.cmat_transpose(ctypes.byref(mat3), ctypes.byref(mat4))
    print(mat4.to_list())

    mat5 = CMatrix.from_list([(1, 0, -1), (2, 2, 3), (0, 1, -5)])
    mat6 = CMatrix.from_list([(2, 1, 4), (0, 2, 0), (0, 3, 1)])
    mat7 = CMatrix.from_list([(2, 1, 4), (0, 2, 0), (0, 3, 1)])
    libcmatrix.cmat_dot(ctypes.byref(mat5), ctypes.byref(mat6), ctypes.byref(mat7))
    # [[2, -2, 3], [4, 15, 11], [0, -13, -5]]
    print(mat7.to_list())
