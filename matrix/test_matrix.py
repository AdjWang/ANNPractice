import ctypes
from ctypes import CDLL

class CMatrix(ctypes.Structure):
    _fields_ = [
        ("row", ctypes.c_int),
        ("column", ctypes.c_int),
        ("data", ctypes.POINTER(ctypes.POINTER(ctypes.c_double))),
    ]

    def __init__(self, mat):
        self.row = len(mat)
        self.column = len(mat[0])
        mat = (ctypes.c_double*self.column*self.row)(*mat)
        LP_C_DOUBLE = ctypes.POINTER(ctypes.c_double)
        self.data = (LP_C_DOUBLE*self.row)(*mat)

# test CMatrix
if __name__ == '__main__':
    libcmatrix = CDLL('./CMatrix/libcmatrix.so')
    libcmatrix.test.argtypes = [ctypes.POINTER(CMatrix)]
    libcmatrix.test.restype = CMatrix

    mat = CMatrix([(10, 2.0), (3.0, 4.0), (5.0, 6.0)])
    new_mat = libcmatrix.test(ctypes.byref(mat))
    # ret = ctypes.pointer(new_mat)
    # print(ret.row, ret.data)

    print(new_mat.row)
    print(new_mat.contents.row)
