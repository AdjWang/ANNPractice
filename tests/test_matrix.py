from matrix import Matrix


def test_add():
    mat1 = Matrix([[1.0, 2.0], [3.0, 4.0]])
    mat2 = Matrix([[2.0, 3.0], [4.0, 5.0]])
    print("add: ", mat1 + mat2)


def test_transpose():
    mat = Matrix([[1.0, 2.0], [3.0, 4.0]])
    print("transpose: ", mat.T)


def test_dot():
    mat1 = Matrix([[1, 0, -1], [2, 2, 3], [0, 1, -5]])
    mat2 = Matrix([[2, 1, 4], [0, 2, 0], [0, 3, 1]])
    # [[2, -2, 3], [4, 15, 11], [0, -13, -5]]
    print("dot mat1 .* mat2: ", Matrix.dot(mat1, mat2))
    # [[4, 6, -19], [4, 4, 6], [6, 7, 4]]
    print("dot mat2 .* mat1: ", Matrix.dot(mat2, mat1))
