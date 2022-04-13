import pytest
from matrix import Matrix

class TestConstructor:
    def test_zeros(self):
        mat = Matrix.zeros(1, 2)
        assert mat.to_list() == [[0.0, 0.0]]

    def test_from_const(self):
        mat = Matrix.from_const(1, 2, 3.0)
        assert mat.to_list() == [[3.0, 3.0]]

    def test_from_generator(self):
        def __gen_count():
            i = 0
            def __gen_single():
                nonlocal i
                i += 1
                return i
            return __gen_single
        mat = Matrix.from_generator(1, 2, __gen_count())
        assert mat.to_list() == [[1, 2]]

    def test_from_list(self):
        assert Matrix.from_list([[1, 2]]).to_list() == [[1, 2]]

    def test_from_diag(self):
        mat = Matrix.from_diag([1, 2])
        assert mat.to_list() == [[1, 0], [0, 2]]

class TestMethod:
    def test_shape(self):
        mat = Matrix.from_list([[1, 2]])
        assert mat.shape == (1, 2)

    def test_transpose(self):
        mat = Matrix.from_list([[1, 2]])
        assert mat.T.to_list() == [[1], [2]]

    def test_getitem(self):
        mat = Matrix.from_list([[1, 2]])
        assert mat[0] == [1, 2]
        assert mat[0, 0] == 1
        assert mat[0, 1] == 2
        with pytest.raises(Exception):
            mat[1]
        with pytest.raises(Exception):
            mat[1, 0]

    def test_setitem(self):
        mat = Matrix.from_list([[1, 2]])
        mat[0] = [2, 1]
        assert mat.to_list() == [[2, 1]]
        mat[0, 1] = 3
        assert mat.to_list() == [[2, 3]]
        with pytest.raises(Exception):
            mat[1] = [2, 1]
        with pytest.raises(Exception):
            mat[1, 0] = 3

    def test_add(self):
        mat1 = Matrix.from_list([[1.0, 2.0], [3.0, 4.0]])
        mat2 = Matrix.from_list([[2.0, 3.0], [4.0, 5.0]])
        mat3 = Matrix.from_list([[3.0, 5.0], [7.0, 9.0]])
        mat1 += mat2
        assert mat1.to_list() == mat3.to_list()
    
    def test_sub(self):
        mat1 = Matrix.from_list([[1.0, 2.0], [3.0, 4.0]])
        mat2 = Matrix.from_list([[2.0, 3.0], [4.0, 5.0]])
        mat3 = Matrix.from_list([[1.0, 1.0], [1.0, 1.0]])
        mat2 -= mat1
        assert mat2.to_list() == mat3.to_list()
    
    def test_mul_mat(self):
        mat1 = Matrix.from_list([[1.0, 2.0], [3.0, 4.0]])
        mat2 = Matrix.from_list([[2.0, 3.0], [4.0, 5.0]])
        mat3 = Matrix.from_list([[2.0, 6.0], [12.0, 20.0]])
        mat1 *= mat2
        assert mat1.to_list() == mat3.to_list()

    def test_rows(self):
        mat = Matrix.from_list([[1, 2], [3, 4]])
        assert [r for r in mat.rows()] == [[1, 2], [3, 4]]

    def test_columns(self):
        mat = Matrix.from_list([[1, 2], [3, 4]])
        assert [r for r in mat.columns()] == [[1, 3], [2, 4]]

    def test_sum(self):
        mat = Matrix.from_list([[1, 2], [3, 4]])
        assert mat.sum() == 10
        assert mat.sum(lambda x : x*2) == 20

    def test_iter(self):
        mat = Matrix.from_list([[1, 2], [3, 4]])
        assert [r for r in mat] == [1, 2, 3, 4]

    def test_apply(self):
        mat = Matrix.from_list([[1, 2], [3, 4]])
        mat = mat.apply(lambda x: x+1)
        assert mat.to_list() == [[2, 3], [4, 5]]
    
    def test_dot(self):
        mat1 = Matrix.from_list([[1, 0, -1], [2, 2, 3], [0, 1, -5]])
        mat2 = Matrix.from_list([[2, 1, 4], [0, 2, 0], [0, 3, 1]])
        mat3 = Matrix.dot(mat1, mat2)
        assert mat3.to_list() == [[2, -2, 3], [4, 15, 11], [0, -13, -5]]
        mat4 = Matrix.dot(mat2, mat1)
        assert mat4.to_list() == [[4, 6, -19], [4, 4, 6], [6, 7, 4]]

    def test_mul_val(self):
        mat = Matrix.from_list([[1, 2], [3, 4]])
        res = Matrix.mul(mat, 2)
        assert res.to_list() == [[2, 4], [6, 8]]
