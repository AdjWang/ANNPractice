from matrix import Matrix
from nnlayer import FullConnection, softmax

def test_fc():
    fc_layer = FullConnection(3, 6, 0.001)
    print(fc_layer)

def test_softmax():
    mat = Matrix.from_list([[0.6, -0.7]]).T
    print(softmax(mat))