from functools import reduce
from copy import deepcopy
from pandas import DataFrame

class matrix:
    def __init__(self, array_of_array):
        self.data = array_of_array
        self.dual = dual_create(array_of_array)
        self.nrow = len(self.data)
        self.ncol = len(self.dual)

    def __setitem__(self, key, val):
        self.data[key[0]][key[1]] = val
        self.dual[key[1]][key[0]] = val

    def __getitem__(self, key):
        return self.data[key[0]][key[1]]

    def __repr__(self):
        return repr(DataFrame(self.data))

    def row(self, i):
        return self.data[i]

    def col(self, i):
        return self.dual[i]

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= self.nrow:
            raise StopIteration
        v = self.data[self.index]
        self.index += 1
        return v

    def __mul__(self, other):
        mat = [[0 for i in range(other.ncol)] for i in range(self.nrow)]
        for (i, row) in enumerate(self.data):
            for (j, col) in enumerate(other.dual):
                mat[i][j] = sum([x*y for (x,y) in zip(row, col)])
        return matrix(mat)

    def __rmul__(self, other):
        mat = [[0 for i in range(self.ncol)] for i in range(self.nrow)]
        for i in range(self.nrow):
            for j in range(self.ncol):
                mat[i][j] = other * self[i,j]
        return matrix(mat)

    def transpose(self):
        return matrix(self.dual)

    def t(self):
        return self.transpose()

def eye(n):
    mat = matrix([[0 for i in range(n)] for i in range(n)])
    for i in range(n):
        mat[i,i] = 1
    return mat

def hcat(x, y):
    a = deepcopy(x.dual)
    b = deepcopy(y.dual)
    a.extend(b)
    return matrix(dual_create(a))

def vcat(x, y):
    a = deepcopy(x.data)
    b = deepcopy(y.data)
    a.extend(b)
    return matrix(a)
        
def dual_create(array_of_array):
    r = len(array_of_array)
    c = len(array_of_array[0])
    copy_mat = [[0 for _i in range(r)] for _i in range(c)]
    for i in range(r):
        for j in range(c):
            copy_mat[j][i] = array_of_array[i][j]
    return copy_mat

m = matrix([[2,-1,0,0],[-1,2,-1,0],[0,-1,2,-1],[0,0,-1,2]])
n = eye(4)

def kron_each(x, Y):
    return x * Y

def kron_map(V, Y):
    return reduce(lambda x,y: hcat(x,y), map(lambda x: kron_each(x,Y), V))

def kron(X,Y):
    return reduce(lambda x,y: vcat(x,y), map(lambda x: kron_map(x, Y), X))

print(kron(m,n))
