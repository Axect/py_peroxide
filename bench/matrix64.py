from functools import reduce   # Functional Programming
from copy import deepcopy      # For Deep copy
from pandas import DataFrame   # To print matrix
from numpy.linalg import eig
from cmath import pi, exp, sin # For complex
import numpy as np             # Just for comparison (FFT) & Plotting
import scipy.fftpack as sfft   # Just for comparison (DFT)
import pylab as plt            # For Plotting
from mpl_toolkits.mplot3d import Axes3D # 3DPlot

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
        if isinstance(key, int):
            if self.nrow == 1:
                return self.data[0][key]
            elif self.ncol == 1:
                return self.dual[0][key]
        if isinstance(key, slice):
            if self.nrow == 1:
                return self.data[0][key]
            elif self.ncol == 1:
                return self.dual[0][key]
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

    def __add__(self, other):
        mat = zeros(self.nrow, self.ncol)
        for i in range(self.nrow):
            for j in range(self.ncol):
                mat[i,j] = self[i,j] + other[i,j]
        return mat

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

    def __len__(self):
        return self.nrow * self.ncol

    def transpose(self):
        return matrix(self.dual)

    def t(self):
        return self.transpose()

def from_index(f, tup):
    m = zeros(tup[0], tup[1])
    for i in range(tup[0]):
        for j in range(tup[1]):
            m[i,j] = f(i,j)
    return m

def from_col(V):
    m = zeros(len(V),1)
    for i in range(len(V)):
        m[i,0] = V[i]
    return m

def from_row(V):
    m = zeros(1, len(V))
    for i in range(len(V)):
        m[0,i] = V[i]
    return m

def eye(n):
    mat = matrix([[0 for i in range(n)] for i in range(n)])
    for i in range(n):
        mat[i,i] = 1
    return mat

def zeros(m, n):
    return matrix([[0 for i in range(n)] for i in range(m)])

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

# number x, matrix Y
def kron_each(x, Y):
    return x * Y

# array V, matrix Y
def kron_map(V, Y):
    return reduce(lambda x,y: hcat(x,y), map(lambda x: kron_each(x,Y), V))

# matrix X, matrix Y
def kron(X,Y):
    return reduce(lambda x,y: vcat(x,y), map(lambda x: kron_map(x, Y), X))

def K(n):
    mat = zeros(n,n)
    for i in range(n):
        mat[i,i] = 2
        if i < n-1:
            mat[i,i+1] = -1
            mat[i+1,i] = -1
    return mat

def K2D(n):
    k = K(n)
    i = eye(n)
    return kron(k,i) + kron(i,k)

def dft(V):
    N = V.nrow
    W = from_index(lambda k, n: exp(-2*pi*1j*k*n / N), (N, N))
    return W*V

def inv_dft(V):
    N = V.nrow
    W = from_index(lambda k, n: exp(2*pi*1j*k*n / N)/N, (N,N))
    return W*V

# Cooley and Tukey
def fft(V):
    N = V.nrow
    if N % 2 > 0:
        raise ValueError("size of V must be a power of 2")
    elif N <= 32: # Optimal number
        return dft(V)
    else:
        V_even = fft(from_col(V[::2]))
        V_odd = fft(from_col(V[1::2]))
        factor = [exp(-2j*pi*k/N) for k in range(N)]
        upper_factor = factor[:N//2]
        lower_factor = factor[N//2:]
        for i in range(len(upper_factor)):
            upper_factor[i] *= V_odd[i]
            lower_factor[i] *= V_odd[i]
        return vcat(V_even + from_col(upper_factor), V_even + from_col(lower_factor))

def y(N, tup):
    m = zeros(N, N)
    k = tup[0]
    l = tup[1]
    for i in range(N):
        for j in range(N):
            m[i,j] = sin(i*k*pi/(N+1)) * sin(j*l*pi/(N+1))
    return m

# Discrete Sine Transform based on DFT
def dst(V):
    m = V.nrow
    Z = zeros(2*m + 2, 1)
    for i in range(m):
        Z[i+1, 0] = V[i,0]
        Z[2*m+1-i,0] = -V[i,0]
    F = dft(Z)
    S = zeros(m, 1)
    for i in range(m):
        S[i,0] = 0.5j * F[i+1,0]
    return S

# Fast Sine Transform based on FFT
def fst(V):
    m = V.nrow
    Z = zeros(2*m + 2, 1)
    for i in range(m):
        Z[i+1, 0] = V[i,0]
        Z[2*m+1-i,0] = -V[i,0]
    F = fft(Z)
    S = zeros(m, 1)
    for i in range(m):
        S[i,0] = 0.5j * F[i+1,0]
    return S

# Simple Fast Possiong Solver (Not based on FFT)
def simple_fps(f, N):
    h = 1 / (N+1)
    x = [i*h for i in range(1,N+1)]
    y = deepcopy(x)
    F = zeros(N,N)
    S = zeros(N,N)
    sigma = [0]*N
    for (i,a) in enumerate(x):
        sigma[i] = sin(a*pi/2)**2
        for (j,b) in enumerate(y):
            F[i,j] = f(a,b)
            S[i,j] = sin(i*j*pi*h)
    G = S * F * S
    X = zeros(N,N)
    for j in range(N):
        for k in range(N):
            X[j,k] = h**4 * G[j,k] / (sigma[j]+sigma[k])
    return S * X * S

def fps(f, N):
    h = 1 / (N+1)
    x = [i*h for i in range(1,N+1)]
    y = deepcopy(x)
    F = zeros(N,N)
    sigma = [0]*N
    for (i,a) in enumerate(x):
        sigma[i] = sin(a*pi/2)**2
        for (j,b) in enumerate(y):
            F[i,j] = f(a,b)
    G1 = fst(F)
    G = fst(G1.transpose()).transpose()
    X = zeros(N,N)
    for j in range(N):
        for k in range(N):
            X[j,k] = h**4 * G[j,k] / (sigma[j]+sigma[k])
    V1 = fst(X)
    return fst(V1.transpose()).transpose()

# For FPS
def frhs(x,y):
    return 1

## N=16 Simple
#x1 = [i/17 for i in range(1,17)]
#y1 = deepcopy(x1)
#X1, Y1 = np.meshgrid(x1, y1)
#Z1 = np.matrix(simple_fps(frhs, 16).data)

## N=32 Simple
#fig = plt.figure(figsize=(10, 6), dpi=300)
#ax = fig.gca(projection='3d')
#x1 = [i/33 for i in range(1,33)]
#y1 = deepcopy(x1)
#X1, Y1 = np.meshgrid(x1, y1)
#Z1 = np.matrix(simple_fps(frhs, 32).data)
#
#surf = ax.plot_surface(X1,Y1,Z1)
#plt.title(r"Diagonalization (N=32)", fontsize=16)
#plt.savefig("diag_n32.png")
#
## N=64 Simple
#fig = plt.figure(figsize=(10, 6), dpi=300)
#ax = fig.gca(projection='3d')
#x1 = [i/65 for i in range(1,65)]
#y1 = deepcopy(x1)
#X1, Y1 = np.meshgrid(x1, y1)
#Z1 = np.matrix(simple_fps(frhs, 64).data)
#
#surf = ax.plot_surface(X1,Y1,Z1)
#plt.title(r"Diagonalization (N=64)", fontsize=16)
#plt.savefig("diag_n64.png")
#
# FFT Based
## N = 15 FFT
#x2 = [i/16 for i in range(1,16)]
#y2 = deepcopy(x2)
#X2, Y2 = np.meshgrid(x2, y2)
#Z2 = np.matrix(simple_fps(frhs, 15).data)
#
## N = 31 FFT
#x2 = [i/32 for i in range(1,32)]
#y2 = deepcopy(x2)
#X2, Y2 = np.meshgrid(x2, y2)
#Z2 = np.matrix(simple_fps(frhs, 31).data)
#
# N = 63 FFT
x2 = [i/64 for i in range(1,64)]
y2 = deepcopy(x2)
X2, Y2 = np.meshgrid(x2, y2)
Z2 = np.matrix(simple_fps(frhs, 63).data)

## N = 127 FFT
#x2 = [i/128 for i in range(1,128)]
#y2 = deepcopy(x2)
#X2, Y2 = np.meshgrid(x2, y2)
#Z2 = np.matrix(simple_fps(frhs, 127).data)
#

## DFT & FFT Test
#x_sample = np.random.rand(16)
#x = from_col(x_sample)
#X2 = fft(x)
#print(np.allclose(X2.col(0), np.fft.fft(x_sample)))
#
## DST Test
#print(np.allclose(dst(x).col(0), 0.5*sfft.dst(x_sample, type=1)))
#
## FFT Test
#x2 = np.random.random(1024)
#print(np.allclose(fft(from_col(x2)).col(0), np.fft.fft(x2)))
#
## FST Test
#x3_sample = np.random.rand(511)
#prsint(np.allclose(fst(from_col(x3_sample)).col(0), 0.5*sfft.dst(x3_sample, type=1)))
