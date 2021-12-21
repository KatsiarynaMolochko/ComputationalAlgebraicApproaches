import numpy as np
import sympy as sp
from math import cos, pi, sin, sqrt

EPS = 10 ** (-7)
N = int(input('Degree of a matrix: '))

# Генерация диагонализируемой матрицы
A = np.full((N, N), 0)
for i in range(len(A)):
    A[i][i] = np.random.randint(-100, 100)
    for j in range(len(A[i])):
        if (i > j):
            A[i][j] = np.random.randint(-100, 100)
        A[j][i] = A[i][j]
print(A)

# Генерация начального вектора
Y_ = np.array([0] * N)
Y_[0] = 1.0
print("Initial vector y: ", Y_)

# Нормированный начальный вектор
U_ = Y_/np.linalg.norm(Y_)
print("Initial norm vector", U_)


def power_iteration_method(A_, Y_, U_):
    K = 0
    Y_ = np.matmul(A_, U_)
    lam = np.dot(Y_, U_)
    U_ = Y_ / np.linalg.norm(Y_)
    crit = np.matmul(A_, U_) - lam * U_

    while(np.linalg.norm(crit) > EPS):
        K = K+1
        Y_ = np.matmul(A_, U_)
        lam = np.dot(Y_, U_)
        U_ = Y_ / np.linalg.norm(Y_)
        crit = np.matmul(A_, U_) - lam * U_

    print("Maximal eigenvalue:", lam)
    print("Result vector:", U_)
    print("Number of itertions:", K)
    print("Test:", crit)
    print("Test norm:", np.linalg.norm(crit))

def find_max(A_):
    mat = np.abs(np.copy(A_))
    for i in range(len(mat)):
        mat[i][i] = 0
        for j in range(len(mat[i])):
            if(i > j):
                mat[i][j] = 0

    # print(mat)
    indecies = np.unravel_index(np.argmax(mat), mat.shape)
    # print("Indecies", indecies)
    return indecies


def sum_of_non_diagonal(A):
    sum = 0
    for i in range(A.shape[0]):
        for j in range(i + 1, A.shape[0]):
            sum += A[i, j] ** 2
    return 2 * sum

def calculate_trig_functions(matrix, row, column):
    if matrix[row, row] == matrix[column, column]:
        return sp.cos(pi/4), sp.sin(pi/4)
    mu = 2*matrix[row, column] / (matrix[row, row] - matrix[column, column])
    tmp = 1/sqrt(1+mu**2)
    mult = -1
    if mu > 0:
        mult *= -1
    return sqrt(0.5*(1+tmp)), mult * sqrt(0.5*(1-tmp))


def jacobi_method(A_):

    number_of_iterations = 0
    A_k = np.copy(A_)
    T = np.matrix(np.diag([1.0 for i in range(A_.shape[0])])).astype(float)
    while (sum_of_non_diagonal(A_k) > EPS):
        number_of_iterations += 1
        indecies = find_max(A_k)
        k = indecies[0]
        l = indecies[1]
        I_c = A_[0:][k]   # i column
        I_r = A_[k][0:]   # i row
        J_c = A_[0:][l]   # j column
        J_r = A_[l][0:]   # j row

        cos, sin = calculate_trig_functions(A_k, k, l)

        # only columns with indexes k,l will change B = AkTkl
        B = np.copy(A_k)
        T_tmp = np.copy(T)
        for i in range(A_.shape[0]):
            B[i, k] = A_k[i, k] * cos + A_k[i, l] * sin
            B[i, l] = A_k[i, l] * cos - A_k[i, k] * sin

            T_tmp[i, k] = T[i, k] * cos + T[i, l] * sin
            T_tmp[i, l] = T[i, l] * cos - T[i, k] * sin
        T = np.copy(T_tmp)
        # C = T^-1klB
        # only rows k,l will change
        C = np.copy(B).astype(float)
        for i in range(A_.shape[0]):
            C[k, i] = B[k, i] * cos + B[l, i] * sin
            C[l, i] = B[l, i] * cos - B[k, i] * sin

        A_k = np.copy(C)

    print("Eigenvalues", np.diagonal(A_k).astype(float))
    print("Number of iterations", number_of_iterations)
    print("Matrix T", T.T)


    return np.diagonal(A_k), T.T, number_of_iterations








power_iteration_method(A, Y_, U_)
jacobi_method(A)
jacobi_result = jacobi_method(A)
print(f"number of iterations: {jacobi_result[2]}\n")
for i in range(A.shape[0]):
    print(f"eigenvalue: {jacobi_result[0][i]}")
    print(f"corresponding eigenvector:\n{jacobi_result[1][i]}")
    print(f"the residual r_i: {np.subtract(np.matmul(A, jacobi_result[1][i]), jacobi_result[0][i] *jacobi_result[1][i] )}\n")

