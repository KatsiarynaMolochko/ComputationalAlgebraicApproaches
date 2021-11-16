import numpy as np
import sys

EPS = 10 ** -7
K_MAX = 5000


# Функция для соблюдения условия диагонального  преобладания
def calculating_sums(matrix):
    sum = 0
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if (j != i):
                sum += np.abs(matrix[i][j])
            A[i][i] = np.random.randint(sum + 7, sum + 7 * 10)

# Функция для подсчета суммы с переменными на k+1 шаге
def calculating_next (A, approx, i):
    sum = 0
    for j in range(i):
        sum += A[i][j]*approx[j]
    return sum

# Функция для подсчета суммы с переменными на k шаге
def calculating_present(A, approx, i, N):
    sum = 0
    for j in range(i + 1, N):
        sum += A[i][j]*approx[j]
    return sum


# Реализация метода минимальных невязок
def minimal_residual_method(A, F, approx, X):
    A.astype(float)
    K = 0
    print("\nMinimal residual method:\n ")
    while (K <= K_MAX):
        K += 1
        residual = np.matmul(A, approx) - F
        scalar_product = np.dot(np.matmul(A, residual), residual) / np.dot(np.matmul(A, residual),
                                                                           np.matmul(A, residual))
        approx = approx - scalar_product * residual
        crit = np.abs(np.matmul(A, approx) - F)
        index = np.argmax(crit)

        if (crit[index] < EPS):
            print("\nNumber of iterations is:", K)
            print("\nReal unknowns is: ", X)
            print("\n||AX_ - F|| = ", crit[index])
            format_string = "{:.16f}"
            print("\nVector of computed terms is:")

            for i in approx:
                print(format_string.format(i))

            sub = np.abs(X - approx)
            index_ = np.argmax(sub)
            print("\nAbsolute error is: ", sub[index_])
            return

    if (K > K_MAX):
        print("\nToo much iterations\n")
        sys.exit()

# Реализация метода релаксации
def relaxation_method(A, approx, F,  W, N):
    A.astype(float)
    K = 0
    print("\nRelaxation method: ")

    while(K <= K_MAX):
        K += 1

        for i in range(len(approx)):
            approx[i] = (1 - W) * approx[i] + W/A[i][i] * (F[i] - calculating_next(A, approx, i) - calculating_present(A, approx, i, N))


        crit = np.abs(np.matmul(A, approx) - F)
        index = np.argmax(crit)

        if (crit[index] < EPS):
            print("\nNumber of iterations is:", K)
            print("\nReal unknowns is: ", X)
            print("\n||AX_ - F|| = ", crit[index])
            format_string = "{:.16f}"
            print("\nVector of computed terms is:")

            for i in approx:
                print(format_string.format(i))

            sub = np.abs(X - approx)
            index_ = np.argmax(sub)
            print("\nAbsolute error is: ", sub[index_])
            return

    if (K > K_MAX):
        print("\nToo much iterations\n")
        sys.exit()

# Генерация симмметрической положительно определённой матрицы с диагональным преобладанием
N = int(input('Input degree of  a square matrix: '))
A = np.full((N, N), 0)
for i in range(len(A)):
    for j in range(len(A[i])):
        if (i > j):
            A[i][j] = np.random.randint(-100, 100)
        A[j][i] = A[i][j]
        calculating_sums(A)

print("\nSymmetric matrix with diagonal predominance:\n", A)

# Генерация вектора неизвестных X = (1, 2, ... , N)^T
X = np.array([0] * N)
for i in range(len(X)):
    X[i] += i + 1
print('\nVector of real unknowns: \n', X)

# Генрация столбца свободных членов F, путём умножения симметрической положительно определенной матрицы
# На вектор неизвестных X = (1, 2, ... , N)^T
F = np.matmul(A, X)
print("\n Constant terms column:\n", F)

# начальное приближение X0 = (0, 0, ... , 0)
init_approx = np.array([0.0] * N).astype(float)
print("Initial approximation: ", init_approx)

# Итерационный процесс
K = 0
Q = 0
approx = np.copy(init_approx)

# Параметры релаксации
W = 0.2



minimal_residual_method(A, F, approx, X)

relaxation_method(A,approx, F, W, N)








