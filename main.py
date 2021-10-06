
import numpy as np


def solve_tridiagonal_matrix(_A, _B, _C, _F):
    degree = _F.shape[0]
    alpha = np.zeros(degree)
    beta = np.zeros(degree)

    alpha[0] = _B[0] / _C[0]
    beta[0] = _F[0] / _C[0]
    for i in range(1, degree):
        tmp = _C[i] - _A[i] * alpha[i - 1]
        alpha[i] = _B[i]/tmp
        beta[i] = (_F[i] + _A[i] * beta[i - 1])/tmp
    beta[-1] = (_F[-1] + _A[-1] * beta[-2]) / (_C[-1] - _A[-1] * alpha[-2])

    # back substitution
    solution = np.zeros(degree)
    solution[-1] = beta[-1]
    for i in range(degree - 2, -1, -1):
        solution[i] = alpha[i] * solution[i + 1] + beta[i]

    return solution


k = int(input("Input degree of a matrix: "))
A = np.random.randint(-100, 100, k)
A[0] = 0
A *= -1
print("Low diagonal \n", A)
B = np.random.randint(-100, 100, k)
B *= -1
B[k - 1] = 0
print("High diagonal \n", B)
C = np.zeros(k)

for i in range(1, k - 1):
    C[i] = np.random.randint(abs(A[i]) + abs(B[i]) + 7, abs(A[i]) + abs(B[i]) + 14)

C[0] = np.random.randint(abs(B[0]) + 7, abs(B[0]) + 14)
C[- 1] = np.random.randint(abs(A[- 1]) + 7, abs(A[- 1]) + 14)
print("Main diagonal \n", C)

Y = np.zeros(k)  ##создание вектора неизвестных

for i in range(len(Y)):
    Y[i] += i + 1

print('\nVector of real unknowns: \n', Y)

F = np.zeros(k)                                             ##генерация столбца свободных членов
F[-1] = (-1) * A[-1] * Y[-2] + C[-1] * Y[-1]
F[0] = C[0] * Y[0] - B[0] * Y[1]

for i in range(1, k - 1):
    F[i] = (-1) * A[i] * Y[i-1] + C[i]*Y[i] - B[i]*Y[i+1]

print('\nVector of constant terms: \n', F)

Y_ = solve_tridiagonal_matrix(A, B, C, F)
print("\nVector of computed unknowns: ")
format_string = "{:.16f}"
for i in Y_:
    print(format_string.format(i))

# вычисление максимум-норм векторов
v_norm_sub = np.array(np.abs(np.subtract(Y, Y_)))
v_norm_unk = np.abs(Y)

# вычиление погрешности
uncertainty = v_norm_sub[np.argmax(v_norm_sub)] / v_norm_unk[np.argmax(v_norm_unk)]
print("\nUncertainty is", format_string.format(uncertainty))
