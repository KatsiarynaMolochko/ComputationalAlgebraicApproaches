import numpy
import numpy as np




def max_in_column(matrix, degree, terms):
    step = np.abs(degree - len(matrix))
    matrix_ = np.abs(matrix)
    max_i = np.argmax(matrix_[step:, step]) + step

    if matrix[max_i, step] == 0:
        print("Modulo maximum element in column is zero. Gaussian method can't be applied.")
        return

    if max_i == step:
        return matrix[step, step]

    if max_i != step:
        matrix[step, :], matrix[max_i, :] = np.copy(matrix[max_i, :]), np.copy(matrix[step, :])
        terms[step], terms[max_i] = np.copy(terms[max_i]), np.copy(terms[step])
        maximal = matrix[step, step]
        return maximal


# деление первой строки подматрицы на первый элемент
def division(matrix, degree, terms):
    step = np.abs(np.copy(degree) - len(matrix))
    divider = max_in_column(matrix, degree, terms)
    float(divider)
    for i in range(step, len(matrix[degree - 1])):
        matrix[step, i] /= divider
    terms[step] /= divider


# умножение строки на 1-ый элемент каждой строки подматрицы
def mult_sub(matrix, degree, terms):
    # и вычитание из каждой строки первой домноженной
    step = np.abs(np.copy(degree) - len(matrix))
    for i in range(step + 1, len(matrix)):
        multiplier = matrix[i, step]
        subtracted = np.multiply(matrix[step, step:], multiplier)
        subtracted_t = terms[step] * multiplier
        terms[i] -= subtracted_t
        matrix[i, step:] = matrix[i, step:] - subtracted


# прямой ход метода Гаусса
def gaussian_elimination(matrix, degree, terms):
    for k in range(degree, 1, -1):
        max_in_column(matrix, k, terms)
        division(matrix, k, terms)
        mult_sub(matrix, k, terms)
    division(matrix, 1, terms)


# обратный ход
def forward_elimination(matrix, degree, terms):
    X_ = np.array([0.0] * degree)
    X_[degree - 1] = terms[degree - 1]
    for i in range(degree - 2, -1, -1):
        sum = 0
        for j in range(degree - 1, 0, -1):
            sum += matrix[i, j] * X_[j]
        X_[i] = terms[i] - np.copy(sum)

    return X_


def inverted_matrix(matrix, degree):
    E = np.diag([1 for i in range(degree)]).astype(float)


    wide = np.hstack((matrix, E)).astype(float)


    for i in range(degree, 0, -1):
        max_in_column_i(wide, i)

        for j in range(0, degree):
            if wide[j, j] != 1.:

                wide[j, :] *= float(1.0 / wide[j, j])

            for row in range(j + 1, degree):
                wide[row, :] -= wide[j, :] * wide[row, j]

        for i in range(degree - 1, 0, -1):

            for row in range(i - 1, -1, -1):
                if (wide[row, i]):
                    wide[row, :] -= wide[i, :] * wide[row, i]
    return np.hsplit(wide, degree // 2)[1]


def max_in_column_i(wide, degree):
    step = np.abs(degree - wide.shape[0])
    matrix_ = np.abs(wide)
    max_i = np.argmax(matrix_[step:, step]) + step

    if wide[max_i, step] == 0:
        print("Modulo maximum element in column is zero. Gaussian method can't be applied.")
        return

    if max_i == step:
        return wide[step, step]

    if max_i != step:
        wide[step, :], wide[max_i, :] = np.copy(wide[max_i, :]), np.copy(wide[step, :])
        maximal = wide[step, step]
        return maximal


k = int(input('Degree of a matrix: '))
# создание квадратной матрицы размера k согласно условию
A = np.random.randint(-100, 100, (k, k)).astype(float)
print('\nSystem matrix is: \n', A)

# создание вектора неизвестных
X = np.array([0] * k)
for i in range(len(X)):
    X[i] += i + 1
print('\nVector of real unknowns: \n', X)

# перемножение вектора неизвестных и матрицы для получения свободных членов
B = numpy.matmul(A, X)
print('\nConstant terms column: \n', B)

gaussian_elimination(A, k, B)
X_ = forward_elimination(A, k, B)
print("\nVector of computed unknowns ", X_)
format_string = "{:.16f}"
for i in X_:
    print(format_string.format(i))

# вычисление максимум-норм векторов
v_norm_sub = np.array(np.abs(np.subtract(X, X_)))
v_norm_unk = np.abs(X)

# вычиление погрешности
uncertainty = v_norm_sub[np.argmax(v_norm_sub)] / v_norm_unk[np.argmax(v_norm_unk)]

print("\nUncertainty is", uncertainty)

inver = np.random.randint(-100, 100, (k, k))
print('\n', inver)
inversed = inverted_matrix(inver, k)
print(inversed)
print(np.matmul(inver, inversed))
