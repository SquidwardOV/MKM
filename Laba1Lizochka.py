import numpy as np
import matplotlib.pyplot as plt

def solve_bessel(p_values, step, a, b, v0, dv0):
    """
    Численное решение уравнения Бесселя методом конечных разностей на отрезке [a, b].
    Уравнение: x^2 v''(x) + x v'(x) + (x^2 - p^2)v(x) = 0.
    """
    results = {}
    grid = np.linspace(a, b, int((b - a) / step) + 1)
    N = grid.size

    for p in p_values:
        v = np.zeros(N)
        v[0] = v0
        v[1] = v0 + step * dv0  # Линейное приближение для первого шага

        for i in range(1, N - 1):
            xi = grid[i]
            A = xi**2 / step**2
            B = xi / (2 * step)
            C = xi**2 - p**2
            if abs(A + B) < 1e-14:
                v[i+1] = v[i]
            else:
                v[i+1] = (2 * A * v[i] - C * v[i] - (A - B) * v[i-1]) / (A + B)

        results[p] = v

    return grid, results

def solve_singular_bessel(gamma, step, a, b, v0, dv0):
    """
    Численное решение сингулярного уравнения Бесселя методом конечных разностей на отрезке [a, b].
    Уравнение: x v''(x) + gamma v'(x) + x v(x) = 0.
    """
    grid = np.linspace(a, b, int((b - a) / step) + 1)
    N = grid.size
    v = np.zeros(N)
    v[0] = v0
    v[1] = v0 + step * dv0  # Линейное приближение для первого шага

    for i in range(1, N - 1):
        xi = grid[i]
        P = xi / step**2
        Q = gamma / (2 * step)
        R = xi
        if abs(P + Q) < 1e-14:
            v[i+1] = v[i]
        else:
            v[i+1] = (2 * P * v[i] - R * v[i] - (P - Q) * v[i-1]) / (P + Q)

    return grid, v

def main():
    try:
        step = float(input("Введите шаг (например, 0.01): "))
    except ValueError:
        print("Ошибка: введён некорректный шаг.")
        return

    try:
        gamma = float(input("Введите значение gamma (не 0): "))
        if gamma == 0:
            print("Ошибка: gamma не должно равняться 0.")
            return
    except ValueError:
        print("Ошибка: введено некорректное значение gamma.")
        return

    # Интервал [0, 1] и начальные условия
    a, b = 0.0, 1.0
    v0, dv0 = 1.0, 0.0

    # Значения параметра p для уравнения Бесселя
    p_values = [0.1, 0.2, 0.3, 0.4]

    # Решение уравнения Бесселя
    x_bessel, bessel_results = solve_bessel(p_values, step, a, b, v0, dv0)

    # Решение сингулярного уравнения Бесселя
    x_singular, singular_solution = solve_singular_bessel(gamma, step, a, b, v0, dv0)

    # Построение графиков
    plt.figure(figsize=(12, 5))

    # График для уравнения Бесселя
    plt.subplot(1, 2, 1)
    for p, v in bessel_results.items():
        plt.plot(x_bessel, v, label=f'p = {p}')
    plt.xlabel("x")
    plt.ylabel("v(x)")
    plt.title("Численное решение уравнения Бесселя")
    plt.legend()
    plt.grid(True)

    # График для сингулярного уравнения Бесселя
    plt.subplot(1, 2, 2)
    plt.plot(x_singular, singular_solution, label=f'gamma = {gamma}', color='r')
    plt.xlabel("x")
    plt.ylabel("v(x)")
    plt.title("Численное решение сингулярного уравнения Бесселя")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
