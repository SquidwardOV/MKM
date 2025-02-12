import numpy as np
import matplotlib.pyplot as plt

def solve_bessel(p, h, x0, xf, v0, dv0):
    """Численное решение уравнения Бесселя методом конечных разностей."""
    x = np.arange(x0, xf + h, h)
    n = len(x)
    v = np.zeros(n)
    v[0] = v0

    # Вычисляем v''(2) из уравнения Бесселя
    vpp0 = -(x0 * dv0 + (x0**2 - p**2) * v0) / (x0**2)
    v[1] = v0 + h * dv0 + (h**2 / 2) * vpp0

    for i in range(1, n - 1):
        xi = x[i]
        coef1 = xi**2 / h**2
        coef2 = xi / (2 * h)
        coef3 = xi**2 - p**2

        v[i + 1] = (2 * coef1 * v[i] - coef3 * v[i] - (coef1 - coef2) * v[i - 1]) / (coef1 + coef2)

    return x, v

def solve_singular_bessel(gamma, h, x0, xf, v0, dv0):
    """Численное решение сингулярного уравнения Бесселя методом конечных разностей."""
    x = np.arange(x0, xf + h, h)
    n = len(x)
    v = np.zeros(n)
    v[0] = v0

    # Вычисляем v''(2) из уравнения
    vpp0 = -(gamma * dv0 + x0 * v0) / (x0**2)
    v[1] = v0 + h * dv0 + (h**2 / 2) * vpp0

    for i in range(1, n - 1):
        xi = x[i]
        coef1 = xi / h**2
        coef2 = gamma / (2 * h)
        coef3 = xi

        v[i + 1] = (2 * coef1 * v[i] - coef3 * v[i] - (coef1 - coef2) * v[i - 1]) / (coef1 + coef2)

    return x, v

def main():
    print("Выберите уравнение для решения:")
    print("1: Уравнение Бесселя")
    print("2: Сингулярное уравнение Бесселя")
    choice = input("Введите 1 или 2: ").strip()

    try:
        h = float(input("Введите шаг h (например, 0.01): "))
    except ValueError:
        print("Некорректное значение шага. Завершаем программу.")
        return

    x0, xf = 2, 4
    v0, dv0 = 2, 0

    if choice == "1":
        ps = [1, 2, 3, 4]
        plt.figure(figsize=(8, 6))
        for p in ps:
            x, v = solve_bessel(p, h, x0, xf, v0, dv0)
            plt.plot(x, v, label=f'p = {p}')
        plt.xlabel("x")
        plt.ylabel("v(x)")
        plt.title("Численное решение уравнения Бесселя")
        plt.legend()
        plt.grid(True)
        plt.show()

    elif choice == "2":
        try:
            gamma = float(input("Введите значение gamma (не 0): "))
            if gamma == 0:
                print("gamma не должно быть равно 0.")
                return
        except ValueError:
            print("Некорректное значение gamma.")
            return

        x, v = solve_singular_bessel(gamma, h, x0, xf, v0, dv0)
        plt.figure(figsize=(8, 6))
        plt.plot(x, v, label=f'gamma = {gamma}')
        plt.xlabel("x")
        plt.ylabel("v(x)")
        plt.title("Численное решение сингулярного уравнения Бесселя")
        plt.legend()
        plt.grid(True)
        plt.show()

    else:
        print("Неверный выбор. Завершаем программу.")

if __name__ == "__main__":
    main()
