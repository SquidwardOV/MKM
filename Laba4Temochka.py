import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def input_float(prompt, default):
    """Запрашивает ввод числа с плавающей запятой, если ввод пуст – возвращает default."""
    inp = input(f"{prompt} (по умолчанию {default}): ")
    return float(inp) if inp.strip() != "" else default


def exact_solution_task1(t, r, K, y0):
    """
    Точное решение логистического уравнения Ферхюльста (задача 1):
    dy/dt = r*y - (r/K)*y^2,  y(0)=y0.
    Формула: y(t) = K / (1 + ((K-y0)/y0)*exp(-r*t))
    """
    return K / (1 + ((K - y0) / y0) * np.exp(-r * t))


def exact_solution_task2(t, r, y0):
    """
    Точное решение уравнения (задача 2):
    dy/dt = r*y^2,  y(0)=y0.
    Решение: y(t) = 1 / ((1/y0) - r*t)
    ВНИМАНИЕ: При t >= 1/(r*y0) решение "взрывается".
    """
    return 1.0 / ((1.0 / y0) - r * t)


def f_task1(t, y, params):
    """
    Правая часть уравнения (задача 1):
    dy/dt = r*y - (r/K)*y^2
    params = (r, K)
    """
    r, K = params
    return r * y - (r / K) * y ** 2


def f_task2(t, y, params):
    """
    Правая часть уравнения (задача 2):
    dy/dt = r*y^2
    params = (r,)
    """
    (r,) = params
    return r * y ** 2


def f_task3(t, y, params):
    """
    Правая часть уравнения (задача 3 – модель с хищником):
    dy/dt = R*y*(1 - y/K) - B*P*(y^2/(A^2 + y^2))
    params = (R, K, B, P, A)
    """
    R, K, B, P, A = params
    return R * y * (1 - y / K) - B * P * (y ** 2 / (A ** 2 + y ** 2))


def euler_method(f, y0, t0, t_end, h, params):
    """
    Универсальная функция для метода Эйлера.
    f(t, y, params) - функция правой части ДУ,
    y0 - начальное условие,
    t0, t_end - границы интервала,
    h - шаг,
    params - кортеж параметров.

    Возвращает массив времени T и массив численного решения Y.
    """
    n_steps = int((t_end - t0) / h)
    T = np.linspace(t0, t_end, n_steps + 1)
    Y = np.zeros(n_steps + 1)
    Y[0] = y0

    for i in range(n_steps):
        Y[i + 1] = Y[i] + h * f(T[i], Y[i], params)

    return T, Y


def main():
    print("Выберите номер задачи:")
    print("1) Логистическое уравнение Ферхюльста")
    print("2) Уравнение dy/dt = r*y²")
    print("3) Модель с хищником: dy/dt = R*y*(1 - y/K) - B*P*(y²/(A² + y²))")
    choice = input("Введите 1, 2 или 3: ")

    if choice == '1':
        print("\nРешаем задачу 1 (логистическое уравнение Ферхюльста).")
        y0 = input_float("Введите начальное значение y0", 10.0)
        r = input_float("Введите значение r", 0.5)
        K = input_float("Введите значение K (емкость)", 1000.0)
        t0 = 0.0
        t_end = input_float("Введите время моделирования t_end", 10.0)
        h = input_float("Введите шаг интегрирования h", 0.1)

        # Точное решение
        def exact(t):
            return exact_solution_task1(t, r, K, y0)

        T, Y_euler = euler_method(f_task1, y0, t0, t_end, h, (r, K))
        Y_exact = exact(T)
        abs_err = np.abs(Y_exact - Y_euler)
        rel_err = abs_err / Y_exact

        step_for_print = int(1.0 / h)
        indices = np.arange(0, len(T), step_for_print)
        df = pd.DataFrame({
            "t": T[indices],
            "Exact": Y_exact[indices],
            "Euler": Y_euler[indices],
            "AbsErr": abs_err[indices],
            "RelErr": rel_err[indices]
        })
        print("\nРезультаты (вывод с шагом 1):")
        print(df)

        plt.figure()
        plt.plot(T, Y_exact, label="Точное решение")
        plt.plot(T, Y_euler, linestyle="--", label="Метод Эйлера")
        plt.xlabel("t")
        plt.ylabel("y(t)")
        plt.title("Задача 1: Логистическое уравнение Ферхюльста")
        plt.legend()
        plt.grid(True)
        plt.show()

    elif choice == '2':
        print("\nРешаем задачу 2 (dy/dt = r*y²).")
        # Регулируемые параметры
        y0 = input_float("Введите начальное значение y0", 1.0)
        r = input_float("Введите значение r", 0.05)
        t0 = 0.0
        t_end = input_float("Введите время моделирования t_end", 15.0)
        h = input_float("Введите шаг интегрирования h", 0.1)

        blow_up_time = 1.0 / (r * y0)
        if t_end >= blow_up_time:
            print(
                f"Внимание: время моделирования t_end должно быть меньше {blow_up_time:.3f}, иначе решение будет расходиться.")
            print("Используйте t_end меньше времени 'взрыва' (blow-up time).")
            return

        def exact(t):
            return exact_solution_task2(t, r, y0)

        T, Y_euler = euler_method(f_task2, y0, t0, t_end, h, (r,))
        Y_exact = exact(T)
        abs_err = np.abs(Y_exact - Y_euler)
        rel_err = abs_err / Y_exact

        step_for_print = int(1.0 / h)
        indices = np.arange(0, len(T), step_for_print)
        df = pd.DataFrame({
            "t": T[indices],
            "Exact": Y_exact[indices],
            "Euler": Y_euler[indices],
            "AbsErr": abs_err[indices],
            "RelErr": rel_err[indices]
        })
        print("\nРезультаты (вывод с шагом 1):")
        print(df)

        plt.figure()
        plt.plot(T, Y_exact, label="Точное решение")
        plt.plot(T, Y_euler, linestyle="--", label="Метод Эйлера")
        plt.xlabel("t")
        plt.ylabel("y(t)")
        plt.title("Задача 2: dy/dt = r*y²")
        plt.legend()
        plt.grid(True)
        plt.show()

    elif choice == '3':
        print("\nРешаем задачу 3 (модель с хищником).")
        # Регулируемые параметры для модели с хищником
        y0 = input_float("Введите начальное значение y0", 50.0)
        R = input_float("Введите значение R (максимальная скорость роста)", 0.3)
        K = input_float("Введите значение K (емкость среды)", 500.0)
        B = input_float("Введите значение B (максимальная скорость хищничества)", 0.02)
        P = input_float("Введите значение P (количество хищников)", 5.0)
        A = input_float("Введите значение A (объём популяции для половинной скорости хищничества)", 50.0)
        t0 = 0.0
        t_end = input_float("Введите время моделирования t_end", 50.0)
        h = input_float("Введите шаг интегрирования h", 0.1)

        # В данной задаче аналитического решения нет – только численное решение методом Эйлера.
        T, Y_euler = euler_method(f_task3, y0, t0, t_end, h, (R, K, B, P, A))

        step_for_print = int(1.0 / h)
        indices = np.arange(0, len(T), step_for_print)
        df = pd.DataFrame({
            "t": T[indices],
            "Euler": Y_euler[indices]
        })
        print("\nРезультаты (вывод с шагом 1):")
        print(df)

        plt.figure()
        plt.plot(T, Y_euler, label="Метод Эйлера (приближённое решение)")
        plt.xlabel("t")
        plt.ylabel("y(t)")
        plt.title("Задача 3: Модель с хищником")
        plt.legend()
        plt.grid(True)
        plt.show()

    else:
        print("Неверный выбор. Пожалуйста, перезапустите программу и введите 1, 2 или 3.")


if __name__ == "__main__":
    main()
