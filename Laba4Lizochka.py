import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def euler_logistic(y0, r, K, t0, t_end, h):
    """
    Метод Эйлера для логистического уравнения:
      dy/dt = r*y - (r/K)*y^2
    Возвращает вектор времени, численное решение и точное решение.
    """
    delta = r / K
    n_steps = int((t_end - t0) / h)
    t = np.linspace(t0, t_end, n_steps + 1)
    y_num = np.zeros(n_steps + 1)
    y_num[0] = y0
    for i in range(n_steps):
        y_num[i + 1] = y_num[i] + h * (r * y_num[i] - delta * y_num[i] ** 2)
    y_exact = K / (1 + ((K - y0) / y0) * np.exp(-r * (t - t0)))
    return t, y_num, y_exact


def euler_quad(y0, r, t0, t_end, h):
    """
    Метод Эйлера для уравнения:
      dy/dt = r*y^2
    Аналитическое решение: y(t) = 1/((1/y0) - r*t)
    """
    n_steps = int((t_end - t0) / h)
    t = np.linspace(t0, t_end, n_steps + 1)
    y_num = np.zeros(n_steps + 1)
    y_num[0] = y0
    for i in range(n_steps):
        y_num[i + 1] = y_num[i] + h * (r * y_num[i] ** 2)
    y_exact = 1.0 / ((1.0 / y0) - r * (t - t0))
    return t, y_num, y_exact


def euler_predator(y0, R, K, B, P, A, t0, t_end, h):
    """
    Метод Эйлера для модели с хищником:
      dy/dt = R*y*(1 - y/K) - B*P*(y^2/(A^2+y^2))
    Аналитического решения нет, поэтому возвращаем только численное решение.
    """
    n_steps = int((t_end - t0) / h)
    t = np.linspace(t0, t_end, n_steps + 1)
    y_num = np.zeros(n_steps + 1)
    y_num[0] = y0
    for i in range(n_steps):
        y_num[i + 1] = y_num[i] + h * (
            R * y_num[i] * (1 - y_num[i] / K)
            - B * P * (y_num[i]**2 / (A**2 + y_num[i]**2))
        )
    return t, y_num


# Основной блок с меню
variant = input("Введите номер задания: 1(Логистическое уравнение Ферхюльста), 2(Уравнение dy/dt=ry^2), 3(Модель с хищником): ")

if variant == "1":
    # Вариант 1: логистическое уравнение для соболей
    # t0 = 0 соответствует 1950 году, t_end = 20 лет (до 1970 года)
    t0 = 0
    t_end = 20
    y0 = 1120
    r = 1.39
    K = 3459
    h = 0.1

    t, y_num, y_exact = euler_logistic(y0, r, K, t0, t_end, h)

    # Построение графика (синий цвет для точного, красный для Эйлера)
    plt.figure(figsize=(10, 6))
    plt.plot(t + 1950, y_exact, 'b-', label="Точное решение")
    plt.plot(t + 1950, y_num, 'r--', label="Метод Эйлера")
    plt.xlabel("Год")
    plt.ylabel("Численность соболей")
    plt.title("Динамика численности соболей (Вариант 1)")
    plt.legend()
    plt.grid(True)
    plt.show()

    sample_indices = np.arange(0, len(t), 10)
    results_table = pd.DataFrame({
        "Год": t[sample_indices] + 1950,
        "Точное решение": y_exact[sample_indices],
        "Метод Эйлера": y_num[sample_indices],
        "Абс. погрешность": np.abs(y_exact[sample_indices] - y_num[sample_indices]),
        "Отн. погрешность": np.abs(y_exact[sample_indices] - y_num[sample_indices]) / y_exact[sample_indices]
    })
    print(results_table)

elif variant == "2":
    # Вариант 2: уравнение dy/dt = r*y^2
    # При y0=1 и r=0.05 критический момент t=20
    t0 = 0
    t_end = 19
    y0 = 1.0
    r = 0.05
    h = 0.1

    t, y_num, y_exact = euler_quad(y0, r, t0, t_end, h)

    # Построение графика (зелёный для точного, пурпурный для Эйлера)
    plt.figure(figsize=(10, 6))
    plt.plot(t, y_exact, 'g-', label="Аналитическое решение")
    plt.plot(t, y_num, 'm--', label="Численное решение (Эйлер)")
    plt.xlabel("Время (лет)")
    plt.ylabel("Численность популяции")
    plt.title("Решение уравнения dy/dt = r*y² (Вариант 2)")
    plt.legend()
    plt.grid(True)
    plt.show()

    sample_indices = np.arange(0, len(t), 10)
    results_table = pd.DataFrame({
        "Время (лет)": t[sample_indices],
        "Аналитическое": y_exact[sample_indices],
        "Численное (Эйлер)": y_num[sample_indices],
        "Абс. погрешность": np.abs(y_exact[sample_indices] - y_num[sample_indices]),
        "Отн. погрешность": np.abs(y_exact[sample_indices] - y_num[sample_indices]) / y_exact[sample_indices]
    })
    print(results_table)

elif variant == "3":
    # Вариант 3: модель с хищником
    t0 = 0
    t_end = 40
    y0 = 100
    R = 0.4
    K = 800
    B = 0.03
    P = 8
    A = 60
    h = 0.1

    t, y_num = euler_predator(y0, R, K, B, P, A, t0, t_end, h)

    # Построение графика (бирюзовый цвет)
    plt.figure(figsize=(10, 6))
    plt.plot(t, y_num, 'c-', label="Численное решение (Эйлер)")
    plt.xlabel("Время (лет)")
    plt.ylabel("Численность популяции")
    plt.title("Модель с хищником (Вариант 3)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Формируем таблицу результатов для варианта 3
    sample_indices = np.arange(0, len(t), 10)
    results_table = pd.DataFrame({
        "Время (лет)": t[sample_indices],
        "Численное (Эйлер)": y_num[sample_indices]
    })
    print(results_table)

else:
    print("Неверный выбор. Пожалуйста, запустите программу и введите 1, 2 или 3.")
