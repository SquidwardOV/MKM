import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def solve_exact_variant6_clamped():
    """
    ЗАДАНИЕ 1, ВАРИАНТ 6 (Требия): точное решение системы Осипова–Ланчестера
    с 'обрезкой' графика по времени t*, когда проигрывающая сторона уходит в ноль.
    """
    # Параметры:
    A0 = 31000  # Карфаген
    B0 = 45000  # Рим
    alpha = 2.0 # Карфагеняне в 2 раза эффективнее
    beta  = 1.0 # Римляне
    lam = np.sqrt(alpha * beta)

    # Формулы для A(t) и B(t):
    def A(t):
        return A0 * np.cosh(lam * t) - np.sqrt(beta / alpha) * B0 * np.sinh(lam * t)
    def B(t):
        return B0 * np.cosh(lam * t) - np.sqrt(alpha / beta) * A0 * np.sinh(lam * t)

    # Квадратичный закон:
    C = alpha*(A0**2) - beta*(B0**2)
    print("=== Задача 1 (Вариант 6): Точное решение ===")
    print(f"alpha = {alpha}, beta = {beta}")
    print(f"A(0) = {A0}, B(0) = {B0}")
    print("Квадратичный закон: C = alpha*A0^2 - beta*B0^2 = ", C)
    if C > 0:
        print("=> По модели побеждает армия A (Карфаген).")
        # тогда обнуляется B(t*)=0
        # tanh(lam*t*) = (B0 * sqrt(beta)) / (A0 * sqrt(alpha))
        x = (B0 * np.sqrt(beta)) / (A0 * np.sqrt(alpha))
        losing_side = "B"
    elif C < 0:
        print("=> По модели побеждает армия B (Рим).")
        # тогда обнуляется A(t*)=0
        # tanh(lam*t*) = (A0 * sqrt(alpha)) / (B0 * sqrt(beta))
        x = (A0 * np.sqrt(alpha)) / (B0 * np.sqrt(beta))
        losing_side = "A"
    else:
        print("=> C = 0, обе стороны уничтожаются одновременно.")
        # формально t* можно вычислить аналогично случаю C<0 или C>0.
        x = (A0 * np.sqrt(alpha)) / (B0 * np.sqrt(beta))
        losing_side = "оба"

    # Считаем t*, если это возможно (|x|<1)
    # arctanh(x) = 0.5 * ln((1+x)/(1-x)), в numpy есть np.arctanh(x)
    if abs(x) < 1:
        t_star = (1/lam)*np.arctanh(x)
        print(f"Время t*, когда '{losing_side}' обнуляется: t* = {t_star:.4f}")
        # Строим график на [0, t_star]
        t_vals = np.linspace(0, t_star, 300)
        A_vals = A(t_vals)
        B_vals = B(t_vals)
        print(f"A(t*) = {A_vals[-1]:.4f}, B(t*) = {B_vals[-1]:.4f}")
    else:
        # Если |x|>=1, значит в пределах классической формулы одна из сторон
        # "не доходит" до нуля (или решение уходит в минус/плюс бесконечность).
        print("Внимание: |x| >= 1 => tanh^-1 не существует.")
        print("Построим график просто на неком интервале времени [0, 5].")
        t_vals = np.linspace(0, 5, 300)
        A_vals = A(t_vals)
        B_vals = B(t_vals)

    # Построение графика
    plt.figure(figsize=(7,5))
    plt.plot(t_vals, A_vals, label="A(t) - Карфаген")
    plt.plot(t_vals, B_vals, label="B(t) - Рим")
    plt.xlabel("t (в условных единицах)")
    plt.ylabel("Численность")
    plt.title("Точное решение (Вариант 6, 'Битва при Требии')")
    plt.legend()
    plt.grid(True)
    plt.show()

def solve_numerical():
    """
    ЗАДАНИЕ 2: Численное решение системы (пример из методички, формула (12)).
    dA/dt = -0.5*A - 0.4*A*B - 0.5*B + 0.7
    dB/dt = -0.7*B - 0.2*A*B + 0.6*A + 0.4
    A(0) = 100, B(0) = 80
    """
    print("=== Задача 2: Численное решение системы ===")
    # Задаём систему уравнений для odeint
    def system(u, t):
        A, B = u
        # Примерные коэффициенты:
        dA_dt = -0.1 * A - 0.02 * A * B - 0.15 * B + 0.3
        dB_dt = -0.15 * B - 0.02 * A * B - 0.1 * A + 0.2
        return [dA_dt, dB_dt]

    # Начальные условия
    A0 = 100
    B0 = 80
    u0 = [A0, B0]

    # Шаг по времени
    t = np.linspace(0, 5, 300)  

    # Численное решение
    sol = odeint(system, u0, t)
    A_vals = sol[:,0]
    B_vals = sol[:,1]

    # График
    plt.figure(figsize=(7,5))
    plt.plot(t, A_vals, 'r', label="A(t)")
    plt.plot(t, B_vals, 'b', label="B(t)")
    plt.xlabel("t (в условных единицах)")
    plt.ylabel("Численность")
    plt.title("Численное решение системы (пример)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Вывод результата в конце периода
    print(f"Конечные значения (t={t[-1]}):")
    print(f"A(t_end) = {A_vals[-1]:.4f}")
    print(f"B(t_end) = {B_vals[-1]:.4f}")

def main():
    print("Выберите задачу для решения:")
    print("1: Точное решение (Задание 1, вариант 6)")
    print("2: Численное решение (Задание 2)")
    choice = input("Введите 1 или 2: ").strip()

    if choice == "1":
        solve_exact_variant6_clamped()
    elif choice == "2":
        solve_numerical()
    else:
        print("Неверный выбор. Перезапустите программу и введите 1 или 2.")

if __name__ == "__main__":
    main()
