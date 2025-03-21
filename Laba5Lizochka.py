import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def solve_exact_variant1_clamped():
    """
    Точное решение системы Осипова–Ланчестера:
        dA/dt = -beta*B,
        dB/dt = -alpha*A
    для Варианта 1 (Битва при Аустерлице):
        A(0) = 85400  (коалиция, русско-австрийская)
        B(0) = 73200  (французская армия)
    Предполагается, что французская армия эффективнее в 1.5 раза:
        alpha = 1.0 (коалиция),  beta = 1.5 (французы).
    Отсечение графика производится по моменту t*, когда проигрывающая сторона уходит в ноль.
    """

    # Начальные данные
    A0 = 85400   # Коалиция
    B0 = 73200   # Французы
    alpha = 1.0  # Эффективность коалиции
    beta  = 1.5  # Эффективность французов
    lam = np.sqrt(alpha * beta)

    # Точные решения через гиперболические функции
    def A(t):
        """Численность коалиции во времени."""
        return A0 * np.cosh(lam*t) - np.sqrt(beta/alpha)*B0 * np.sinh(lam*t)

    def B(t):
        """Численность французской армии во времени."""
        return B0 * np.cosh(lam*t) - np.sqrt(alpha/beta)*A0 * np.sinh(lam*t)

    # Квадратичный закон Ланчестера:
    C = alpha*(A0**2) - beta*(B0**2)
    print("=== Точное решение (Вариант 1: Аустерлиц) ===")
    print(f"alpha = {alpha}, beta = {beta}")
    print(f"A(0) = {A0}, B(0) = {B0}")
    print("Квадратичный закон: C = alpha*A(0)^2 - beta*B(0)^2 =", C)

    # Определяем, кто проигрывает
    if C > 0:
        # Побеждает A, значит проигрывает B -> B(t*)=0
        print("=> Побеждает коалиция (A). Проигрывает французская армия (B).")
        # Формула: tanh(lam * t*) = [ B(0)*sqrt(beta ) ] / [ A(0)*sqrt(alpha) ]
        x = (B0 * np.sqrt(beta)) / (A0 * np.sqrt(alpha))
        losing_side = "B"
    elif C < 0:
        # Побеждает B, значит проигрывает A -> A(t*)=0
        print("=> Побеждает французская армия (B). Проигрывает коалиция (A).")
        # Формула: tanh(lam * t*) = [ A(0)*sqrt(alpha) ] / [ B(0)*sqrt(beta) ]
        x = (A0 * np.sqrt(alpha)) / (B0 * np.sqrt(beta))
        losing_side = "A"
    else:
        print("=> C=0, обе стороны уничтожаются одновременно.")
        # Можно взять формулу для одного из случаев
        x = (A0 * np.sqrt(alpha)) / (B0 * np.sqrt(beta))
        losing_side = "оба"

    if abs(x) < 1:
        # Вычисляем t*
        t_star = (1.0 / lam) * np.arctanh(x)
        print(f"Момент t*, когда сторона '{losing_side}' достигает 0: t* = {t_star:.4f}")
        # Будем строить график только на [0, t_star]
        t_vals = np.linspace(0, t_star, 300)
    else:
        # Если |x|>=1, то формально решение не "обнуляется" в конечном t* < бесконечность
        # или формулы дают комплексный корень. Тогда график на [0,5].
        print("|x| >= 1 => не удаётся отсечь по t*. Рисуем на [0, 5].")
        t_vals = np.linspace(0, 5, 300)

    A_vals = A(t_vals)
    B_vals = B(t_vals)

    plt.figure(figsize=(7,5))
    plt.plot(t_vals, A_vals, label="Coalition(t)")
    plt.plot(t_vals, B_vals, label="French(t)")
    plt.xlabel("t (ед. времени)")
    plt.ylabel("Численность")
    plt.title("Точное решение (Вариант 1: Аустерлиц)")
    plt.legend()
    plt.grid(True)
    plt.show()

def solve_num_variant1():
    """
    ЗАДАНИЕ 2 (Вариант 1): Численное решение расширенной системы.
    dA/dt = -0.05*A - 0.025*A*B - 0.10*B + 1.0
    dB/dt = -0.04*B - 0.025*A*B - 0.08*A + 0.8
    A(0)=85400, B(0)=73200.
    """

    print("=== Численное решение (Вариант 1) ===")

    def extended_system(u, t):
        A_val, B_val = u
        dA_dt = -0.05*A_val - 0.025*A_val*B_val - 0.10*B_val + 1.0
        dB_dt = -0.04*B_val - 0.025*A_val*B_val - 0.08*A_val + 0.8
        return [dA_dt, dB_dt]

    # Начальные условия
    coalition_init =  100
    french_init = 90
    u0 = [coalition_init, french_init]

    # Шаг по времени, можно менять при необходимости
    t = np.linspace(0, 5, 300)

    sol = odeint(extended_system, u0, t)
    A_vals = sol[:, 0]
    B_vals = sol[:, 1]

    plt.figure(figsize=(7,5))
    plt.plot(t, A_vals, 'r', label="Coalition(t)")
    plt.plot(t, B_vals, 'b', label="French(t)")
    plt.xlabel("t (ед. времени)")
    plt.ylabel("Численность")
    plt.title("Численное решение (Вариант 1)")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("Конечные значения при t =", t[-1])
    print("Coalition(t_end) =", A_vals[-1])
    print("French(t_end)    =", B_vals[-1])

def main():
    print("Выберите задачу для решения (Вариант 1):")
    print("1: Точное решение с обрезкой по t*")
    print("2: Численное решение расширенной системы")
    choice = input("Введите 1 или 2: ").strip()
    if choice == "1":
        solve_exact_variant1_clamped()
    elif choice == "2":
        solve_num_variant1()
    else:
        print("Неверный выбор. Перезапустите программу и введите 1 или 2.")

if __name__ == "__main__":
    main()
