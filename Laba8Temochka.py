import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# -------------------- Выбор задачи --------------------

def choose_problem():
    print("Выберите задачу для моделирования:")
    print("  1 — Задача Коши (6-й вариант)")
    print("  2 — Смешанная задача (6-й вариант)")
    choice = input("Ваш выбор (1 или 2): ")
    return choice.strip()

# ---------------------- Задача Коши (6) ----------------------

a = 1.0  # скорость распространения волны

def g(x):
    return 1.0 / (1.0 + x**2)

def h(x):
    return np.cos(x)

def f(x, t):
    return x * t  # f(x,t) = x·t

def u_cauchy(x, t):
    # Аналитически по Даламберу:
    #   u(x,t) = (g(x-at)+g(x+at))/2
    #          + 1/(2a) ∫_{x-at}^{x+at} h(ξ) dξ
    #          + 1/(2a) ∫_{0}^{t} [ ∫_{x-a(t-τ)}^{x+a(t-τ)} f(s,τ) ds ] dτ
    term1 = 0.5*(g(x - a*t) + g(x + a*t))
    I_h, _ = quad(lambda xi: h(xi), x - a*t, x + a*t)
    term2 = 0.5/a * I_h
    def inner(t1):
        L = x - a*(t - t1)
        R = x + a*(t - t1)
        I_f, _ = quad(lambda s: f(s, t1), L, R)
        return I_f
    I_ff, _ = quad(inner, 0, t)
    term3 = 0.5/a * I_ff
    return term1 + term2 + term3

def simulate_cauchy():
    xs = np.linspace(-5, 5, 300)

    # 0) начальное состояние
    plt.figure(figsize=(5, 3))
    plt.plot(xs, [g(x) for x in xs])
    plt.title("Начальное состояние: u(x,0)=g(x)")
    plt.xlabel("x")
    plt.grid(True)
    plt.show()

    # 1–4) четыре момента времени
    times = [0.25, 0.5, 0.75, 1.0]
    plt.figure(figsize=(8, 6))
    for i, T in enumerate(times, 1):
        us = [u_cauchy(x, T) for x in xs]
        plt.subplot(2, 2, i)
        plt.plot(xs, us)
        plt.title(f"t = {T}")
        plt.ylim(-2, 2)
        plt.grid(True)
    plt.suptitle("Задача Коши, вариант 6", y=1.02)
    plt.tight_layout()
    plt.show()

# ------------------ Смешанная задача (6) ------------------

L = 1.0   # длина струны
a_m = 1.0

def phi(x):
    return np.cos(np.pi * x / 2)

def psi(x):
    return 1.0

def mu(n):
    # собственные числа для BC ux(0)=0, u(1)=0: (2n-1)π/(2L)
    return np.pi * (2*n - 1) / (2 * L)

def X(n, x):
    return np.cos(mu(n) * x)  # чтобы X'(0)=0

def A(n):
    I, _ = quad(lambda xi: phi(xi) * X(n, xi), 0, L)
    return 2.0 / L * I

def B(n):
    I, _ = quad(lambda xi: psi(xi) * X(n, xi), 0, L)
    return 2.0 / (a_m * mu(n)) * I

def u_mixed(x, t, N=50):
    # Аналитически по Фурье (четвёртая краевая задача):
    # u(t,x) = Σ_{n=1..∞} [ A_n cos(a μ_n t) + B_n sin(a μ_n t) ] X_n(x)
    s = 0.0
    for n in range(1, N+1):
        lam = mu(n)
        s += (A(n)*np.cos(a_m*lam*t) + B(n)*np.sin(a_m*lam*t)) * X(n, x)
    return s

def simulate_mixed():
    xs = np.linspace(0, L, 300)

    # 0) начальное состояние
    plt.figure(figsize=(5, 3))
    plt.plot(xs, [phi(x) for x in xs])
    plt.title("Начальное состояние: u(x,0)=φ(x)")
    plt.xlabel("x")
    plt.grid(True)
    plt.show()

    # 1–4) четыре момента времени
    times = [0.5, 1.0, 2.0, 3.0]
    plt.figure(figsize=(8, 6))
    for i, T in enumerate(times, 1):
        us = [u_mixed(x, T) for x in xs]
        plt.subplot(2, 2, i)
        plt.plot(xs, us)
        plt.title(f"t = {T}")
        plt.ylim(-1.5, 1.5)
        plt.grid(True)
    plt.suptitle("Смешанная задача, вариант 6", y=1.02)
    plt.tight_layout()
    plt.show()

# ------------------------- Main -------------------------

if __name__ == "__main__":
    choice = choose_problem()
    if choice == "1":
        simulate_cauchy()
    elif choice == "2":
        simulate_mixed()
    else:
        print("Неверный выбор, запустите заново и введите 1 или 2.")
