import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# -------------------- Выбор задачи --------------------

def choose_problem():
    print("Выберите задачу для 1-го варианта:")
    print("  1 — Задача Коши")
    print("  2 — Смешанная задача")
    choice = input("Ваш выбор (1 или 2): ")
    return choice.strip()

# ---------------------- Задача Коши (вариант 1) ----------------------

alpha = 2.0  # скорость распространения волны

def G(x):
    # начальное смещение g(x) = x^2
    return x**2

def H(x):
    # начальное распределение скоростей h(x) = x
    return x

def F(x, t):
    # вынужденная сила f(x,t) = x * cos(t)
    return x * np.cos(t)

def u_cauchy_v1(x, t):
    # Формула Даламбера для одной точки (x,t)
    # 1) свободная волна
    w1 = 0.5*(G(x - alpha*t) + G(x + alpha*t))
    # 2) вклад начальных скоростей
    I_H, _ = quad(lambda xi: H(xi), x - alpha*t, x + alpha*t)
    w2 = 0.5/alpha * I_H
    # 3) вклад вынужденной силы
    def inner(tau):
        L = x - alpha*(t - tau)
        R = x + alpha*(t - tau)
        I_F, _ = quad(lambda s: F(s, tau), L, R)
        return I_F
    I_FF, _ = quad(inner, 0, t)
    w3 = 0.5/alpha * I_FF
    return w1 + w2 + w3

def simulate_cauchy_v1():
    xs = np.linspace(-2, 2, 400)

    # 0) начальное состояние
    plt.figure(figsize=(6,3))
    plt.plot(xs, [G(x) for x in xs], color='darkorange', lw=2)
    plt.title("Начальное состояние: u(x,0)=G(x)=x^2")
    plt.xlabel("x"); plt.grid(True)
    plt.show()

    # 1–4) эволюция в одном графике
    times = [0.25, 0.5, 0.75, 1.0]
    colors = ['C0','C1','C2','C3']
    # посчитаем все профили
    curves = []
    for T in times:
        curves.append(np.array([u_cauchy_v1(x, T) for x in xs]))
    all_vals = np.hstack(curves)
    ymin, ymax = all_vals.min(), all_vals.max()
    pad = 0.1*(ymax - ymin)

    plt.figure(figsize=(8,5))
    for idx, T in enumerate(times):
        plt.plot(xs, curves[idx], color=colors[idx], lw=2, label=f"t={T}")
    plt.ylim(ymin - pad, ymax + pad)
    plt.title("Эволюция u(x,t) — Задача Коши (вариант 1)")
    plt.xlabel("x"); plt.ylabel("u(x,t)")
    plt.legend(); plt.grid(True)
    plt.show()

# ------------------ Смешанная задача (вариант 1) ------------------

L1   = 1.0
a_m1 = 1.0

def phi1(x):
    # φ(x)=x(x-1)
    return x*(x-1)

def psi1(x):
    # ψ(x)=1
    return 1.0

def mu1(n):
    # собственные числа для u(0)=u(L)=0
    return n * np.pi / L1

def X1(n, x):
    return np.sin(mu1(n) * x)

def A1(n):
    I, _ = quad(lambda xi: phi1(xi) * X1(n, xi), 0, L1)
    return 2.0 / L1 * I

def B1(n):
    I, _ = quad(lambda xi: psi1(xi) * X1(n, xi), 0, L1)
    return 2.0 / (a_m1 * mu1(n) * L1) * I

def u_mixed_v1(x, t, N=50):
    s = 0.0
    for n in range(1, N+1):
        lam = mu1(n)
        s += (A1(n)*np.cos(a_m1*lam*t) + B1(n)*np.sin(a_m1*lam*t)) * X1(n, x)
    return s

def simulate_mixed_v1():
    xs = np.linspace(0, L1, 400)

    # 0) начальное состояние
    plt.figure(figsize=(6,3))
    plt.plot(xs, [phi1(x) for x in xs], color='teal', lw=2)
    plt.title("Начальное состояние: u(x,0)=φ(x)=x(x-1)")
    plt.xlabel("x"); plt.grid(True)
    plt.show()

    # 1–4) эволюция в одном графике
    times = [0.5, 1.0, 1.5, 2.0]
    colors = ['magenta','olive','navy','brown']
    curves = []
    for T in times:
        curves.append(np.array([u_mixed_v1(x, T) for x in xs]))
    all_vals = np.hstack(curves)
    ymin, ymax = all_vals.min(), all_vals.max()
    pad = 0.1*(ymax - ymin)

    plt.figure(figsize=(8,5))
    for idx, T in enumerate(times):
        plt.plot(xs, curves[idx], color=colors[idx], lw=2, label=f"t={T}")
    plt.ylim(ymin - pad, ymax + pad)
    plt.title("Эволюция u(x,t) — Смешанная задача (вариант 1)")
    plt.xlabel("x"); plt.ylabel("u(x,t)")
    plt.legend(); plt.grid(True)
    plt.show()

# ------------------------- Main -------------------------

if __name__ == "__main__":
    choice = choose_problem()
    if choice == "1":
        simulate_cauchy_v1()
    elif choice == "2":
        simulate_mixed_v1()
    else:
        print("Неверный выбор — введите 1 или 2 и перезапустите.")
