import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from mpmath import gamma, polygamma, zeta

# =======================
# 1. parameters
# =======================
Q0 = 4.0
au, bu = 0.669857985496613, 3.520485627621301
ad, bd = 0.685024518931466, 3.168547953579216
LAMBDA = 0.2132482281120760
cu, du, cd, dd = 0.1990, 1.498, 0.5399, -1.4000
F, CA, CF, TR = 4.0, 3.0, 4.0 / 3.0, 0.5
LAM = LAMBDA ** 2
Z1 = 0.57721566490153

# =======================
# 2. momman
# =======================
def UVnDA1(n):
    num = 2 * gamma(1 + bu) * (
        gamma(-1 + au + n) / gamma(au + bu + n)
        + cu * gamma(-0.5 + au + n) / gamma(0.5 + au + bu + n)
        + du * gamma(au + n) / gamma(1 + au + bu + n)
    )
    den = gamma(1 + bu) * (
        gamma(au) / gamma(1 + au + bu)
        + cu * gamma(0.5 + au) / gamma(1.5 + au + bu)
        + du * gamma(1 + au) / gamma(2 + au + bu)
    )
    return num / den


def DVnDA1(n):
    num = gamma(1 + bd) * (
        gamma(-1 + ad + n) / gamma(ad + bd + n)
        + cd * gamma(-0.5 + ad + n) / gamma(0.5 + ad + bd + n)
        + dd * gamma(ad + n) / gamma(1 + ad + bd + n)
    )
    den = gamma(1 + bd) * (
        gamma(ad) / gamma(1 + ad + bd)
        + cd * gamma(0.5 + ad) / gamma(1.5 + ad + bd)
        + dd * gamma(1 + ad) / gamma(2 + ad + bd)
    )
    return num / den

# =======================
# 3. αs(Q²) 
# =======================
def alpha_s(Q, F=4):
    FF = F + 1
    B0 = 11 - 2 * FF / 3
    B1 = 102 - 38 * FF / 3
    B2 = 1428.5 - (5033 * FF) / 18 + (325 * FF ** 2) / 54
    B3 = 29243 - 6946.3 * FF + 405.089 * FF ** 2 + 1.49931 * FF ** 3
    L = mp.log(Q / LAM)
    ASLO = 1 / (B0 * L)
    mc2 = 1.5 ** 2
    lh = mp.log(Q / mc2)
    part1 = ASLO * lh / 6
    part2 = (ASLO ** 2) * ((lh / 6) ** 2 - (19 * lh / 24) - (7 / 24))
    c3 = (
        -80507 / 27648 * zeta(3)
        - (2 / 3) * zeta(2) * (1 / 3 * 0.693147 + 1)
        - 58933 / 124416
        + FF / 9 * (zeta(2) + 2479 / 3456)
    )
    part3 = (ASLO ** 3) * (-lh ** 3 / 216 - 131 * lh ** 2 / 576 + lh / 1728 * (-8521 + 409 * FF) + c3)
    return ASLO * (1 - part1 + part2 + part3)

# =======================
# 
# =======================
def P0NS(NN):
    CF = 4.0 / 3.0
    return CF * (-4 * (polygamma(0, NN + 1) + Z1) + 3 + 2 / (NN * (NN + 1)))


def mnsLOV(NN, Q):
    AS_Q, AS_Q0 = alpha_s(Q), alpha_s(Q0)
    BETA0 = 11.0 - (2.0 * F) / 3.0
    S = mp.log(AS_Q0 / AS_Q)
    return mp.e ** (S * P0NS(NN) / BETA0)

# =======================
# 5. 
# =======================
alpha, beta = 3.0, 0.5
Nmax = 9

def ckj(k, j):
    pf = (-1) ** k / (mp.factorial(k) * mp.factorial(j))
    term = mp.sqrt(
        (mp.factorial(k) * (2 * k + alpha + beta + 1) * gamma(k + alpha + beta + 1))
        / (gamma(k + alpha + 1) * gamma(k + beta + 1))
    )
    return pf * term * mp.rf(-k, j) * mp.rf(alpha + beta + k + 1, j) * mp.rf(beta + j + 1, k - j)


def a_ab_uv(k, NN_func):
    return sum(ckj(k, j) * UVnDA1(j + 2) * NN_func(j + 2) for j in range(k + 1))


def a_ab_dv(k, NN_func):
    return sum(ckj(k, j) * DVnDA1(j + 2) * NN_func(j + 2) for j in range(k + 1))


def theta_k(k, x):
    def f(t):
        return t ** (beta + k) * (1 - t) ** (alpha + k)
    return ((-1) ** k / mp.factorial(k)) * mp.diff(f, x, k)

# =======================
# 6. PDF
# =======================
def xuvLO(x, Q):
    return sum(theta_k(k, x) * a_ab_uv(k, lambda n: mnsLOV(n, Q)) for k in range(Nmax + 1))

def xdvLO(x, Q):
    return sum(theta_k(k, x) * a_ab_dv(k, lambda n: mnsLOV(n, Q)) for k in range(Nmax + 1))

def FP(x, Q): return (1 / 9.0) * (4 * xuvLO(x, Q) + xdvLO(x, Q))
def Fd(x, Q): return (5 / 18.0) * (xuvLO(x, Q) + xdvLO(x, Q))
def FNs(x, Q): return (1 / 3.0) * (xuvLO(x, Q) - xdvLO(x, Q))

# =======================
# 7. test
# =======================
print('--- Test values for F_NS ---')
for Q in [5.460, 6.960, 9.000, 11.440, 14.110]:
    val = float(FNs(0.035, Q))
    print(f"F_NS(x=0.035, Q={Q}) = {val:.6e}")

print("\nF_d, F_P, F_NS at x=0.1, Q=20")
print(f"Fd = {float(Fd(0.1, 20)):.6e}")
print(f"Fp = {float(FP(0.1, 20)):.6e}")
print(f"Fns = {float(FNs(0.1, 20)):.6e}")

# =======================
# 8. plot
# =======================
xs = np.logspace(-3, 0, 150)
Qs = [50, 100, 1000]

# --- نمودار x d_v
plt.figure(figsize=(6.5, 4.5))
for Q in Qs:
    y = [float(xdvLO(x, Q)) for x in xs]
    plt.semilogx(xs, y, label=f'xdvLO Q={Q}')
plt.ylim(0, 0.8)
plt.xlabel('x'); plt.ylabel('x d_v(x,Q²)')
plt.legend(); plt.grid(True, which='both'); plt.tight_layout()
plt.show()

# --- x u_v
plt.figure(figsize=(6.5, 4.5))
for Q in Qs:
    y = [float(xuvLO(x, Q)) for x in xs]
    plt.semilogx(xs, y, label=f'xuvLO Q={Q}')
plt.ylim(0, 0.8)
plt.xlabel('x'); plt.ylabel('x u_v(x,Q²)')
plt.legend(); plt.grid(True, which='both'); plt.tight_layout()
plt.show()

# ---  F2(x=0.75,Q)
Qgrid = np.linspace(1, 1000, 200)
plt.figure(figsize=(6, 4))
plt.semilogx(Qgrid, [float(FP(0.75, Q)) for Q in Qgrid], label='Fp(0.75,Q)')
plt.semilogx(Qgrid, [float(Fd(0.75, Q)) for Q in Qgrid], label='Fd(0.75,Q)')
plt.legend(); plt.ylim(0, 0.1)
plt.xlabel("Q [GeV]"); plt.ylabel("F₂(x=0.75,Q²)")
plt.grid(True, which='both')
plt.tight_layout(); plt.show()
