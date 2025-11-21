
import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from mpmath import gamma, polygamma, zeta

# ===========================
# 1. Constants and parameters
# ===========================
Q0 = 4.0
LAMBDA = 0.2414482281120760
LAM = LAMBDA**2
F = 4.0
Z1 = 0.57721566490153
CA, CF, TR = 3.0, 4.0/3.0, 0.5
au, bu = 0.7673432111, 4.034351053734136
ad, bd = 0.7864451584, 3.733874856965
cu, du = 0.1459, 1.1589
cd, dd = 0.1987, -1.2124

# ===========================
# 2. Input Moments (U_v, D_v)
# ===========================
def UVnDA1(n):
    num = 2 * gamma(1+bu) * (
        gamma(-1+au+n)/gamma(au+bu+n)
        + cu*gamma(-0.5+au+n)/gamma(0.5+au+bu+n)
        + du*gamma(au+n)/gamma(1+au+bu+n)
    )
    den = gamma(1+bu) * (
        gamma(au)/gamma(1+au+bu)
        + cu*gamma(0.5+au)/gamma(1.5+au+bu)
        + du*gamma(1+au)/gamma(2+au+bu)
    )
    return num / den

def DVnDA1(n):
    num = gamma(1+bd) * (
        gamma(-1+ad+n)/gamma(ad+bd+n)
        + cd*gamma(-0.5+ad+n)/gamma(0.5+ad+bd+n)
        + dd*gamma(ad+n)/gamma(1+ad+bd+n)
    )
    den = gamma(1+bd) * (
        gamma(ad)/gamma(1+ad+bd)
        + cd*gamma(0.5+ad)/gamma(1.5+ad+bd)
        + dd*gamma(1+ad)/gamma(2+ad+bd)
    )
    return num / den

# ===========================
# 3. αs(Q²) NNLO running
# ===========================
def alpha_s(Q):
    FF = F + 1
    beta0 = 11.0 - 2.0*FF/3.0
    beta1 = 102.0 - 38.0*FF/3.0
    beta2 = 1428.5 - 5033.0*FF/18.0 + 325.0*FF**2/54.0
    L = mp.log(Q/LAM)
    a0 = 1.0/(beta0*L)
    c1 = -beta1/(beta0**2)*mp.log(L)/L
    c2 = (beta1**2/(beta0**4))*(mp.log(L)**2 - mp.log(L) - 1 + beta2*beta0/beta1**2)
    return a0*(1 + c1 + c2)

# ===========================
# 4. NNLO Splitting Functions
# ===========================
def P0_NS(N):  # LO
    return CF*(-4*(polygamma(0,N+1)+Z1)+3+2/(N*(N+1)))

def P1_NS(N):  # NLO — approximate nonsinglet correction
    return CF**2*( -3*(polygamma(0,N+1)+Z1)**2 + 4*zeta(2) - 6.5 )

def P2_NS(N):  # NNLO term — simplified placeholder from literature
    return CF*CA*( -8*(polygamma(0,N+1)+Z1)**3 + 12*zeta(3) - 9.2 )

# ===========================
# 5. NNLO Evolution Kernel
# ===========================
def mnsNNLO(N,Q):
    aQ, aQ0 = alpha_s(Q), alpha_s(Q0)
    beta0 = 11 - 2*F/3
    beta1 = 102 - 38*F/3
    beta2 = 1428.5 - 5033*F/18 + 325*F**2/54
    P0, P1, P2 = P0_NS(N), P1_NS(N), P2_NS(N)
    LQ = mp.log(aQ/aQ0)
    E = (P0/beta0)*LQ + ((P1*beta0 - P0*beta1)/beta0**2)*(aQ - aQ0) + ((P2*beta0**2 - P1*beta0*beta1 + P0*(beta1**2 - beta0*beta2))/beta0**3)*(aQ**2 - aQ0**2)
    return mp.e**E

# ===========================
# 6. Laguerre reconstruction
# ===========================
Nmax = 9
alpha, beta = 3.0, 0.5

def ckj(k,j):
    pf = (-1)**k / (mp.factorial(k)*mp.factorial(j))
    coeff = mp.sqrt((mp.factorial(k)*(2*k+alpha+beta+1)*gamma(k+alpha+beta+1))/(gamma(k+alpha+1)*gamma(k+beta+1)))
    return pf * coeff * mp.rf(-k,j)*mp.rf(alpha+beta+k+1,j)*mp.rf(beta+j+1,k-j)

def theta_k(k,x):
    def f(t): return t**(beta+k)*(1-t)**(alpha+k)
    return ((-1)**k/mp.factorial(k)) * mp.diff(f,x,k)

def a_ab_uv(k,Q):
    return sum(ckj(k,j)*UVnDA1(j+2)*mnsNNLO(j+2,Q) for j in range(k+1))

def a_ab_dv(k,Q):
    return sum(ckj(k,j)*DVnDA1(j+2)*mnsNNLO(j+2,Q) for j in range(k+1))

# ===========================
# 7. x qvNNLO(x,Q)
# ===========================
def xuvNNLO(x,Q):
    return mp.re(sum(theta_k(k,x)*a_ab_uv(k,Q) for k in range(Nmax+1)))

def xdvNNLO(x,Q):
    return mp.re(sum(theta_k(k,x)*a_ab_dv(k,Q) for k in range(Nmax+1)))

# ===========================
# 8. Plot results
# ===========================
xs = np.logspace(-3,0,150)
Qs = [50,100,1000]

plt.figure(figsize=(6.5,4.5))
for Q in Qs:
    y = [float(xdvNNLO(x,Q)) for x in xs]
    plt.semilogx(xs, y, label=f'xdvNNLO Q={Q}')
plt.ylim(0,0.8)
plt.xlabel('x'); plt.ylabel('x d_v(x,Q²)')
plt.legend(); plt.grid(True,which='both'); plt.tight_layout(); plt.show()

plt.figure(figsize=(6.5,4.5))
for Q in Qs:
    y = [float(xuvNNLO(x,Q)) for x in xs]
    plt.semilogx(xs, y, label=f'xuvNNLO Q={Q}')
plt.ylim(0,0.8)
plt.xlabel('x'); plt.ylabel('x u_v(x,Q²)')
plt.legend(); plt.grid(True,which='both'); plt.tight_layout(); plt.show()
