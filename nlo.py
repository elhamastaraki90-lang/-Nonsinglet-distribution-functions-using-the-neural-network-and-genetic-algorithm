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
au, bu = 0.76458881889422, 3.897451053734136
ad, bd = 0.7441541640714, 3.44512285255513
cu, du = 0.2125, 1.3100
cd, dd = 0.4207, -1.343
Z1 = 0.57721566490153
CA, CF, TR = 3.0, 4.0/3.0, 0.5

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
# 3. αs(Q²) NLO running
# ===========================
def alpha_s(Q):
    FF = F + 1
    B0 = 11 - 2*FF/3
    B1 = 102 - 38*FF/3
    B2 = 1428.5 - 5033*FF/18 + 325*FF**2/54
    B3 = 29243 - 6946.3*FF + 405.089*FF**2 + 1.49931*FF**3
    L = mp.log(Q/LAM)
    ASLO = 1/(B0*L)
    ASNLO = -1/(B0*L)**2 * (B1*mp.log(L))
    ALFASu = ASLO + ASNLO
    mc2 = 1.5**2
    lh = mp.log(Q/mc2)
    part1 = ALFASu*lh/6
    part2 = (ALFASu**2)*((lh/6)**2 - (19*lh/24) - (7/24))
    c3 = -80507/27648*zeta(3) - (2/3)*zeta(2)*(1/3*0.693147 + 1) - 58933/124416 + FF/9*(zeta(2)+2479/3456)
    part3 = (ALFASu**3)*(-lh**3/216 - 131/576*lh**2 + lh/1728*(-8521 + 409*FF) + c3)
    return ALFASu*(1 - part1 + part2 + part3)

# ===========================
# 4. Splitting functions (First approximation for NLO)
# ===========================
def P0NS(NN):
    return CF*(-4*(polygamma(0, NN+1) + Z1) + 3 + 2/(NN*(NN+1)))

# ===========================
# 5. mnsNLOV Evolution kernel
# ===========================
def mnsNLOV(NN, Q):
    ASF, ASI = alpha_s(Q), alpha_s(Q0)
    B0 = 11 - 2*F/3
    S = mp.log(ASI/ASF)
    LNS = mp.e**(S*P0NS(NN)/B0)
    return LNS*(1 + (ASF-ASI)*(1-ASI))

# ===========================
# 6. Laguerre reconstruction
# ===========================
Nmax = 9
alpha, beta = 3.0, 0.5

def ckj(k, j):
    pf = (-1)**k / (mp.factorial(k) * mp.factorial(j))
    coeff = mp.sqrt((mp.factorial(k)*(2*k+alpha+beta+1)*gamma(k+alpha+beta+1))
                    /(gamma(k+alpha+1)*gamma(k+beta+1)))
    return pf * coeff * mp.rf(-k,j)*mp.rf(alpha+beta+k+1,j)*mp.rf(beta+j+1,k-j)

def theta_k(k, x):
    def f(t): return t**(beta+k)*(1-t)**(alpha+k)
    return ((-1)**k/mp.factorial(k)) * mp.diff(f, x, k)

def a_ab_uv(k, Q):
    return sum(ckj(k,j)*UVnDA1(j+2)*mnsNLOV(j+2,Q) for j in range(k+1))

def a_ab_dv(k, Q):
    return sum(ckj(k,j)*DVnDA1(j+2)*mnsNLOV(j+2,Q) for j in range(k+1))

# ===========================
# 7. x qvNLO(x,Q)
# ===========================
def xuvNLO(x,Q):
    return sum(theta_k(k,x)*a_ab_uv(k,Q) for k in range(Nmax+1))

def xdvNLO(x,Q):
    return sum(theta_k(k,x)*a_ab_dv(k,Q) for k in range(Nmax+1))

# ===========================
# 8. Test and Plot
# ===========================
print('xuvNLO(0.1,20)=', float(xuvNLO(0.1,20)))

xs = np.logspace(-3,0,150)
Qs = [50,100,1000]

plt.figure(figsize=(6.5,4.5))
for Q in Qs:
    y = [float(xdvNLO(x,Q)) for x in xs]
    plt.semilogx(xs, y, label=f'xdvNLO Q={Q}')
plt.ylim(0,0.8)
plt.xlabel('x')
plt.ylabel('x d_v(x,Q²)')
plt.legend(); plt.grid(True,which='both')
plt.tight_layout(); plt.show()

plt.figure(figsize=(6.5,4.5))
for Q in Qs:
    y = [float(xuvNLO(x,Q)) for x in xs]
    plt.semilogx(xs, y, label=f'xuvNLO Q={Q}')
plt.ylim(0,0.8)
plt.xlabel('x')
plt.ylabel('x u_v(x,Q²)')
plt.legend(); plt.grid(True,which='both')
plt.tight_layout(); plt.show()
