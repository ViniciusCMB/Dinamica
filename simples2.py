import numpy as np
import sympy as smp
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import PillowWriter

t, m, g = smp.symbols('t m g')
the = smp.symbols(r'\theta', cls=smp.Function)
the = the(t)
the_d = smp.diff(the, t)
the_dd = smp.diff(the_d, t)

x, y = smp.symbols('x y', cls=smp.Function)
x = x(the)
y = y(the)

x = -1*smp.sin(the)
y = -1*smp.cos(the)
x_f = smp.lambdify(the, x)
y_f = smp.lambdify(the, y)

T = 1/2 * m * (smp.diff(x,t)**2 + smp.diff(y,t)**2)
V = m*g*y
L = T-V

LE = smp.diff(L, the) - smp.diff(smp.diff(L, the_d), t)
LE = LE.simplify()

deriv_2 =smp.solve(LE, the_dd)[0]
deriv_1 = the_d

deriv2_f = smp.lambdify((g, the, the_d), deriv_2)
deriv1_f = smp.lambdify(the_d, the_d)

def dSdt(S, t):
    return [
        deriv1_f(S[1]), #dtheta/dt
        deriv2_f(g, S[0], S[1]) #domega/dt
    ]

t = np.linspace(0, 30, 1000)
g = 9.81
ans1 = odeint(dSdt, y0=[np.deg2rad(30), 0], t=t)

# plt.plot(t,ans1.T[0])

def get_xy(theta):
    return -1*np.sin(theta), -1*np.cos(theta)

x1, y1 = get_xy(ans1.T[0])

plt.plot(x1, y1)
# plt.show()
plt.savefig('simples2.png')