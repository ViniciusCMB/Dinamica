import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
from scipy.integrate import odeint
from matplotlib import animation
from matplotlib.animation import PillowWriter
import time

inicial = time.time()

# Simbolos e equações do movimento e lagrangiana
m1, m2, l1, l2, g, t = sp.symbols('m1 m2 l1 l2 g t')
the1, the2 = sp.symbols(r'theta1 theta2', cls=sp.Function)

the1 = the1(t)
the2 = the2(t)
the1_d = sp.diff(the1, t)
the2_d = sp.diff(the2, t)
the1_dd = sp.diff(the1_d, t)
the2_dd = sp.diff(the2_d, t)

x1 = -l1*sp.sin(the1)
y1 = -l1*sp.cos(the1)
x2 = -l2*sp.sin(the2) + x1
y2 = -l2*sp.cos(the2) + y1

T1 = 1/2*m1*(sp.diff(x1, t)**2 + sp.diff(y1, t)**2)
U1 = y1*m1*g

T2 = 1/2*m2*(sp.diff(x2, t)**2 + sp.diff(y2, t)**2)
U2 = y2*m2*g

T = T1 + T2
U = U1 + U2

L = T - U
L

EqL1 = (sp.diff(sp.diff(L, the1_d), t)-sp.diff(L, the1)).simplify()
EqL2 = (sp.diff(sp.diff(L, the2_d), t)-sp.diff(L, the2)).simplify()

sol = sp.solve([EqL1, EqL2], (the1_dd, the2_dd))

v1_f = sp.lambdify(the1_d, the1_d)
v2_f = sp.lambdify(the2_d, the2_d)

dv1_f = sp.lambdify((t, g, m1, m2, l1, l2, the1, the2,
                    the1_d, the2_d), sol[the1_dd])
dv2_f = sp.lambdify((t, g, m1, m2, l1, l2, the1, the2,
                    the1_d, the2_d), sol[the2_dd])


def dSdt(S, t, g, m1, m2, l1, l2):
    the1, the2, v1, v2 = S
    return [
        v1_f(v1),
        v2_f(v2),
        dv1_f(t, g, m1, m2, l1, l2, the1, the2, v1, v2),
        dv2_f(t, g, m1, m2, l1, l2, the1, the2, v1, v2),
    ]


t_s = 30  # s
n_passo = 1000
t = np.linspace(0, t_s, n_passo)


# Condições iniciais
m1 = 3  # Kg
m2 = 3  # Kg
l1 = 1  # m
l2 = 1  # m

g = 9.81  # m/s2
g2 = 9.69  # m/s2

the1_10 = 2  # rad
the2_10 = 2  # rad
dthe1_10 = 0  # rad/s
dthe2_10 = 0  # rad/s

the1_20 = 2  # rad
the2_20 = 2  # rad
dthe1_20 = 0  # rad/s
dthe2_20 = 0  # rad/s

# the1_30 = 1.99  # rad
# the2_30 = 1.99  # rad
# dthe1_30 = 0  # rad/s
# dthe2_30 = 0  # rad/s


# Resolve o sistema de equações diferenciais
resposta1 = odeint(
    dSdt, y0=[the1_10, the2_10, dthe1_10, dthe2_10], t=t, args=(g, m1, m2, l1, l2))
resposta2 = odeint(dSdt, y0=[the1_20, the2_20, dthe1_20, dthe2_20], t=t, args=(g2, m1, m2, l1, l2))
# resposta3 = odeint(dSdt, y0=[the1_30, the2_30, dthe1_30, dthe2_30], t=t, args=(g, m1, m2, l1, l2))

the1_1t = resposta1.T[0]
the2_1t = resposta1.T[1]
dthe1_1t = resposta1.T[2]
dthe2_1t = resposta1.T[3]

the1_2t = resposta2.T[0]
the2_2t = resposta2.T[1]
dthe1_2t = resposta2.T[2]
dthe2_2t = resposta2.T[3]

# the1_3t = resposta3.T[0]
# the2_3t = resposta3.T[1]
# dthe1_3t = resposta3.T[2]
# dthe2_3t = resposta3.T[3]


# Calcula a posição cartesiana do pêndulo
def pos(t, the1, the2, l1, l2):
    x1 = -l1*np.sin(the1)
    y1 = -l1*np.cos(the1)
    x2 = -l2*np.sin(the2) + x1
    y2 = -l2*np.cos(the2) + y1
    return [
        x1, y1, x2, y2
    ]


x11, y11, x12, y12 = pos(t, the1_1t, the2_1t, l1, l2)
x21, y21, x22, y22 = pos(t, the1_2t, the2_2t, l1, l2)
# x31, y31, x32, y32 = pos(t, the1_3t, the2_3t, l1, l2)


# Responsável por animar o gráfico
def animate(i):
    ln.set_data([0, x11[i], x12[i]], [0, y11[i], y12[i]])
    cur.set_data(x12[:i+1], y12[:i+1])
    py.set_data(t[:i+1], y12[:i+1])
    px.set_data(t[:i+1], x12[:i+1])

    ln2.set_data([0, x21[i], x22[i]], [0, y21[i], y22[i]])
    cur2.set_data(x22[:i+1], y22[:i+1])
    py2.set_data(t[:i+1], y22[:i+1])
    px2.set_data(t[:i+1], x22[:i+1])

    # ln3.set_data([0, x31[i], x32[i]], [0, y31[i], y32[i]])
    # cur3.set_data(x32[:i+1], y32[:i+1])
    # py3.set_data(t[:i+1], y32[:i+1])
    # px3.set_data(t[:i+1], x32[:i+1])


# Plota os gráficos
fig = plt.figure(figsize=(19, 10))
ax = plt.subplot(121)
ax2 = plt.subplot(222)
ax3 = plt.subplot(224)

ax.set_title('Pêndulo - Terra/Urano')
ax.set_ylim(-3, 3)
ax.set_xlim(-3, 3)
ax.grid()

ax2.set_title('Posição y - Pêndulo Secundário')
ax2.set_ylabel('Posição y')
ax2.set_xlabel('Tempo (s)')
ax2.set_ylim(-2.5, 1.5)
ax2.set_xlim(0, t_s)
ax2.grid()

ax3.set_title('Posição x - Pêndulo Secundário')
ax3.set_ylabel('Posição x')
ax3.set_xlabel('Tempo (s)')
ax3.set_ylim(-2.5, 2.5)
ax3.set_xlim(0, t_s)
ax3.grid()

ln, = ax.plot([], [], 'bo--', lw=2, markersize=8)
cur, = ax.plot(x11[0], y11[0], 'b', lw=1)
py, = ax2.plot([], [], 'b')
px, = ax3.plot([], [], 'b')

ln2, = ax.plot([], [], 'go--', lw=2, markersize=8)
cur2, = ax.plot(x21[0], y21[0], 'g', lw=1)
py2, = ax2.plot([], [], 'g')
px2, = ax3.plot([], [], 'g')

# ln3, = ax.plot([], [], 'ro--', lw=2, markersize=8)
# cur3, = ax.plot(x31[0], y31[0], 'r', lw=1)
# py3, = ax2.plot([], [], 'r')
# px3, = ax3.plot([], [], 'r')

ani = animation.FuncAnimation(fig, animate, frames=n_passo, interval=t_s)
ani.save('penTerra-Urano.gif', writer='pillow', fps=len(t[t < 1]))
# plt.show()
plt.savefig('penTerra-Urano.png')


final = time.time()
print(f'{final-inicial:.4f}')
