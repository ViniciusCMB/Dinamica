import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
from scipy.integrate import odeint
from matplotlib import animation
from matplotlib.animation import PillowWriter

m1,m2,l1,l2,g,t  = sp.symbols('m1 m2 l1 l2 g t')
the1,the2 = sp.symbols(r'theta1 theta2',cls=sp.Function)

the1 = the1(t)
the2 = the2(t)
the1_d = sp.diff(the1,t)
the2_d = sp.diff(the2,t)
the1_dd = sp.diff(the1_d,t)
the2_dd = sp.diff(the2_d,t)

x1 = -l1*sp.sin(the1)
y1 = -l1*sp.cos(the1)
x2 = -l2*sp.sin(the2) + x1
y2 = -l2*sp.cos(the2) + y1

T1 = 1/2*m1*(sp.diff(x1,t)**2 + sp.diff(y1,t)**2)
U1 = y1*m1*g

T2 = 1/2*m2*(sp.diff(x2,t)**2 + sp.diff(y2,t)**2)
U2 = y2*m2*g

T = T1 + T2
U = U1 + U2

L = T - U
L

EqL1 = (sp.diff(sp.diff(L,the1_d),t)-sp.diff(L,the1)).simplify()
EqL2 = (sp.diff(sp.diff(L,the2_d),t)-sp.diff(L,the2)).simplify()

sol = sp.solve([EqL1,EqL2],(the1_dd,the2_dd))

v1_f = sp.lambdify(the1_d,the1_d)
v2_f = sp.lambdify(the2_d,the2_d)

dv1_f = sp.lambdify((t,g,m1,m2,l1,l2,the1,the2,the1_d,the2_d),sol[the1_dd])
dv2_f = sp.lambdify((t,g,m1,m2,l1,l2,the1,the2,the1_d,the2_d),sol[the2_dd])

def dSdt(S,t,g,m1,m2,l1,l2):
    the1, the2, v1, v2 = S
    return[
        v1_f(v1),
        v2_f(v2),
        dv1_f(t,g,m1,m2,l1,l2,the1,the2,v1,v2),
        dv2_f(t,g,m1,m2,l1,l2,the1,the2,v1,v2),
    ]

t_f = 30 #s
n_passo = 1000
t = np.linspace(0,t_f,n_passo)
t2 = np.linspace(0,t_f+45,n_passo)

# ===== Valores das constantes
m1 = 3 # Kg
m2 = 3 # Kg
l1 = 1 # m
l2 = 1 # m
g = 9.8 # m/s2
g2 = 1.62 # m/s2

# ===== Condições iniciais para os três casos
the1_10 = 2 # rad
the2_10 = 2 # rad
dthe1_10 = 0  # rad/s
dthe2_10 = 0  # rad/s

the1_20 = 2 # rad
the2_20 = 2 # rad
dthe1_20 = 0  # rad/s
dthe2_20 = 0 # rad/s

# the1_30 = 1.99 # rad
# the2_30 = 1.99 # rad
# dthe1_30 = 0  # rad/s
# dthe2_30 = 0  # rad/s

resposta1 = odeint(dSdt,y0=[the1_10,the2_10,dthe1_10,dthe2_10],t=t,args=(g,m1,m2,l1,l2))
resposta2 = odeint(dSdt,y0=[the1_20,the2_20,dthe1_20,dthe2_20],t=t2,args=(g2,m1,m2,l1,l2))
# resposta3 = odeint(dSdt,y0=[the1_30,the2_30,dthe1_30,dthe2_30],t=t,args=(g,m1,m2,l1,l2))

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


# plt.plot(t,the1_1t)
# plt.show()

def pos(t,the1,the2,l1,l2):
    x1 = -l1*np.sin(the1)
    y1 = -l1*np.cos(the1)
    x2 = -l2*np.sin(the2) + x1
    y2 = -l2*np.cos(the2) + y1
    return[
        x1,y1,x2,y2
    ]

x11,y11,x12,y12 = pos(t,the1_1t,the2_1t,l1,l2)
x21,y21,x22,y22 = pos(t,the1_2t,the2_2t,l1,l2)
# x31,y31,x32,y32 = pos(t,the1_3t,the2_3t,l1,l2)
plt.plot(t,y12, 'b')
plt.plot(t,y22, 'r')
5
def animate(i):
    ln1.set_data([0, x11[i], x12[i]], [0, y11[i], y12[i]])
    cur.set_data(x12[:i+1],y12[:i+1])
    
def animate2(i):
    ln2.set_data([0, x21[i], x22[i]], [0, y21[i], y22[i]])
    cur2.set_data(x22[:i+1],y22[:i+1])

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot()
fig2 = plt.figure(figsize=(8,8))
ax2 = fig2.add_subplot()

# ax.set_facecolor('k')
# ax.get_xaxis().set_ticks([])    # Para tirar o eixo x
# ax.get_yaxis().set_ticks([])    

ln1, = ax.plot([], [], 'ro--', lw=2, markersize=8)
cur, = ax.plot(x11[0],y11[0],'b',lw=1)
ln2, = ax2.plot([], [], 'bo--', lw=2, markersize=8)
cur2, = ax2.plot(x21[0],y21[0],'r',lw=1)
# ln3, = plt.plot([], [], 'go--', lw=2, markersize=8)

ax.set_ylim(-4,4)
ax.set_xlim(-4,4)
ax2.set_ylim(-4,4)
ax2.set_xlim(-4,4)
ani = animation.FuncAnimation(fig, animate, frames=n_passo, interval=30)
ani2 = animation.FuncAnimation(fig2, animate2, frames=n_passo, interval=30+45)
# ani.save('pen.gif',writer='pillow',fps=len(t[t<1])) # FPS deve ser o número de intervalos em 1 segundo
plt.show()