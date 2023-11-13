import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation

g = 9.81
ell = 1

# initial conditions: theta=30 deg, velocity=0
theta0 = np.deg2rad(30)
theta_dot0 = 0

# our system of differential equations
# y[0] is theta, y[1] is theta_dot
def pendulum_ODE(t, y): 
    return (y[1], -g*np.sin(y[0])/ell)

# solve the ODE, 30 fps
sol = solve_ivp(pendulum_ODE, [0, 1000], (theta0, theta_dot0), 
    t_eval=np.linspace(0, 30, 1000))

# output of the solver
theta, theta_dot = sol.y
t = sol.t

# animate everything together!
fig = plt.figure()

# pendulum
def pend_pos(theta):
    return (ell*np.sin(theta), -ell*np.cos(theta))

ax2 = fig.add_subplot()
# ax2.set_xlim(-1, 1)
# ax2.set_ylim(-1.5, 0.5)

# draw the pendulum
x0, y0 = pend_pos(theta0)
# line, = ax2.plot([0, x0], [0, y0], 'ro--', lw=2, markersize=8)

x, y = pend_pos(theta)
plt.plot(x, y)
# def animate(i):

#     x, y = pend_pos(theta[i])
#     line.set_data([0, x], [0, y])


# # save a video: 30 fps
# ani = animation.FuncAnimation(fig, animate, frames=len(t))
# plt.show()
plt.savefig('simples1.png')