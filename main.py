import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from functools import partial


amplitude = 2
frequencia_angular = 2
deslocamento = 0

fig, ax = plt.subplots()
line1, = ax.plot([], [], 'ro')

def init():
    ax.set_xlim(-1, 2*np.pi+1)
    ax.set_ylim(-3, 3)
    return line1,

def update(frame, ln, x, y):
    x.append(frame)
    y.append(deslocamento + (amplitude * np.cos((frequencia_angular * frame) + np.pi/2)))
    ln.set_data(x, y)
    return ln,

if __name__ == '__main__':
    ani = FuncAnimation(
        fig, partial(update, ln=line1, x=[], y=[]),
        frames=np.linspace(0, 2*np.pi, 126),
        init_func=init, blit=True)
    ani.save("my_animation.gif", fps=60)
    plt.show()
    