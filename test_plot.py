# Source: https://pythonprogramming.net/live-graphs-matplotlib-tutorial/
# https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.FuncAnimation.html

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib import style

style.use('fivethirtyeight')

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)

i = 1

def animate(frame, *fargs):
    global i
    print(i)
    xs = np.arange(0, i+1, 1)
    ys = xs ** 2
    i += 1
    ax1.clear()
    ax1.plot(xs, ys)
    return [ax1]


ani = animation.FuncAnimation(fig, animate, frames=100, interval=33, repeat=False)
plt.show()
