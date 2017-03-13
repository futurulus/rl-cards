"""
Based on dynamic_image.py from the Matplotlib docs.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class AnimImage(object):
    def __init__(self, gen):
        self.iter = iter(gen)

    def update(self, mat):
        self.im.set_array(mat)
        return self.im,

    def frames(self):
        for mat in self.iter:
            yield mat

    def show(self):
        self.fig = plt.figure()
        self.im = plt.imshow(next(self.iter), cmap='gnuplot2',
                             interpolation='nearest', animated=True)
        self.ani = animation.FuncAnimation(self.fig, self.update, frames=self.frames,
                                           repeat=False, interval=50, blit=True)
        plt.show()


def slices(arr, axis=0):
    for i in range(arr.shape[axis]):
        yield np.take(arr, i, axis=axis)


def f(x, y):
    return np.sin(x) + np.cos(y)


if __name__ == '__main__':
    x = np.linspace(0, 2 * np.pi, 120)
    y = np.linspace(0, 2 * np.pi, 100)
    dx = np.linspace(0, 8 * np.pi, 120)
    dy = np.linspace(0, 6 * np.pi, 120)

    xt = x[:, np.newaxis, np.newaxis] + dx
    yt = y[np.newaxis, :, np.newaxis] + dy
    A = f(xt, yt)

    AnimImage(slices(A, axis=2)).show()
