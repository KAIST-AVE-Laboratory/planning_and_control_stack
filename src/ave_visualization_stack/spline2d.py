import scipy
import scipy.interpolate as ip
from scipy.interpolate import splrep, splev
import numpy as np
import math
from scipy import optimize
from matplotlib import pyplot as plt


class Spline2D():
    def __init__(self, x, y):
        self.unit = 0.1
        dx = np.diff(x)
        dy = np.diff(y)
        dt = [
            math.sqrt(idx ** 2 + idy ** 2 ) for (idx, idy) in zip(dx, dy)
        ]
        zero_index = np.where(np.array(dt)<1e-3)
        dt = np.delete(dt, zero_index)
        x = np.delete(x, zero_index)
        y = np.delete(y, zero_index)

        self.t=[0]
        self.t.extend(np.cumsum(dt))

        k = min(len(self.t)-1, 3)
        self.spl_x = splrep(self.t, x, k=k)
        self.spl_y = splrep(self.t, y, k=k)

        xs, ys = self.calc_world_position(np.arange(0, self.t[-1], self.unit))
        self.points = np.stack((xs,ys),1)

    def calc_world_position(self, t):
        x = splev(t, self.spl_x)
        y = splev(t, self.spl_y)

        return x, y

    def calc_t(self, world_x, world_y):
        location = np.array([world_x, world_y])
        d = np.sum((self.points - location)**2, 1)
        t = np.argmin(d) * self.unit
        return t

if __name__ == '__main__':

    print("it is for test")