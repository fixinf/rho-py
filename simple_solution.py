from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

from scipy.optimize._root import root

m = 1
a = 1
n0 = 1.1
b = .25
alpha = 1

def E(x, n):
    return m**2 * x[0]**2 / - n*x[0] - alpha/2 * x[1]**2*(
        (x[0] - a)**2 - b**2
    )

def eqs(x, n):
    return [
        m**2 * x[0] - n - x[1]**2 * (x[0] - a),
        alpha*x[1]*((x[0]-a)**2 - b**2)
    ]

def sol(n, init=[0., 0.]):
    return root(lambda z: eqs(z, n), init)['x']

print(sol(2*n0, init=[0, 10.]))

print(optimize.minimize(lambda z: E(z, 2*n0), [0.7, 2.]))
exit()
xrange = np.linspace(0, 5, 100)
plt.plot(xrange, [eqs([1.25, z], n0) for z in xrange])
plt.plot(xrange, [eqs([0.75, z], n0) for z in xrange])
plt.plot(xrange, [0. for x in xrange])
plt.ylim([-5, 5])
plt.show()




