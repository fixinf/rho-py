import eosWrap as eos
import numpy as np
from matplotlib import pyplot as plt
from scipy.misc.common import derivative
from scipy.optimize._root import root

C = eos.KVOR()
n0 = C.n0

def E_r(f, n_n, n_p, rho_0, rho_c, mu_ch):
    n_in = np.array([f, n_n, n_p])
    return eos.E_rho(n_in, rho_0, rho_c, mu_ch, C)


def rho_eq(n_n, n_p, mu_ch):
    def eq(x):
        f = x[0]
        rho_0 = x[1]
        rho_c = x[2]

        eq1 = derivative(lambda z: E_r(z, n_n, n_p, rho_0, rho_c, mu_ch), f, dx=1e-3)
        eq2 = derivative(lambda z: E_r(f, n_n, n_p, z, rho_c, mu_ch), rho_0, dx=1e-3)
        eq3 = derivative(lambda z: E_r(f, n_n, n_p, rho_0, z, mu_ch), rho_c, dx=1e-3)

        return [eq1, eq2, eq3]

    return root(eq, [0.2, 0., 0.,])['x']

n = n0
print(rho_eq(n0, 0., 0))
print(rho_eq(9*n0, 0., 4))
murange = np.linspace(0, 8., 100)
rho_of_mu = np.array([rho_eq(9*n0, 0., z) for z in murange])
plt.plot(murange, rho_of_mu)
plt.show()
