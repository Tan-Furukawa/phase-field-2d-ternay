# %%

from phase_field_2d_ternary.free_energy import get_free_energy
import numpy as np
from numpy.typing import NDArray
import cupy as cp
import matplotlib.pyplot as plt

# from phase_field_2d_ternary.free_energy import get_free_energy
from phase_field_2d_ternary import PhaseField2d3c
from phase_field_2d_ternary.matrix_plot_tools import Ternary

c1 = np.linspace(0.001, 0.999, 100)
c2 = np.linspace(0.001, 0.999, 100)
c1, c2 = np.meshgrid(c1, c2)

_, _, g_ = get_free_energy(cp.array(c1), cp.array(c2), -0.1, -0.1, -10)
g = cp.asnumpy(g_)

dg_dx = np.gradient(g, axis=1)
dg_dy = np.gradient(g, axis=0)

d2z_dx2 = np.gradient(dg_dx, axis=1)
d2z_dy2 = np.gradient(dg_dy, axis=0)
d2z_dxdy = np.gradient(dg_dx, axis=0)

# plt.imshow(cp.asnumpy(g))
# np.gradient(g)
z = d2z_dx2 * d2z_dy2 - d2z_dxdy * d2z_dxdy
r = 10
Nx, Ny = z.shape
c1 = c1.flatten()
c2 = c2.flatten()
z = z.flatten()
c2[c1 + c2 >= 0.99] = np.nan
zz = z
zz[z > 0] = 1
zz[z <= 0] = -1
plt.scatter(c1, c2, c=zz, s=4)
plt.colorbar()
plt.show()

#%%


def wrapper(x: NDArray, y: NDArray) -> NDArray:
    _, _, res = get_free_energy(cp.array(x), cp.array(y), 10, 10, 10)
    return cp.asnumpy(res)

Ternary.plot_ternary_contour_and_color_map(wrapper, 100)
plt.show()

#%%
k = 100
s = PhaseField2d3c(k, k, k, c10=0.333, c20=0.3333)
s.dtime = 0.0000001
s.noise = 0.01
s.contour_level = 100
s.start()

#%%

def test(c1: NDArray, c2: NDArray, w12: float, w13: float, w23: float) -> NDArray:
    l = w12**2 + w13**2 + w23**2 - 2 * (w12 * w23 + w23 * w13 + w13 * w12)
    c3 = 1 - c1 - c2
    return 1 - 2 * (w12 * c1 * c2 + w23 * c2 * c3 + w13 * c1 * c3) - l * c1 * c2 * c3

k = 10
c1, c2 = Ternary.generate_triangle_mesh(200)
z = test(c1, c2, k, k, k)
zz = z
zz[z > 0] = 1
zz[z < 0] = -1
Ternary.plot_ternary_color_map(c1, c2, c=z, s=.4)
plt.colorbar()

# %%
