# %%

import cupy as cp


class CDArray(cp.ndarray): ...


def get_free_energy(
    con1: CDArray, con2: CDArray, w12: float, w23: float, w13: float
) -> tuple[CDArray, CDArray, CDArray]:
    """calculate free energy of the system. The free energy is formulated as x log x + (1-x) log (1-x) + w x (1-x) in this function.

    ## Args:
        con (CDArray): n * n cupy array
        w (float): float, Margulus parameter divided by RT

    ## Returns:
        tuple[CDArray, CDArray]: dg/dx, g where g is molar free energy, x is partial mole fraction.
    """
    min_c = 0.001
    max_c = 0.999

    # calculate dg/dx
    con1[con1 < min_c] = min_c
    con1[con1 > max_c] = max_c
    con2[con2 < min_c] = min_c
    con2[con2 > max_c] = max_c
    # dfdcon = w * (1 - 2 * con) + (cp.log(con) - cp.log(1 - con))
    dfdcon1 = (
        cp.log(con1)
        - cp.log(1 - con1 - con2)
        + w12 * con2
        - w23 * con2
        + w13 * (1 - 2 * con1 - con2)
    )
    dfdcon2 = (
        cp.log(con2)
        - cp.log(1 - con1 - con2)
        + w12 * con1
        - w13 * con1
        + w23 * (1 - con1 - 2 * con2)
    )

    g = (
        w12 * con1 * con2
        + w13 * con1 * (1 - con1 - con2)
        + w23 * con2 * (1 - con1 - con2)
        + (
            con1 * cp.log(con1)
            + con2 * cp.log(con2)
            + (1 - con1 - con2) * cp.log(1 - con1 - con2)
        )
    )

    return dfdcon1, dfdcon2, g


# import numpy as np
# from numpy.typing import NDArray
# x: NDArray = np.array([1,2,3])
# get_free_energy(x, 0.2)
# %%
