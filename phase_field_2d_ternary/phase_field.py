# %%
import cupy as cp
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

from free_energy import get_free_energy
from initial_distribution import make_initial_distribution
from prepare_fft import prepare_fft
import phase_field_2d_ternary.matrix_plot_tools as myplt
import save as mysave

from error_check import ErrorCheck



class CDArray(cp.ndarray): ...


class PhaseField_2d_3c:
    """2D phase field modeling of non elastic system"""

    def __init__(
        self,
        w12: float,
        w13: float,
        w23: float,
        c10: float,
        c20: float,
    ) -> None:
        """__init__

        Args:
            self (Self): Self
            w (float): Margulus parameter
            temperature (float): annealing temperature (K)
        """
        self.w12 = w12
        self.w13 = w13
        self.w23 = w23
        self.c10 = c10
        self.c20 = c20

        self.Nx = 128
        self.Ny = 128
        self.dx: float = 1.0
        self.dy: float = 1.0
        self.nstep: int = 100000
        self.nprint: int = 1000
        self.dtime: float = 1e-2
        self.ttime: float = 0.0
        self.noise: float = 0.01

        self.L12: float = -1.0
        self.L13: float = -1.0
        self.L23: float = -1.0

        self.k11: float = 16
        self.k22: float = 16
        self.k12: float = 8

        self.istep: int = 0

    def update(self) -> None:
        """update properties computed from the variables defined in __init__()."""
        self.prepare_result_array()
        self.calc_fft_parameters()
        self.set_initial_distribution()

    def start(self) -> None:
        """start all computation."""
        self.update()
        self.compute_phase_field()

    def prepare_result_array(self) -> None:
        """prepare the computation result matrix and array."""
        self.energy_g = np.zeros(self.nstep) + np.nan
        self.energy_el = np.zeros(self.nstep) + np.nan

        self.con1: CDArray = cp.zeros((self.Nx, self.Ny))
        self.con2: CDArray = cp.zeros((self.Nx, self.Ny))

        self.con1k: CDArray = cp.zeros((self.Nx, self.Ny))
        self.con2k: CDArray = cp.zeros((self.Nx, self.Ny))

        self.g: CDArray = cp.zeros((self.Nx, self.Ny))
        self.dfdcon1: CDArray = cp.zeros((self.Nx, self.Ny))
        self.dfdcon1k: CDArray = cp.zeros((self.Nx, self.Ny))

        self.dfdcon2: CDArray = cp.zeros((self.Nx, self.Ny))
        self.dfdcon2k: CDArray = cp.zeros((self.Nx, self.Ny))

    def calc_fft_parameters(
        self,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """calculate parameters which need to FFT.

        ## Args:
            self (Self): Self

        ## Returns:
            tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: kx, ky, k2, k4
        """
        kx, ky, k2, k4 = prepare_fft(self.Nx, self.Ny, self.dx, self.dy)
        self.kx = cp.array(kx)
        self.ky = cp.array(ky)
        self.k2 = cp.array(k2)
        self.k4 = cp.array(k4)

        return kx, ky, k2, k4

    def set_initial_distribution(self) -> None:
        """calculate initial composition and set to the property."""
        con1 = make_initial_distribution(self.Nx, self.Ny, self.c10, self.noise)
        self.con1 = cp.array(con1)
        con2 = make_initial_distribution(self.Nx, self.Ny, self.c20, self.noise)
        self.con2 = cp.array(con2)


    def compute_phase_field(self) -> None:
        """compute main part of phase field."""
        for istep in range(1, self.nstep + 1):
            self.istep = istep

            ErrorCheck.check_nan(self.con1, "self.con1")
            ErrorCheck.check_nan(self.con2, "self.con2")

            # print(np.any(cp.asnumpy(.isnun(self.con1)).flatten()))
            if np.any(np.isnan((cp.asnumpy(self.con1)))):
                raise ValueError("")
            # print(self.con1)

            self.dfdcon1, self.dfdcon2, self.g = get_free_energy(
                self.con1, self.con2, self.w12, self.w23, self.w13
            )

            self.energy_g[istep - 1] = cp.sum(self.g)

            self.con1k = cp.fft.fft2(self.con1)
            self.dfdcon1k = cp.fft.fft2(self.dfdcon1)

            self.con2k = cp.fft.fft2(self.con2)
            self.dfdcon2k = cp.fft.fft2(self.dfdcon2)

            p1 = self.k2 * (
                (self.L12 + self.L13) * self.dfdcon1k - self.L12 * self.dfdcon2k
            )
            p2 = self.k2 * (
                (self.L12 + self.L23) * self.dfdcon2k - self.L12 * self.dfdcon1k
            )

            a11 = -self.k4 * (self.L12 * self.k12 - self.k11 * (self.L12 + self.L13))
            a12 = -self.k4 * (self.L12 * self.k22 - self.k12 * (self.L12 + self.L13))
            a21 = -self.k4 * (self.L12 * self.k11 - self.k12 * (self.L12 + self.L23))
            a22 = -self.k4 * (self.L12 * self.k12 - self.k22 * (self.L12 + self.L23))

            d11 = 1 - self.dtime * a11
            d12 = -self.dtime * a12
            d21 = -self.dtime * a21
            d22 = 1 - self.dtime * a22

            denom = d11 * d22 - d12 * d21

            tmp1 = self.con1k
            tmp2 = self.con2k

            self.con1k = (
                d22 * (tmp1 + self.dtime * p1) - d12 * (tmp2 + self.dtime * p2)
            ) / denom
            self.con1 = np.real(cp.fft.ifft2(self.con1k))

            self.con2k = (
                -d21 * (tmp1 + self.dtime * p1) + d11 * (tmp2 + self.dtime * p2)
            ) / denom
            self.con2 = np.real(cp.fft.ifft2(self.con2k))

            if (istep % self.nprint == 0) or (istep == 1):
                # plt.imshow(cp.asnumpy(cp.abs(1 + a11 * self.con1k + a12 * self.con2k)))
                # plt.colorbar()
                # plt.show()
                # plt.imshow(con_disp)の図の向きは、
                # y
                # ↑
                # |
                # + --→ x [100]
                # となる。
                con1_res = cp.asnumpy(self.con1.transpose())
                con2_res = cp.asnumpy(self.con2.transpose())
                x_flat = con1_res.flatten()
                y_flat = con2_res.flatten()

                col = np.array(myplt.assign_rgb_to_end_member_color(x_flat, y_flat))
                res = col.reshape((con1_res.shape) + (4,))
                plt.imshow(res)
                plt.show()
                # plt.savefig("test.pdf")

    def summary(self) -> None:
        """result summary at t=istep"""
        print("-------------------------------------")
        print(f"phase filed result at t={self.istep}")
        print("-------------------------------------")

        con1 = cp.asnumpy(self.con1)

        print("composition")
        myplt.get_matrix_image(con1)

        print(f"composition distribution result at t={self.istep}")
        myplt.plot_con_hist(con1)

    # def save(self) -> None:
    #     mysave.create_directory("result")
    #     dirname = mysave.make_dir_name()
    #     mysave.create_directoryj(f"result/{dirname}")
    #     np.save(
    #         f"result/{dirname}/con_c0_{self.c0}-w_{self.w}-T_{self.T}-t_{int(self.istep*self.dtime)}.npy",
    #         self.con,
    #     )


if __name__ == "__main__":
    phase_field = PhaseField_2d_3c(3, 3, 3, 0.5, 0.8)
    phase_field.dtime = 0.01
    phase_field.start()

# %%
