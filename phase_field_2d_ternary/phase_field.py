# %%
import cupy as cp
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from yaml import dump

from free_energy import get_free_energy
from initial_distribution import make_initial_distribution
from prepare_fft import prepare_fft
import phase_field_2d_ternary.matrix_plot_tools as mplt
import save as mysave

from error_check import ErrorCheck


class CDArray(cp.ndarray): ...


class PhaseField2d3c:
    """2D phase field modeling of non-elastic ternary system.

    This class models a two-dimensional phase field for a non-elastic ternary system.
    It uses Margules parameters to describe the interaction between the components
    and computes the evolution of the system over time using FFT.

    Attributes:
        w12 (float): Margules parameter between components 1 and 2.
        w13 (float): Margules parameter between components 1 and 3.
        w23 (float): Margules parameter between components 2 and 3.
        c10 (float): Initial composition of component 1.
        c20 (float): Initial composition of component 2.

        Nx (int): Number of grid points in the x direction (default: 128).
        Ny (int): Number of grid points in the y direction (default: 128).
        dx (float): Grid spacing in the x direction (default: 1.0).
        dy (float): Grid spacing in the y direction (default: 1.0).
        nstep (int): Number of time steps for the simulation (default: 100000).
        nprint (int): Interval of steps at which the results are printed and visualized (default: 1000).
        dtime (float): Time increment for each simulation step (default: 1e-2).
        ttime (float): Total simulation time (default: 0.0).
        noise (float): Initial noise level for the composition distribution (default: 0.01).

        L12 (float): Coupling parameter between components 1 and 2 (default: -1.0).
        L13 (float): Coupling parameter between components 1 and 3 (default: -1.0).
        L23 (float): Coupling parameter between components 2 and 3 (default: -1.0).

        k11 (float): Gradient energy coefficient for component 1 (default: 16).
        k22 (float): Gradient energy coefficient for component 2 (default: 16).
        k12 (float): Cross-gradient energy coefficient (default: 8).

        istep (int): Current simulation step (default: 0).

        energy_g (NDArray[np.float64]): Array to store free energy values at each step.
        energy_el (NDArray[np.float64]): Array to store elastic energy values at each step.
        con1 (CDArray): Composition array for component 1.
        con2 (CDArray): Composition array for component 2.
        con1k (CDArray): Fourier-transformed composition array for component 1.
        con2k (CDArray): Fourier-transformed composition array for component 2.
        g (CDArray): Free energy density array.
        dfdcon1 (CDArray): Derivative of free energy with respect to composition of component 1.
        dfdcon1k (CDArray): Fourier-transformed derivative of free energy with respect to composition of component 1.
        dfdcon2 (CDArray): Derivative of free energy with respect to composition of component 2.
        dfdcon2k (CDArray): Fourier-transformed derivative of free energy with respect to composition of component 2.
    """

    def __init__(
        self,
        w12: float,
        w13: float,
        w23: float,
        c10: float,
        c20: float,
    ) -> None:
        """
        Args:
            w12 (float): Margulus parameter between 1 and 2
            w13 (float): Margulus parameter between 1 and 3
            w23 (float): Margulus parameter between 2 and 3
            c10 (float): the initial composition of 1
            c20 (float): the initial composition of 2
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
        self.noise: float = 0.1
        # self.noise: float = 0.1

        self.L12: float = -1.0
        self.L13: float = -1.0
        self.L23: float = -1.0

        self.k11: float = 8
        self.k22: float = 8
        self.k12: float = 4

        self.istep: int = 0
        self.contour_level: int = 100

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
        con1 = make_initial_distribution(self.Nx, self.Ny, self.c10, self.noise, 123)
        self.con1 = cp.array(con1)
        con2 = make_initial_distribution(self.Nx, self.Ny, self.c20, self.noise, 234)
        self.con2 = cp.array(con2)

    # developing
    # 初期分布に適当な不均一性を与える
    def set_initial_heterogenesis_composition(self) -> None:
        d = 5
        Nx, Ny = self.con1.shape
        self.con1[
            int(Nx / 2) - d : int(Nx / 2) + d, int(Ny / 2) - d : int(Ny / 2) + d
        ] = 0.5

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
                mplt.Ternary.imshow3(con1_res, con2_res)
                plt.show()

                self.plot_ternary_contour_and_composition(con1_res, con2_res)
                plt.show()

                # x_flat = con1_res.flatten()
                # y_flat = con2_res.flatten()
                # col = np.array(mplt.assign_rgb_to_end_member_color(x_flat, y_flat))
                # res = col.reshape((con1_res.shape) + (4,))
                # plt.imshow(res)
                # plt.savefig("test.pdf")

    def plot_ternary_contour_and_composition(
        self, con1: NDArray, con2: NDArray
    ) -> None:
        con1_res = cp.asnumpy(con1.transpose())
        con2_res = cp.asnumpy(con2.transpose())

        def wrapper(x: NDArray, y: NDArray) -> NDArray:
            _, _, res = get_free_energy(
                cp.array(x), cp.array(y), self.w12, self.w23, self.w13
            )
            return cp.asnumpy(res)

        mplt.Ternary.plot_ternary_contour_and_color_map(wrapper, self.contour_level)
        x_plot, y_plot = mplt.Ternary.convert_ternary_to_Cartesian_coordinate(
            con1_res.flatten(), con2_res.flatten()
        )
        choice = np.random.choice(len(x_plot), 1000)
        plt.scatter(x_plot[choice], y_plot[choice], c="red", s=0.3)

    def summary(self) -> None:
        """result summary at t=istep"""
        print("-------------------------------------")
        print(f"phase filed result at t={self.istep}")
        print("-------------------------------------")

        con1 = cp.asnumpy(self.con1)
        con2 = cp.asnumpy(self.con2)

        print("composition")
        mplt.imshow3(con1, con2)
        plt.show()

        print(f"composition distribution result at t={self.istep}")
        mplt.plot_con_hist(con1)
        plt.show()

    def save(self) -> None:
        mysave.create_directory("result")
        dirname = mysave.make_dir_name()
        mysave.create_directory(f"result/{dirname}")
        np.save(
            f"result/{dirname}/con1_{int(self.istep*self.dtime)}.npy",
            self.con1,
        )
        np.save(
            f"result/{dirname}/con2_{int(self.istep*self.dtime)}.npy",
            self.con1,
        )
        instance_dict = mysave.instance_to_dict(
            self,
            ["w12", "w13", "w23", "c10", "c20", "Nx", "Ny", "dx",
              "dy", "nstep", "nprint", "dtime", "ttime", "noise", "L12",
              "L13", "L23", "k11", "k22", "k12", "istep",],
        )
        yaml_str = dump(instance_dict)
        mysave.save_str(f"result/{dirname}/test.yaml", yaml_str)


if __name__ == "__main__":
    # c01の値によりめっちゃびんかんにかわる
    phase_field = PhaseField2d3c(4, 4, 4, 0.33333, 0.33333)
    phase_field.dtime = 0.01
    phase_field.start()
    phase_field.summary()

    # %%
    phase_field.save()

# %%
