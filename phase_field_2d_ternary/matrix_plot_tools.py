# %%
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import matplotlib.contour as contour
from numpy.typing import NDArray
from typing import Any, Callable
import numpy as np
from scipy.interpolate import griddata


def get_matrix_image(
    mat: NDArray,
    vmin: float = 0,
    vmax: float = 1,
    isnorm: bool = True,
    theme: str = "Greys",
    show: bool = True,
) -> None:
    """plot the phase field result matrix

    Args:
        mat (NDArray): numpy 2d NDArray
        vmin (float, optional): min value of plot. The value below vmin is displayed as red blue. Defaults to 0.
        vmax (float, optional): max value of plot. The value above vmax is displayed as red color. Defaults to 1.
        isnorm (bool, optional): plotted as [vmin, vmax] range. Defaults to True.
        theme (str, optional): the theme color. see matplotlib.pyplot.get_cmap. Defaults to "Greys".
    """
    cmap = plt.get_cmap(theme)
    # カラーマップの正規化
    norm = Normalize(vmin=vmin, vmax=vmax)
    # con < 0 のセルを赤色に変更
    cmap.set_under("blue")  # below lower limit blue
    cmap.set_over("red")  # above upper limit red
    if isnorm:
        plt.imshow(mat, cmap=cmap, norm=norm)
        # plt.show()
    else:
        plt.imshow(mat, cmap=cmap)
        # plt.show()
    if show:
        # plt.savefig("tmp.svg")
        plt.colorbar()
        plt.show()

def imshow3(x: NDArray, y: NDArray)->None:
    Ternary.imshow3(x, y)

def plot_con_hist(con: NDArray) -> None:
    """plot composition histogram

    Args:
        con (NDArray): composition matrix

    Example:
        >>> import numpy as np
        >>> x = np.array([[1,2,3,4,5],[2,3,4,5,6]])
        >>> plot_con_hist(x)
    """
    con = con.reshape(1, -1)[0]
    plt.hist(con, bins=200)
    plt.show()

class Ternary():

    @staticmethod
    def remove_all_axis_and_lavel() -> None:
        """
        Removes all axes and labels from the current plot.
        """
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["bottom"].set_visible(False)
        plt.gca().spines["left"].set_visible(False)
        plt.tick_params(
            labelbottom=False,
            labelleft=False,
            labelright=False,
            labeltop=False,
            bottom=False,
            left=False,
            right=False,
            top=False,
        )

    @staticmethod
    def plot_blank_ternary(axis: tuple[str, str, str] = ("A", "B", "C")) -> None:
        """
        Plots a basic ternary diagram with the specified axis labels.

        Args:
            axis (tuple[str, str, str]): The labels for the three axes. Defaults to ("A", "B", "C").
        """

        plt.figure(figsize=(4, 4))
        Ternary.remove_all_axis_and_lavel()

        plt.plot([0.0, 1.0, 0.5, 0.0], [0.0, 0.0, np.sqrt(3.0) / 2, 0])
        padding = 0.3
        plt.xlim((0 - padding, 1 + padding))
        plt.ylim((0 - padding, 1 + padding))
        dtext = 0.07
        text_args: dict = {"va": "center", "ha": "center", "size": 16}
        plt.text(-dtext, -dtext, axis[0], **text_args)
        plt.text(1 + dtext, -dtext, axis[1], **text_args)
        plt.text(1.0 / 2.0, np.sqrt(3.0) / 2 + np.sqrt(2.0) * dtext, axis[2], **text_args)


    @staticmethod
    def convert_ternary_to_Cartesian_coordinate(x: NDArray, y: NDArray) -> tuple[NDArray, NDArray]:

        """
        Converts x + y + z = 1 coordinates to Cartesian coordinates.

        Args:
            x (NDArray): The x-coordinates in ternary space.
            y (NDArray): The y-coordinates in ternary space.

        Returns:
            tuple[NDArray, NDArray]: The x and y coordinates in ternary plot space.
        """
        z = 1.0 - x - y
        x_plot = y + 0.5 * z
        y_plot = z * np.sqrt(3.0) / 2
        return x_plot, y_plot


    @staticmethod
    def plot_ternary_color_map(x: NDArray, y: NDArray, **kwargs: Any) -> None:
        """
        Plots a color map on a ternary plot.

        Args:
            x (NDArray): The x-coordinates in ternary space (x + y + z = 1).
            y (NDArray): The y-coordinates in ternary space (x + y + z = 1).
            **kwargs: Additional keyword arguments for the scatter plot.
        """
        x_plot, y_plot = Ternary.convert_ternary_to_Cartesian_coordinate(x, y)
        plt.scatter(x_plot, y_plot, **kwargs)


    @staticmethod
    def generate_triangle_mesh(n: int) -> tuple[NDArray, NDArray]:
        """
        Generates a triangular mesh grid with `n` points per side.

        Args:
            n (int): The number of points per side of the triangle.

        Returns:
            tuple[NDArray, NDArray]: The x and y coordinates of the mesh grid.
        """
        nodes = []
        for i in range(n):
            y = i / (n - 1)
            num_points = n - i
            for j in range(num_points):
                x = j / (n - 1)
                nodes.append((x, y))

        res_x = [n[0] for n in nodes]
        res_y = [n[1] for n in nodes]

        return np.array(res_x), np.array(res_y)


    @staticmethod
    def assign_rgb_to_end_member_color(
        x: NDArray,
        y: NDArray,
        end_member1: tuple[float, float, float] = (2, -0.5, -0.2),
        end_member2: tuple[float, float, float] = (-0.2, 2, -0.5),
        end_member3: tuple[float, float, float] = (-0.5, -0.2, 2),
    ) -> list[tuple[float, float, float, float]]:
        """
        Assigns RGB colors to points based on their positions relative to three end members.

        Args:
            x (NDArray): The x-coordinates in ternary space.
            y (NDArray): The y-coordinates in ternary space.
            end_member1 (tuple[float, float, float]): The RGB values for the first end member. Defaults to (2, -0.5, -0.2).
            end_member2 (tuple[float, float, float]): The RGB values for the second end member. Defaults to (-0.2, 2, -0.5).
            end_member3 (tuple[float, float, float]): The RGB values for the third end member. Defaults to (-0.5, -0.2, 2).

        Returns:
            list[tuple[float, float, float, float]]: The list of RGBA color values.

        Note:
            the end_menber1, 2, 3 can deviate [0,1]. The deviated value is calculated by assigning 0 or 1 after computation of weighted mean.
        """
        # def color_filter(x, y, z):
        #     k = (x - y)**2 + (y - z)**2 + (z - x)**2
        #     return np.tanh(k**2*40)

        z = 1.0 - x - y
        r = x * end_member1[0] + y * end_member2[0] + z * end_member3[0]
        g = x * end_member1[1] + y * end_member2[1] + z * end_member3[1]
        b = x * end_member1[2] + y * end_member2[2] + z * end_member3[2]
        r[r < 0] = 0
        r[r > 1] = 1
        g[g < 0] = 0
        g[g > 1] = 1
        b[b < 0] = 0
        b[b > 1] = 1

        # t = (np.abs((r - g)) + np.abs(g - b) + np.abs(b - r))/2
        t = np.zeros_like(r) + 1
        # t = ((r - g)**2 + (g - b)**2 + (b - r)**2)/2.
        # print(t)

        return [(t[i] * r[i], t[i] * g[i], t[i] * b[i], 1.0) for i in range(len(r))]

    @staticmethod

    def prepare_contour(
        X: NDArray, Y: NDArray, Z: NDArray, level: int
    ) -> contour.QuadContourSet:
        """
        Prepares contour data for plotting.

        Args:
            X (NDArray): Array of x coordinates.
            Y (NDArray): Array of y coordinates.
            Z (NDArray): Array of z values for contouring.
            level (int): Number of contour levels.

        Returns:
            contour.QuadContourSet: Contour data.
        """
        fig1 = plt.subplot(111)
        contour_result = fig1.contour(X, Y, Z, level)
        fig1.remove()
        return contour_result

    @staticmethod
    def plot_contour(contour_result: contour.QuadContourSet) -> None:
        """
        Plots contour lines on a ternary diagram.

        Args:
            contour_result (contour.QuadContourSet): Contour data to plot. The result of Ternary.prepare_contour.
        """
        for level in contour_result.allsegs:
            for segment in level:
                # print("aaa")
                x = np.array([s[0] for s in segment])
                y = np.array([s[1] for s in segment])
                z = 1.0 - x - y
                x_plot = y + 0.5 * z
                y_plot = z * np.sqrt(3.0) / 2
                x_plot[y_plot < 0] = np.nan
                y_plot[y_plot < 0] = np.nan
                plt.plot(x_plot, y_plot, c="black", lw=1)


    @staticmethod
    def imshow3(x: NDArray, y: NDArray) -> None:
        """
        Displays a matrix image with RGB colors. like plt.imshow.

        Args:
            x (NDArray): Array of x coordinates. same size as y.
            y (NDArray): Array of y coordinates. same size as x.
        """
        x_flat = x.flatten()
        y_flat = y.flatten()
        col = np.array(Ternary.assign_rgb_to_end_member_color(x_flat, y_flat))
        res = col.reshape((x.shape) + (4,))
        plt.imshow(res)


    @staticmethod
    def plot_ternary_contour_and_color_map(func: Callable[[NDArray, NDArray], NDArray], contour_num: int) -> None:
        """
        Plots ternary contours and value map based on a given function.

        Args:
            func (Callable[[NDArray, NDArray], NDArray]): Function to generate values for contouring.
            contour_num (int): Number of contour levels.
        """
        x, y = Ternary.generate_triangle_mesh(100)
        num = 100

        X = np.linspace(0, 1, num)
        Y = np.linspace(0, 1, num)
        X, Y = np.meshgrid(X, Y)
        Z = 1.0 - X - Y
        g = func(x, y)
        G = func(X, Y)
        Z = G

        # これをplotの前にかかないといけない(微妙な実装)
        res = Ternary.prepare_contour(X, Y, Z, contour_num)
        Ternary.plot_blank_ternary()
        Ternary.plot_ternary_color_map(x, y, c=g, cmap="gray")
        Ternary.plot_contour(res)


if __name__ == "__main__":
    import numpy as np
    import cupy as cp
    from phase_field_2d_ternary.free_energy import get_free_energy

    # example1
    print("example 1")
    x = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
    plot_con_hist(x)
    plt.show()

    # example2
    print("example 2")
    x, y = Ternary.generate_triangle_mesh(23)
    col = Ternary.assign_rgb_to_end_member_color(x, y)

    # plt.contour(x,y,col, 5, Vmax=1,colors=['black'])
    Ternary.plot_blank_ternary()
    # plot_contours(points, values)
    Ternary.plot_ternary_color_map(x, y, c=col, marker="h")
    plt.show()

    print("example 3")

    def wrapper(x: NDArray, y: NDArray) -> NDArray:
        _, _, res = get_free_energy(cp.array(x), cp.array(y), 4, 4, 4)
        return cp.asnumpy(res)

    Ternary.plot_ternary_contour_and_color_map(wrapper, 100)
    plt.show()

#%%