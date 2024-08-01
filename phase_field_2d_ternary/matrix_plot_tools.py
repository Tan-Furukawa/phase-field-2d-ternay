# %%
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import Any
import numpy as np


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


def plot_ternary(axis: tuple[str, str, str] = ("A", "B", "C")) -> None:
    plt.figure(figsize=(4, 4))
    plt.plot([0.0, 1.0, 0.5, 0.0], [0.0, 0.0, np.sqrt(3.0) / 2, 0])
    padding = 0.3
    plt.xlim((0 - padding, 1 + padding))
    plt.ylim((0 - padding, 1 + padding))
    dtext = 0.07
    text_args: dict = {"va": "center", "ha": "center", "size": 16}
    plt.text(-dtext, -dtext, axis[0], **text_args)
    plt.text(1 + dtext, -dtext, axis[1], **text_args)
    plt.text(1.0 / 2.0, np.sqrt(3.0) / 2 + np.sqrt(2.0) * dtext, axis[2], **text_args)


def plot_ternary_color_map(x: NDArray, y: NDArray, **kwargs: Any) -> None:
    z = 1.0 - x - y
    x_plot = y + 0.5 * z
    y_plot = z * np.sqrt(3.0) / 2
    plt.scatter(x_plot, y_plot, **kwargs)


def generate_triangle_mesh(n: int) -> tuple[NDArray, NDArray]:
    # 節点のリストを初期化
    nodes = []

    # 各行の節点を生成
    for i in range(n):
        # 各行のy座標
        y = i / (n - 1)
        # 各行のx座標の個数
        num_points = n - i
        for j in range(num_points):
            x = j / (n - 1)
            nodes.append((x, y))

    res_x = [n[0] for n in nodes]
    res_y = [n[1] for n in nodes]

    return np.array(res_x), np.array(res_y)


def assign_rgb_to_end_member_color(
    x: NDArray,
    y: NDArray,
    end_member1: tuple[float, float, float] = (2, -0.5, -0.2),
    end_member2: tuple[float, float, float] = (-0.2, 2, -0.5),
    end_member3: tuple[float, float, float] = (-0.5, -0.2, 2),
) -> list[tuple[float, float, float, float]]:

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


if __name__ == "__main__":
    import numpy as np

    # example1
    print("example 1")
    x = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
    plot_con_hist(x)


    # example2
    print("example 2")
    x, y = generate_triangle_mesh(23)
    col = assign_rgb_to_end_member_color(x, y)
    plot_ternary()
    plot_ternary_color_map(x, y, c=col, marker="h")

# %%

x = np.linspace(0,1, num=100)
