# %%
import numpy as np
import phase_field_2d_ternary.matrix_plot_tools as mplt
import matplotlib.pyplot as plt


from phase_field_2d_ternary.phase_field import PhaseField2d3c


class PhaseFieldBreak(PhaseField2d3c):
    def __init__(
        self,
        threshold: int,
        w12: float,
        w13: float,
        w23: float,
        k11: float = 8,
        k22: float = 8,
        k12: float = 4,
        c10: float = 0.333333,
        c20: float = 0.333333,
        L12: float = -1,
        L13: float = -1,
        L23: float = -1,
        record: bool = False,
        stop_if_error: bool = True,
    ) -> None:
        super().__init__(
            w12,
            w13,
            w23,
            k11,
            k22,
            k12,
            c10,
            c20,
            L12,
            L13,
            L23,
            record,
            stop_if_error,
        )
        self.threshold = threshold

    def check(self) -> bool:
        arr = self.energy_int[~np.isnan(self.energy_int)]
        if self.istep > self.threshold + 1:
            if np.all(np.diff(arr[-self.threshold:]) < 0):
                return False
            else:
                return True
        else:
            return True


s = PhaseFieldBreak(10000, 4, 4, 4)
s.start()

#%%
interf = s.energy_int[100:] - s.energy_int[100]
g = s.energy_g[100:] - s.energy_g[100]

np.array(list(filter(lambda x: not np.isnan(x), s.energy_int)))

plt.plot(np.diff(interf))
# plt.plot(np.diff(g))
plt.plot(np.zeros_like(interf))


# s = PhaseField2d3c(4.1, 3, 3)
# s.dtime = 0.003
# s.start()
# plt.plot(s.energy_g - s.energy_g[0])
# plt.plot(s.energy_int - s.energy_int[0])

# threshold = 10000