# %%
import numpy as np
import cupy as cp
from typing import Any, Callable


class CDArray(cp.ndarray): ...


def check_nan(mat: CDArray, variable_name: str) -> None:
    if np.any(np.isnan((cp.asnumpy(mat)))):
        print(f"{variable_name} = ")
        print(mat)
        raise ValueError(
            f"CDArray should not have nan component but have {variable_name} have."
        )


def is_within_range(mat: CDArray, r: tuple[float, float]) -> bool:
    """
    Check if all elements in the matrix `mat` are within the range specified by `r`.

    Parameters:
    mat (NDArray): A NumPy array of any shape containing numerical values.
    r (tuple[float, float]): A tuple with two float values representing the inclusive range (min, max).

    Returns:
    bool: Returns False if there is any element in `mat` that is less than `r[0]` or greater than `r[1]`, otherwise True.
    """
    return bool(np.all((cp.asnumpy(mat) > r[0]) & (cp.asnumpy(mat) < r[1])))


def in_development(func: Callable) -> Callable:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        print(f"Warning: {func.__name__} is still in development.")
        return func(*args, **kwargs)

    return wrapper


if __name__ == "__main__":

    @in_development
    def example_function() -> None:
        pass

    example_function()
# %%
