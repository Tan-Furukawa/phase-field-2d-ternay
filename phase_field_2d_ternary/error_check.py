import numpy as np
import cupy as cp
from typing import Any


class CDArray(cp.ndarray): ...


class ErrorCheck:
    def __init__(self) -> None: ...

    @staticmethod
    def check_nan(mat: CDArray, variable_name: str) -> Any:
        if np.any(np.isnan((cp.asnumpy(mat)))):
            print(f"{variable_name} = ")
            print(mat)
            raise ValueError(f"CDArray have nan component occured at {variable_name}.")