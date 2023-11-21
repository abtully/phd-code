"""
Denoising CNN functions
@author: Alexandra Tully
@date: February 2023
"""

import re
from igorwriter import IgorWave
from typing import List, Optional, Union
import logging
import numpy as np

logger = logging.getLogger()


def save_to_igor_itx(
    file_path: str,
    xs: List[np.ndarray],
    datas: List[np.ndarray],
    names: List[str],
    ys: Optional[List[np.ndarray]] = None,
    x_labels: Optional[Union[str, List[str]]] = None,
    y_labels: Optional[Union[str, List[str]]] = None,
):
    """Save data to a .itx file which can be dropped into Igor"""

    def check_axis_linear(
        arr: np.ndarray, axis: str, name: str, current_waves: list
    ) -> bool:
        if arr.shape[-1] > 1 and not np.all(np.isclose(np.diff(arr), np.diff(arr)[0])):
            if np.all(np.isclose(np.diff(x), np.diff(x)[0], atol=1e-5)):
                logger.warning(
                    f"{file_path}: Changed atol from 1e-8 to 1e-5. {axis}-axis might be non-linear."
                )
                return True
            else:
                logger.warning(
                f"{file_path}: Igor doesn't support a non-linear {axis}-axis. Saving as separate wave"
            )
            axis_wave = IgorWave(arr, name=name + f"_{axis}")
            current_waves.append(axis_wave)
            return False
        else:
            return True

    if x_labels is None or isinstance(x_labels, str):
        x_labels = [x_labels] * len(datas)
    if y_labels is None or isinstance(y_labels, str):
        y_labels = [y_labels] * len(datas)
    if ys is None:
        ys = [None] * len(datas)
    assert all([len(datas) == len(list_) for list_ in [xs, names, x_labels, y_labels]])

    waves = []
    for x, y, data, name, x_label, y_label in zip(
        xs, ys, datas, names, x_labels, y_labels
    ):
        wave = IgorWave(data, name=name)
        if x is not None:
            if check_axis_linear(x, "x", name, waves):
                wave.set_dimscale("x", x[0], np.mean(np.diff(x)), units=x_label)
        if y is not None:
            if check_axis_linear(y, "y", name, waves):
                wave.set_dimscale("y", y[0], np.mean(np.diff(y)), units=y_label)
        elif y_label is not None:
            wave.set_datascale(y_label)
        waves.append(wave)

    with open(file_path, "w") as fp:
        for wave in waves:
            wave.save_itx(
                fp, image=True
            )  # Image = True hopefully makes np and igor match in x/y


def fix_itx_format(filename: str = "test.itx"):
    with open(filename, "r") as f:
        lines = f.readlines()
    #     for i in range(4):
    #         lines.append(f.readline())

    for ln in [1, -3, -2, -1]:
        lines[ln] = lines[ln].replace("'", "")

    new = lines[:-6]
    new[1] = "WAVES/D/N=" + lines[1][12:]
    new.append("END\n")

    dscale = lines[-3][2:-1].replace(",", " ", 1)
    xscale = lines[-2][2:-1].replace(",", " ", 1).replace(" ", "", 1)
    yscale = lines[-1][2:-1].replace(",", " ", 1).replace(" ", "", 1)

    new.append(f"X {xscale}; {yscale}; {dscale}".replace('"",', '"", '))

    with open("test_fixed.itx", "w") as f:
        f.writelines(new)


"""This function lives in loading_functions file"""
# def load_denoised_data(file_path: str, filename: str):
#     data_path = os.path.join(file_path, filename)
#     df = pd.read_csv(data_path, skiprows=3, header=None, sep="\s+", skipfooter=2)
#     data = np.array(df).T
#     with open(data_path, "r") as f:
#         lines = f.readlines()
#     last = lines[-1]
#
#     x_start, x_step = [
#         float(v) for v in re.search("x\s*(-?\d*.\d+),\s*(-?\d+.\d+)", last).groups()
#     ]
#     y_start, y_step = [
#         float(v) for v in re.search("y\s*(-?\d*.\d+),\s*(-?\d+.\d+)", last).groups()
#     ]
#
#     x = np.linspace(x_start, x_start + data.shape[1] * x_step, data.shape[1])
#     y = np.linspace(y_start, y_start + data.shape[0] * y_step, data.shape[0])
#     return x, y, data