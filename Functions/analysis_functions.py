"""
General analysis functions
@author: Alexandra Tully
@date: November 2020
"""

from .misc_functions import get_data_index
import numpy as np


# set region of data
def get_data_region(data: np.ndarray, xaxis: np.ndarray, yaxis: np.ndarray,
                    xbounds: tuple = None, ybounds: tuple = None,
                    EB=False
                    ) -> tuple:
    if xbounds is not None:
        data_x = data[:, np.logical_and(xbounds[0] < xaxis, xaxis < xbounds[1])]
        new_xaxis = xaxis[np.logical_and(xbounds[0] < xaxis, xaxis < xbounds[1])]
    elif xbounds is None:
        data_x = data
        new_xaxis = xaxis
    if ybounds is not None:
        if EB:
            new_region = data_x[np.logical_and(ybounds[0] < yaxis - 16.8, yaxis - 16.8 < ybounds[1])]
            new_yaxis = yaxis[np.logical_and(ybounds[0] < yaxis - 16.8, yaxis - 16.8 < ybounds[1])]
        if not EB:
            new_region = data_x[np.logical_and(ybounds[0] < yaxis, yaxis < ybounds[1])]
            new_yaxis = yaxis[np.logical_and(ybounds[0] < yaxis, yaxis < ybounds[1])]
    elif ybounds is None:
        if EB:
            raise ValueError(f'{ybounds} is None, do not specify EB')
        elif not EB:
            pass
        new_region = data_x
        new_yaxis = yaxis
    return new_region, new_xaxis, new_yaxis


def gaussian(x, amplitude, center, width, const):
    return amplitude * np.exp(-(x - center) ** 2 / width) + const


def get_vertical_slice(data, axis, value, interval):
    """
    Gets vertical chunk of data

    Args:
        data: numpy array
        axis: typically x
        value: in the axis unit (or can specify x[500] if you want the 500th index)
        interval: total interval in the axis unit, centered on value

    Returns:
        1D or 2D vertical chunk of data

    """
    high = value + interval / 2
    low = value - interval / 2
    low_index, high_index = get_data_index(axis, (low, high))
    if low_index == high_index:
        chunk = data[:, low_index]
    else:
        chunk = data[:, low_index:high_index]
    return chunk


def get_horizontal_slice(data, axis, value, interval):
    """
    Gets horizontal chunk of data

    Args:
        data: numpy array
        axis: typically y
        value: in the axis unit (or can specify y[500] if you want the 500th index)
        interval: total interval in the axis unit, centered on value

    Returns:
        1D or 2D horizontal chunk of data

    """
    high = value + interval / 2
    low = value - interval / 2
    low_index, high_index = get_data_index(axis, (low, high))
    if low_index == high_index:
        chunk = data[low_index, :]
    else:
        chunk = data[low_index:high_index, :]
    return chunk


def get_averaged_slice(data, axis) -> np.ndarray:
    """
    Averages a slice of data
    Args:
        data: numpy array
        axis: specify x or y

    Returns:
        1D numpy array

    """
    data = np.atleast_2d(data)
    if axis == 'x':
        return np.mean(data, axis=1)
    if axis == 'y':
        return np.mean(data, axis=0)


def get_2Dslice(x: np.ndarray, y: np.ndarray, z: np.ndarray, data: np.ndarray, slice_dim: str, slice_val: float,
                int_range: float = 0):
    """
    Gets 2D slice of 3D data (based on plot_3D function in plotting_functions)
    Args:
        x: 1D numpy array
        y: 1D numpy array
        z: 1D numpy array
        data: 3D numpy array
        int_range: range for integration along slice dimension (float)
        slice_val: center value for slice along slice dimension (float)
        slice_dim: x, y, or z (typically y for constant energy, or z for kx vs energy (constant phi))

    Returns:
        axes_2d[0]: 1D numpy array
        axes_2d[1]: 1D numpy array
        data2d: 2D numpy array

    """
    if slice_dim == 'x':
        a = x
        axis_from = 2
        axes_2d = (y, z)
    elif slice_dim == 'y':
        a = y
        axis_from = 1
        axes_2d = (x, z)
    elif slice_dim == 'z':
        a = z
        axis_from = 0
        axes_2d = (x, y)
    else:
        raise ValueError(f'{slice_dim} is not x, y, or z')
    slice_index = np.argmin(np.abs(a - slice_val))  # gives n; z[n] = value closest to slice_val
    int_index_range = np.floor(int_range / (2 * np.mean(np.diff(a)))).astype(
        int)  # gets avg delta between z, rounds down
    data = np.moveaxis(data, axis_from, 0)
    if int_index_range > 0:
        low = slice_index - int_index_range
        low = low if low > 0 else None
        high = slice_index + int_index_range
        high = high if high < data.shape[0] else None
        data2d = data[low: high].mean(axis=0)
    else:
        data2d = data[slice_index]
    return axes_2d[0], axes_2d[1], data2d


def norm_data(data):
    data = data - np.nanmin(data)
    return 1/np.nanmax(data) * data


def limit_dataset(x, y, data, xlim, ylim, EF=None):
    if xlim is None:
        xlim = (np.min(x), np.max(x))
    if ylim is None:
        ylim = (np.min(y), np.max(y))
    if EF is not None:
        ylim = (ylim[0]+EF, ylim[1]+EF)
    data = data[np.logical_and(y >= ylim[0], y <= ylim[1])]
    data = data[:, np.logical_and(x >= xlim[0], x <= xlim[1])]
    y = y[np.logical_and(y >= ylim[0], y <= ylim[1])]
    x = x[np.logical_and(x >= xlim[0], x <= xlim[1])]
    return x, y, data
