"""
Miscellaneous Functions for ARPES Data
@author: Alexandra Tully
@date: November 2020
"""

from typing import Union, List, Optional, Tuple, Iterable
import numpy as np
import os
import igor.binarywave
# import cv2

from arpes_functions.arpes_dataclasses import Data2D

"""Load Datasets"""


def load2D_notkw(ddir, fn, summary=True):
    """
    Notes: When creating axes, use data.shape[1] for x because numpy reads right to left
    (I tranposed data matrix before that).
    """
    if fn[-4:] != '.ibw':
        raise ValueError(f'file is {fn[-4]}, function expects .ibw')
    path = os.path.join(ddir, fn)
    fullfile = igor.binarywave.load(path)

    wave = fullfile['wave']
    version = fullfile['version']
    rows, cols = wave['wave_header']['nDim'][0:2]

    data = np.array(wave['wData'], dtype='float32').T
    deltas, offsets = wave['wave_header']['sfA'], wave['wave_header']['sfB']
    yaxis, xaxis = (np.arange(data.shape[1], dtype='float32') * deltas[0]) + offsets[0], \
                   (np.arange(data.shape[0], dtype='float32') * deltas[1]) + offsets[1]  # create x and y axes

    params = wave['note'].decode('utf-8').split('\r')
    orig_filename = params[2][12:]
    orig_filepath = params[20][5:]
    comments = params[27][9:]
    datetime = params[28][5:] + ' ' + params[29][5:]
    if summary:
        summary = f'\033[1mFile Summary\033[0m\nfilename: {fn}\n' \
                  f'xaxis (theta): {len(xaxis)}, x_min: {np.min(xaxis):.2f}, x_max: {np.max(xaxis):.2f}\n' \
                  f'yaxis (energy): {len(yaxis)}, y_min: {np.min(yaxis):.2f}, y_max: {np.max(yaxis):.2f}\n' \
                  f'datetime: {datetime}\ncomments: {comments}\n' \
                  f'current filepath: {path}\nraw data filepath: {orig_filepath}'

        print(summary)
    return xaxis, yaxis, data.T


def multi_load(scan_numbers: Union[List[int], np.ndarray], scan_type: str = 'Raster',
               month: str = 'October', year: str = '2020', cryo_temp: str = 'RT') -> List[
    Data2D]:
    datas = []
    for num in scan_numbers:
        if scan_type == 'Raster':
            fn = f'Raster{num:04d}_{num:03d}.ibw'
        else:
            raise ValueError(f'{scan_type} is not Raster')
        datas.append(Data2D.single_load(month=month, year=year, filename=fn, cryo_temp=cryo_temp))
    return datas


""" Binning Data"""


#  Alternative method of binning, seems to work but cv2 = black box. Use bin_2d function.
# def _bin_data2D(data, bin_size: int):
#     x_bins = int(np.ceil(len(data.xaxis) / bin_size))
#     y_bins = int(np.ceil(len(data.yaxis) / bin_size))
#     x = np.linspace(data.xaxis[0], data.xaxis[-1], x_bins)
#     y = np.linspace(data.yaxis[0], data.yaxis[-1], y_bins)
#     return x, y, cv2.resize(data.data, dsize=(y_bins, x_bins), interpolation=cv2.INTER_AREA)


# bin_x and bin_y are bin sizes (users choice); num_x is the length of x axis in data array divided by bin_x; we take
# the floor because you can't have bin_x*num_x > data length; create new x axis from data.xaxis[0] to
# data.xaxis[num_x*bin_x] --> new x axis (starts at the beginning of the bins and ends at the end of the bins, probably
# should fix so my x-value shows the middle of every bin) ; then take data up to num_x*bin_x (cut off the ends);
# reshape data array: (-1 =) let y axis be free, have num_x number of entries in x dimension with bin_x number of
# datapoints per bin, then take the mean on that axis (so we have a 3D array but we take the mean in the bin_x dimension
# and collapse back down to a 2D array.

def bin_2D(data, bin_x: int, bin_y: int) -> np.ndarray:  # takes class for data
    # REMEMBER: data is index y, x (i.e. data[y,x] = value at (x, y))
    d = data.data
    num_y, num_x = [np.floor(s / b).astype(int) for s, b in zip(d.shape, [bin_y, bin_x])]
    x = np.linspace(data.xaxis[0], data.xaxis[num_x * bin_x - 1], num_x)
    y = np.linspace(data.yaxis[0], data.yaxis[num_y * bin_y - 1], num_y)
    d = d[:num_y * bin_y, :num_x * bin_x]
    d = d.reshape((-1, num_x, bin_x)).mean(axis=2)
    d = d.reshape((num_y, bin_y, -1)).mean(axis=1)
    return x, y, d


def bin_2D_array(data, xaxis, yaxis, bin_x: int, bin_y: int) -> np.ndarray:
    # REMEMBER: data is index y, x (i.e. data[y,x] = value at (x, y))
    num_y, num_x = [np.floor(s / b).astype(int) for s, b in zip(data.shape, [bin_y, bin_x])]
    x = np.linspace(xaxis[0], xaxis[num_x * bin_x - 1], num_x)
    y = np.linspace(yaxis[0], yaxis[num_y * bin_y - 1], num_y)
    data = data[:num_y * bin_y, :num_x * bin_x]
    data = data.reshape((-1, num_x, bin_x)).mean(axis=2)
    data = data.reshape((num_y, bin_y, -1)).mean(axis=1)
    return x, y, data


def bin_data(data: np.ndarray, bin_x: int = 1, bin_y: int = 1, bin_z: int = 1) -> np.ndarray:
    """
    Bins up to 3D data in x then y then z. If bin_y == 1 then it will only bin in x direction (similar for z)
    )

    Args:
        data (np.ndarray): 1D, 2D or 3D data to bin in x and or y axis and or z axis
        bin_x (): Bin size in x
        bin_y (): Bin size in y
        bin_z (): Bin size in z
    Returns:
        data

    """
    ndim = data.ndim
    data = np.array(data, ndmin=3)
    os = data.shape
    num_z, num_y, num_x = [np.floor(s / b).astype(int) for s, b in zip(data.shape, [bin_z, bin_y, bin_x])]
    # ^^ Floor so e.g. num_x*bin_x does not exceed len x
    chop_z, chop_y, chop_x = [s - n * b for s, n, b in zip(data.shape, [num_z, num_y, num_x], [bin_z, bin_y, bin_x])]
    # ^^ How much needs to be chopped off in total to make it a nice round number
    data = data[
           np.floor(chop_z / 2).astype(int): os[0] - np.ceil(chop_z / 2).astype(int),
           np.floor(chop_y / 2).astype(int): os[1] - np.ceil(chop_y / 2).astype(int),
           np.floor(chop_x / 2).astype(int): os[2] - np.ceil(chop_x / 2).astype(int)
           ]
    rs = data.shape
    data = data.reshape((rs[0], rs[1], num_x, bin_x)).mean(axis=3)
    data = data.reshape((rs[0], num_y, bin_y, num_x)).mean(axis=2)
    data = data.reshape((num_z, bin_z, num_y, num_x)).mean(axis=1)

    if ndim == 3:
        return data
    elif ndim == 2:
        return data[0]
    elif ndim == 1:
        return data[0, 0]
    return data


def bin_data_with_axes(data: np.ndarray,
                       bin_sizes: Union[int, Iterable[int]],
                       x: Optional[np.ndarray] = None,
                       y: Optional[np.ndarray] = None,
                       z: Optional[np.ndarray] = None):
    """
        Resamples given data using self.MAX_POINTS and self.RESAMPLE_METHOD.
        Will always return data, then optionally ,x, y, z incrementally (i.e. can do only x or only x, y but cannot do
        e.g. x, z)
        Args:
            data (): Data to resample down to < self.MAX_POINTS in each dimension
            bin_sizes (): Sizes to bin with in up to 3 dimensions
            x (): Optional x array to resample the same amount as data
            y (): Optional y ...
            z (): Optional z ...

        Returns:
            (Any): Matching combination of what was passed in (e.g. data, x, y ... or data only, or data, x, y, z)
        """

    def check_dim_sizes(data, x, y, z) -> bool:
        """If x, y, z are provided, checks that they match the corresponding data dimension"""
        for arr, expected_shape in zip([z, y, x], data.shape):
            if arr is not None:
                if arr.shape[0] != expected_shape:
                    raise RuntimeError(f'data.shape: {data.shape}, (z, y, x).shape: '
                                       f'({[arr.shape if arr is not None else arr for arr in [z, y, x]]}). '
                                       f'at least one of x, y, z has the wrong shape (None is allowed)')
        return True

    def make_bin_sizes_len_3(orig_bs) -> Tuple[int, int, int]:
        if isinstance(orig_bs, int) or len(orig_bs) == 1:
            return (orig_bs, 1, 1)
        elif len(orig_bs) == 2:
            return (orig_bs[0], orig_bs[1], 1)
        elif len(orig_bs) == 3:
            return tuple(orig_bs)
        else:
            raise ValueError(f'{orig_bs} not valid')

    bin_sizes = make_bin_sizes_len_3(bin_sizes)

    ndim = data.ndim
    data = np.array(data, ndmin=3)
    check_dim_sizes(data, x, y, z)
    data = bin_data(data, *bin_sizes)
    x, y, z = [bin_data(arr, cs) if arr is not None else arr for arr, cs in zip([x, y, z], bin_sizes)]

    if ndim == 1:
        data = data[0, 0]
        if x is not None:
            return data, x
        return data

    elif ndim == 2:
        data = data[0]
        if x is not None:
            if y is not None:
                return data, x, y
            return data, x
        return data

    elif ndim == 3:
        if x is not None:
            if y is not None:
                if z is not None:
                    return data, x, y, z
                return data, x, y
            return data, x
        return data
    raise ValueError(f'Most likely something wrong with {data}')


def get_data_index(data1d, val, is_sorted=False):
    """
    Returns index position(s) of nearest data value(s) in 1d data.
    Args:
        is_sorted (bool): If data1d is already sorted, set sorted = True to improve performance
        data1d (np.ndarray): data to compare values
        val (Union(float, list, tuple, np.ndarray)): value(s) to find index positions of

    Returns:
        Union[int, np.ndarray]: index value(s)

    """

    def find_nearest_index(array, value):
        idx = np.searchsorted(array, value, side="left")
        if idx > 0 and (idx == len(array) or abs(value - array[idx - 1]) < abs(
                value - array[idx])):  # TODO: if abs doesn't work, use math.fabs
            return idx - 1
        else:
            return idx

    data = np.asarray(data1d)
    val = np.atleast_1d(np.asarray(val))
    nones = np.where(val == None)
    if nones[0].size != 0:
        val[nones] = np.nan  # Just to not throw errors, will replace with Nones before returning
    assert data.ndim == 1
    if is_sorted is False:
        arr_index = np.argsort(data)  # get copy of indexes of sorted data
        data = np.sort(data)  # Creates copy of sorted data
        index = arr_index[np.array([find_nearest_index(data, v) for v in val])]
    else:
        index = np.array([find_nearest_index(data, v) for v in val])
    index = index.astype('O')
    if nones[0].size != 0:
        index[nones] = None
    if index.shape[0] == 1:
        index = index[0]
    return index


def sanitize(value, target_type, default_val):
    if not value:
        return default_val
    if target_type == 'float':
        return float(value)
    if target_type == 'csv-to-valuelist':
        return [float(v) for v in value.split(',')]


def line_intersection(fit1, fit2):
    a1, b1 = fit1.best_values['i0_slope'], fit1.best_values['i0_intercept']
    a2, b2 = fit2.best_values['i0_slope'], fit2.best_values['i0_intercept']
    x = np.asarray((b2 - b1)/(a1 - a2))
    y = np.asarray(a1 * x + b1)
    y_check = np.asarray(a2 * x + b2)
    if y != y_check:
        raise ValueError(f'({x}, {y}) != ({x}, {y_check}), fits: y1 = {a1}x + {b1}, y2 = {a2}x + {b2}')
    return x, y


def limit_coords(x_coords, y_coords, xlim, ylim=None):
    x_subset=x_line[np.where(np.logical_and(x_coords > xlim[0], x_coords < xlim[1]))]
    y_subset=y_line[np.where(np.logical_and(x_coords > xlim[0], x_coords < xlim[1]))]
    if ylim:
        x_subset=x_subset[np.where(np.logical_and(y_coords > ylim[0], y_coords < ylim[1]))]
        y_subset=y_subset[np.where(np.logical_and(y_coords > ylim[0], y_coords < ylim[1]))]
    return x_subset, y_subset



