"""
Functions for time-resolved ARPES
@author: Alexandra Tully
@date: February 2023
"""
import copy

import h5py
import os
import numpy as np
from dataclasses import dataclass
from tqdm.auto import tqdm
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d, interp2d, LinearNDInterpolator, RegularGridInterpolator
from functools import lru_cache

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

import sys

from arpes_functions import analysis_functions, misc_functions, plotting_functions


@dataclass
class ArpesData:
    data: np.ndarray
    theta: np.ndarray = None
    energy: np.ndarray = None
    phi_or_time: np.ndarray = None


@dataclass
class Data:
    data: np.ndarray
    x: np.ndarray = None
    y: np.ndarray = None
    z: np.ndarray = None


@dataclass
class ArpesAttrs:
    notes: str
    start: float
    end: float
    temperature: float


def load_attrs_hdf5(fp, fn=None):
    if fn is not None:
        filepath = os.path.join(fp, fn)
    else:
        filepath = fp
    with h5py.File(filepath, "r") as f:  # Read only
        try:
            entry = f["entry1"]
            notes = entry.attrs.get("notes", None)
            start = entry["StartTime"][0]
            end = entry["EndTime"][0]
            temperature = entry["Sample"]["temperature"][0]
        except:
            notes = None
            start = 0
            end = 1
            temperature = -1
    return ArpesAttrs(notes=notes, start=start, end=end, temperature=temperature)


# def get_arpes_data(filenames: List[str]) -> ArpesData:
#     datas = [ARPES_DATA[f] for f in filenames]
#     all_data = np.stack([data.data for data in datas])
#     datas[0].data = np.mean(all_data, axis=0)
#     return datas[0]


# def get_sliced_data(filenames, slice_dim, slice_val, int_range):
#     ad = get_arpes_data(filenames)
#
#     # Get 2D data
#     x, y, d = analysis_functions.get_2Dslice(
#         x=ad.theta,
#         y=ad.energy,
#         z=ad.phi_or_time,
#         data=ad.data,
#         slice_dim=slice_dim,
#         slice_val=slice_val,
#         int_range=int_range,
#     )
#     if slice_dim == "x":
#         return y, x, d.T
#     return x, y, d


def mm_to_ps(mm_val, time_zero=0):
    delay = ((mm_val - time_zero) * 1e-3 * 2) / (3e8)
    return delay * 1e12  # e-12 is pico, e-15 is femto


def ps_to_mm(val, time_zero_mm=0):
    delay = val * 3e8 / (1e-3 * 2) + time_zero_mm
    return delay / 1e12


def sig_to_fwhm(sigma):
    return 2 * sigma * np.sqrt(2 * np.log(2))


def fwhm_to_sig(fwhm):
    return fwhm / (2 * np.sqrt(2 * np.log(2)))


def default_fig():
    """To make it easy to change the style of the figures later"""
    fig = go.Figure()
    fig.update_layout(
        {
            "template": "plotly_white",
            "xaxis": {
                "mirror": True,
                "ticks": "outside",
                "showline": True,
                "linecolor": "black",
            },
            "yaxis": {
                "mirror": True,
                "ticks": "outside",
                "showline": True,
                "linecolor": "black",
            },
        }
    )
    fig.update_layout(
        newshape=dict(
            line=dict(color="black", width=8, dash="solid"),
            #             line_color="yellow",
            #             fillcolor="rgba(0,0,0,0.51)",
            #             opacity=1,
        ),
        # margin=dict(b=0, t=30, l=20, r=0),
        margin=dict(l=100),
        width=800,
        height=600,
    )
    return fig


def thesis_fig(title="Title", xaxis_title="$x$", yaxis_title="$y$", dtick_x=None, dtick_y=None, colorscale="ice",
               reversescale=True, showscale=True, width=900, height=600, equiv_axes=True, gridlines=True):
    fig = default_fig()

    fig.update_coloraxes(colorscale=colorscale, reversescale=reversescale, showscale=showscale)

    fig.update_layout(
        title=dict(
            text=title, x=0.5, xanchor="center", yanchor="top", font_size=28
        ),
    )
    fig.update_xaxes(
        title_text=xaxis_title, title_font=dict(size=24), tickfont_size=20, dtick=dtick_x,
    )
    fig.update_yaxes(
        title_text=yaxis_title, title_font=dict(size=24), tickfont_size=20, dtick=dtick_y,
    )
    if equiv_axes:
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
    if gridlines is False:
        fig.update_yaxes(showgrid=False, zeroline=False)
        fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_layout(
        width=width, height=height, margin=dict(l=130, b=100)
    )  # margin=dict(b=0, t=30, l=20, r=0)
    return fig


## TODO: add definition of ARPES_DATA to this function (see Jupyter notebooks)
# def average_timescans(files, ddir, new_filename):
#     datas = []
#     for i in range(0, len(files)):
#         ad = ARPES_DATA[files[i]]
#         datas.append(ad.data)
#     data_avg = np.mean(datas, axis=0)
#     print(data_avg.shape)
#
#     new_data = data_avg
#
#     new_fn = os.path.join(ddir, new_filename)
#
#     with h5py.File(
#         new_fn, "w"
#     ) as f:  # Note: 'w' creates a new empty file (or overwrites), use 'r+' to modify an existing file
#         f["data"] = new_data.T
#         axes_names = [
#             "angles",
#             "energies",
#         ]  # Change these to match your axes labels
#         axes = [ad.theta, ad.energy]
#         for axis, name in zip(axes, axes_names):
#             f[name] = np.atleast_2d(axis).T
#         entry_group = f.require_group("entry1")
#         entry_group["ScanValues"] = np.atleast_2d(ad.phi_or_time).T
#     return new_fn


def get_sliced_data(ad_dataclass, slice_dim, slice_val, int_range):
    """

    Args:
        ad_dataclass: dataclass
        slice_dim: x, y, or z
        slice_val: center value for 2D slice
        int_range: integration range for 2D slice

    Returns: xaxis, yaxis, data (all np arrays)

    """
    ad = ad_dataclass

    # Get 2D data
    x, y, d = analysis_functions.get_2Dslice(
        x=ad.theta,
        y=ad.energy,
        z=ad.phi_or_time,
        data=ad.data,
        slice_dim=slice_dim,
        slice_val=slice_val,
        int_range=int_range,
    )
    if slice_dim == "x":
        return y, x, d.T
    return x, y, d


def get_1d_y_slice(x, y, data, xlims, y_range):
    if xlims is None:
        xlims = np.min(x), np.max(x)
    if y_range is None:
        y_range = np.min(y), np.max(y)
    x, y, d = analysis_functions.limit_dataset(
        x=x, y=y, data=data, xlim=xlims, ylim=[np.nanmin(y), np.nanmax(y)]
    )
    yval = np.mean(y_range)
    slice_window = np.diff(y_range)[0] / 2  # Half difference between values
    row = analysis_functions.get_averaged_slice(
        analysis_functions.get_horizontal_slice(
            data=d, axis=y, value=yval, interval=slice_window
        ),
        axis="y",
    )
    return x, row


def get_1d_x_slice(x, y, data, ylims, x_range):
    if ylims is None:
        ylims = np.min(y), np.max(y)
    if x_range is None:
        x_range = np.min(x), np.max(x)
    x, y, d = analysis_functions.limit_dataset(
        x=x, y=y, data=data, ylim=ylims, xlim=[np.nanmin(x), np.nanmax(x)]
    )
    xval = np.mean(x_range)
    slice_window = np.diff(x_range)[0] / 2  # Half difference between values
    col = analysis_functions.get_averaged_slice(
        analysis_functions.get_vertical_slice(
            data=d, axis=x, value=xval, interval=slice_window
        ),
        axis="x",
    )
    return y, col


# @lru_cache
# def _get_intensity(filenames, slice_dim, int_range, x_range, y_range):
#     ad = get_arpes_data(filenames)
#     if slice_dim == "y":
#         data = Data(
#             x=ad.theta,
#             y=ad.phi_or_time,
#             z=ad.energy,
#             data=np.moveaxis(ad.data, (0, 1, 2), (1, 0, 2)),
#         )
#     elif slice_dim == "z":
#         data = Data(
#             x=ad.theta,
#             y=ad.energy,
#             z=ad.phi_or_time,
#             data=np.moveaxis(ad.data, (0, 1, 2), (0, 1, 2)),
#         )
#     else:
#         raise NotImplementedError(
#             f"Intensity for slice_dim {slice_dim} not implemented"
#         )
#
#     if x_range is None or y_range is None:
#         x_range = np.min(data.x), np.max(data.x)
#         y_range = np.min(data.y), np.max(data.y)
#
#     x_indexes = analysis_functions.get_data_index(data.x, x_range)
#     y_indexes = analysis_functions.get_data_index(data.y, y_range)
#
#     x_slice = np.s_[x_indexes[0] : x_indexes[1]]
#     y_slice = np.s_[y_indexes[0] : y_indexes[1]]
#     data.x = data.x[x_slice]
#     data.y = data.y[y_slice]
#     data.data = data.data[:, y_slice, x_slice]
#
#     intensities = np.mean(data.data, axis=(1, 2))
#
#     smooth_num = round(int_range / np.mean(np.diff(data.z)))
#     if smooth_num > 3:
#         intensities = savgol_filter(intensities, smooth_num, polyorder=3)
#
#     return data.z, intensities
#
#
# def get_intensity(filenames, slice_dim, int_range, x_range, y_range):
#     return _get_intensity(
#         tuple(filenames),
#         slice_dim,
#         int_range,
#         tuple(x_range) if x_range is not None else None,
#         tuple(y_range) if y_range is not None else None,
#     )


def slice_datacube(
        ad_dataclass,
        slice_dim,
        slice_val,
        int_range=0.05,
        xlim=None,
        ylim=None,
        x_bin=1,
        y_bin=1,
        norm_data=True,
        plot_data=False,
):
    """

    Args:
        ad_dataclass: dataclass
        slice_dim: x, y, or z
        slice_val: center value for 2D slice
        int_range: integration range for 2D slice
        xlim: limits dataset and xaxis (before binning)
        ylim: limits dataset and yaxis (before binning)
        x_bin: bins dataset, preserves axis(?)
        y_bin: bins dataset, preserves axis(?)
        norm_data: takes True or False
        plot_data: can be False, 'mpl' or 'plotly'

    Returns:
        x, y, data (all np arrays)

    """
    x, y, d = get_sliced_data(
        ad_dataclass=ad_dataclass,
        slice_dim=slice_dim,
        slice_val=slice_val,
        int_range=int_range)

    # x, y, d = analysis_functions.get_2Dslice(
    #     x=ad_dataclass.theta,
    #     y=ad_dataclass.energy,
    #     z=ad_dataclass.phi_or_time,
    #     data=ad_dataclass.data,
    #     slice_dim=slice_dim,
    #     slice_val=slice_val,
    #     int_range=int_range,
    # )
    #
    # if slice_dim == "x":
    #     x, y, d = y, x, d.T

    if xlim is None:
        xlim = (np.min(x), np.max(x))

    if ylim is None:
        ylim = (np.min(y), np.max(y))

    x_2d, y_2d, d_2d = analysis_functions.limit_dataset(
        x=x, y=y, data=d, xlim=xlim, ylim=ylim
    )

    if norm_data:
        d_2d = analysis_functions.norm_data(d_2d)

    # Bin Data
    d_2d = misc_functions.bin_data(data=d_2d, bin_x=x_bin, bin_y=y_bin)
    x_2d = misc_functions.bin_data(data=x_2d, bin_x=x_bin)
    y_2d = misc_functions.bin_data(data=y_2d, bin_x=y_bin)

    # Plot Data
    if plot_data == 'mpl':
        fig, ax = plotting_functions.plot_2D_mpl(
            x=x_2d,
            y=y_2d,
            data=d_2d,
            xlabel="x",
            ylabel="y",
            title=f"2D slice",
            # cmap="gray",
            cmap="Blues",
        )
        ratio = 1  # set aspect ratio
        x_left, x_right = ax.get_xlim()
        y_low, y_high = ax.get_ylim()
        ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)
        fig.show()

    if plot_data == 'plotly':
        fig = default_fig()
        fig.add_trace(go.Heatmap(x=x_2d, y=y_2d, z=d_2d, coloraxis="coloraxis"))
        fig.update_coloraxes(colorscale="Blues", showscale=True)
        fig.update_layout(
            title=f"2D slice",
            xaxis_title="x",
            yaxis_title="y",
        )
        fig.show()
    return x_2d, y_2d, d_2d


def get_avg_background(x, y, data, xlim, ylim):
    x_bg, y_bg, d_bg = analysis_functions.limit_dataset(
        x=x,
        y=y,
        data=data,
        xlim=(np.min(xlim), np.max(xlim)),
        ylim=(np.min(ylim), np.max(ylim)),
    )

    return np.mean(d_bg)


def interpolate_dataset(x, y, data, xref=None, yref=None, kind="linear"):
    """
    - interpolate data to match resolution of another dataset (yref or xref)
    - returns interpolated data, no axes (because this function doesn't currently modify axes at all)
    """
    interper = interp2d(x, y, data, kind=kind)
    if xref is not None and yref is None:
        new_d = interper(xref, y)
    elif yref is not None and xref is None:
        new_d = interper(x, yref)
    elif xref and yref:
        new_d = interper(xref, yref)
    else:
        raise ValueError(
            f"xref={xref}, yref={yref}. Need either xref or yref to define interpolation parameters for new x or y "
            f"axis of dataset."
        )
    return new_d


from scipy.interpolate import interp1d


def stitch_2_datasets(dataslice1, x1, y1, dataslice2, x2, y2, stitch_dim="x"):
    # Generate new axes
    new_x = np.linspace(min(min(x1), min(x2)), max(max(x1), max(x2)), 1000)
    new_y = np.linspace(min(min(y1), min(y2)), max(max(y1), max(y2)), 1000)

    # Generate new grid for data
    new_datas = []

    # Stitching in X
    if stitch_dim == "x":

        # TODO: Ensure order is right to left
        # Ensure dataset1 is on right, dataset2 is on left
        if x1.min() < x2.min():
            x1, x2 = x2, x1
            dataslice1, dataslice2 = dataslice2, dataslice1
        elif x1.min() > x2.min():
            x1, x2 = x1, x2
            dataslice1, dataslice2 = dataslice1, dataslice2
        else:
            raise ValueError(f"Check x axes: {x1.min()} = {x2.min()}")

        for x, data in zip([x1, x2], [dataslice1, dataslice2]):
            nd = []
            for row in data:
                interper = interp1d(x, row, fill_value=np.nan, bounds_error=False)
                nd.append(interper(new_x))
            new_datas.append(np.array(nd))

        # Find overlap region
        index = int(dataslice1.shape[0] / 2)
        l = np.min(x1[dataslice1[index] > 0.01])
        r = np.max(x2[dataslice2[index] > 0.01])
        left = min(l, r)
        right = max(l, r)
        print(left, right)

        overlap_indices = (
            np.where(new_x > left)[0][0],
            np.where(new_x < right)[0][-1],
        )  # indices over which the datasets will overlap
        # overlap_indices = np.min(new_x[new_x > left]), np.max(new_x[new_x < right])
        print(overlap_indices)

        # Create weighting arrays
        w1 = np.linspace(0, 1, overlap_indices[1] - overlap_indices[0])
        w2 = np.flip(w1)

        overlap1 = w1 * new_datas[0][:, overlap_indices[0]: overlap_indices[1]]
        overlap2 = w2 * new_datas[1][:, overlap_indices[0]: overlap_indices[1]]

        overlap = overlap1 + overlap2

        new_data = np.concatenate(
            (
                new_datas[1][:, : overlap_indices[0]],
                overlap,
                new_datas[0][:, overlap_indices[1]:],
            ),
            axis=-1,
        )
        print(new_data.shape)

        return new_x, y1, new_data

    # Stitching in Y
    if stitch_dim == "y":

        # Ensure order is top to bottom
        if y1.min() < y2.min():
            y1, y2 = y2, y1
            dataslice1, dataslice2 = dataslice2, dataslice1
        elif y1.min() > y2.min():
            y1, y2 = y1, y2
            dataslice1, dataslice2 = dataslice1, dataslice2
        else:
            raise ValueError(f"Check y axes: {y1.min()} = {y2.min()}")

        for y, data in zip([y1, y2], [dataslice1, dataslice2]):
            nd = []
            for col in data.T:
                interper = interp1d(y, col, fill_value=np.nan, bounds_error=False)
                nd.append(interper(new_y))
            new_datas.append(np.array(nd))

        # Find overlap region
        index = int(dataslice1.shape[1] / 2)
        bottom = np.min(y1[dataslice1.T[index] > 0.01])
        top = np.max(y2[dataslice2.T[index] > 0.01])

        print(bottom, top)

        overlap_indices = (
            np.where(new_y > bottom)[0][0],
            np.where(new_y < top)[0][-1],
        )  # indices over which the datasets will overlap
        # overlap_indices = np.min(new_theta[new_theta > left]), np.max(new_theta[new_theta < right])
        # print(overlap_indices)

        # Create weighting arrays
        w1 = np.linspace(0, 1, overlap_indices[1] - overlap_indices[0])
        w2 = np.flip(w1)

        overlap1 = w1 * new_datas[0][:, overlap_indices[0]: overlap_indices[1]]
        overlap2 = w2 * new_datas[1][:, overlap_indices[0]: overlap_indices[1]]

        overlap = overlap1 + overlap2

        d_lower = new_datas[1][:, : overlap_indices[0]]
        d_upper = new_datas[0][:, overlap_indices[1]:]

        print(d_lower.T.shape, overlap.T.shape, d_upper.T.shape)
        new_data = np.vstack([d_lower.T, overlap.T, d_upper.T])

        return x1, new_y, new_data


def stitch_and_avg(x1, y1, data1, x2, y2, data2, no_avg=False, equal_intensity=False):
    # Create new axes, 1000 x 1000 is the desired final resolution
    new_x = np.linspace(min(min(x1), min(x2)), max(max(x1), max(x2)), 1000)
    new_y = np.linspace(min(min(y1), min(y2)), max(max(y1), max(y2)), 1000)

    # Generate new grid for data
    new_datas = []

    # Interpolate datasets onto new meshgrid (rqeuires defining interper function)
    for x, y, data in zip([x1, x2], [y1, y2], [data1, data2]):
        interper = RegularGridInterpolator(
            (y, x), data, fill_value=np.nan, bounds_error=False
        )
        xx, yy = np.meshgrid(new_x, new_y, indexing="ij")

        new_datas.append(interper((yy, xx)).T)

    if equal_intensity:
        # Find overlap region
        overlap_indices = (
            np.where(new_y > max(min(y1), min(y2)))[0][0],
            np.where(new_y < min(max(y1), max(y2)))[0][-1],
        )

        # Scale second dataset so that its average value in the overlap region is the same as the first dataset
        avg1 = np.nanmean(new_datas[0][:, overlap_indices[0]: overlap_indices[1]])
        avg2 = np.nanmean(new_datas[1][:, overlap_indices[0]: overlap_indices[1]])
        new_datas[1] *= avg1 / avg2

        # Create weighting arrays
        w1 = np.linspace(0, 1, overlap_indices[1] - overlap_indices[0])
        w2 = np.flip(w1)

        # Apply weights to the overlap region
        new_datas[0][:, overlap_indices[0]: overlap_indices[1]] *= w1
        new_datas[1][:, overlap_indices[0]: overlap_indices[1]] *= w2

    # Combine preserving Data1
    if no_avg:
        new_data = new_datas[1]
        new_data[np.where(~np.isnan(new_datas[0]))] = new_datas[0][~np.isnan(new_datas[0])]
    else:
        # Average dataslices together where they overlap (otherwise keep the original data)
        new_data = np.nanmean(new_datas, axis=0)

    return new_x, new_y, new_data


# def stitch_and_avg(x1, y1, data1, x2, y2, data2, no_avg=False):
#     """
#     Takes 2 datasets (plus axes) with partial overlap and returns a single dataset (plus axes) where the overlap has
#     been averaged together. The regions with no overlap maintain the original data.
#     Args:
#         x1: xaxis array of dataset 1
#         y1: yaxis array of dataset 1
#         data1: dataset 1 (2D slice)
#         x2: xaxis array of dataset 2
#         y2: yaxis array of dataset 2
#         data2: dataset 2 (2D slice)
#
#     Returns: new xaxis array, new yaxis array, new dataset (2D np array)
#
#     """
#     # Create new axes, 1000 x 1000 is the desired final resolution
#     new_x = np.linspace(min(min(x1), min(x2)), max(max(x1), max(x2)), 1000)
#     new_y = np.linspace(min(min(y1), min(y2)), max(max(y1), max(y2)), 1000)
#
#     # Generate new grid for data
#     new_datas = []
#
#     # Interpolate datasets onto new meshgrid (rqeuires defining interper function)
#     for x, y, data in zip([x1, x2], [y1, y2], [data1, data2]):
#         interper = RegularGridInterpolator(
#             (y, x), data, fill_value=np.nan, bounds_error=False
#         )
#         xx, yy = np.meshgrid(new_x, new_y, indexing="ij")
#
#         new_datas.append(interper((yy, xx)).T)
#
#     # Combine preserving Data1
#     if no_avg:
#         new_data = new_datas[1]
#         new_data[np.where(~np.isnan(new_datas[0]))] = new_datas[0][~np.isnan(new_datas[0])]
#
#     # Average dataslices together where they overlap (otherwise keep the original data)
#     else:
#         new_data = np.nanmean(new_datas, axis=0)
#
#     return new_x, new_y, new_data


# def stitch_2_datasets_in_y(data1, x1, y1, data2, x2, y2, axis="y", inc_weight_array=False):
#     if axis == "y":
#         if y1.min() < y2.min():
#             # if np.min([y1, y2]) in y1:
#             y_lower, y_upper = y1, y2
#             d_lower, d_upper = data1, data2
#         elif y1.min() > y2.min():
#             # elif np.min([y1, y2]) in y2:
#             y_lower, y_upper = y2, y1
#             d_lower, d_upper = data2, data1
#         else:
#             raise ValueError(f"Check y axes: {y1.min()} = {y2.min()}")
#
#         # get thresholds of overlap in datasets
#         # give index --> y_lower[threshold_1] = np.min(y_upper)
#         threshold_1 = np.where(np.isclose(y_lower, np.min(y_upper)))[0][0]
#         threshold_2 = np.where(np.isclose(y_upper, np.max(y_lower), atol=1e-6))[0][0]
#         # threshold_1 give lower bound of overlap, threshold_2 gives upper bound
#         d_avg_1 = d_lower[threshold_1:,]
#         d_avg_2 = d_upper[:threshold_2,]
#
#         dim_diff = d_avg_1.shape[0] - d_avg_2.shape[0]
#
#         # if d_avg_1.shape[0] != d_avg_2.shape[0]:
#         if dim_diff > 0:
#             d_avg_1 = d_avg_1[: -1 * dim_diff, :]
#
#         if dim_diff < 0:
#             d_avg_2 = d_avg_2[dim_diff:, :]
#
#         print(f'd_avg_1 = {d_avg_1.shape}, d_avg_2 = {d_avg_2.shape}')
#
#         if inc_weight_array:
#             # create weighting for arrays
#             w1 = np.linspace(0, 1, d_avg_1.shape[0])[:, None]
#             w2 = np.flip(w1)
#             d_avg_1 = d_avg_1 * w1
#             d_avg_2 = d_avg_2 * w2
#             d_avg = d_avg_1 + d_avg_2
#
#         else:
#             d_avg = np.mean([d_avg_1, d_avg_2], axis=0)
#
#         d_lower = d_lower[:threshold_1,]
#         d_upper = d_upper[threshold_2:,]
#
#         steps = d_lower.shape[0] + d_avg.shape[0] + d_upper.shape[0]
#
#         y = np.linspace(np.min(y_lower), np.max(y_upper), steps)
#
#         x = x1
#
#         data = np.vstack([d_lower, d_avg, d_upper])
#
#         return x, y, data


def x_y_to_coords(x, y):
    return np.stack(np.meshgrid(x, y, indexing="xy"))


def rotate_2d_array(coords, angle, center):
    # coords has shape (2, X, Y)
    coords = copy.deepcopy(coords)

    # Convert angle to radians
    theta = np.radians(angle)

    # Flatten coords after storing what the shape was
    shape = coords.shape
    coords = coords.reshape((2, -1))

    # Translate array to origin
    coords -= np.array(center)[:, None]

    # Define rotation matrix
    rotation_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )

    # Apply rotation to array
    rotated_coords = np.dot(rotation_matrix, coords)

    # Translate array back to original position
    rotated_coords += np.array(center)[:, None]

    rotated_coords = rotated_coords.reshape(shape)

    return rotated_coords


def interpolate(rotated_coords, data):
    """Get back to having a regular x and y array for rotated data"""
    rx = rotated_coords[0]
    ry = rotated_coords[1]
    rotated_coords = np.array([rx.flatten(), ry.flatten()]).T
    print(f"rotated_coords: {rotated_coords.shape}, data: {data.flatten().shape}")
    # interper = interp2d(rx.flatten(), ry.flatten(), data.flatten(), fill_value=np.nan)
    interper = LinearNDInterpolator(rotated_coords, data.flatten(), fill_value=np.nan)
    nx = np.linspace(np.nanmin(rx), np.nanmax(rx), 1000)
    ny = np.linspace(np.nanmin(ry), np.nanmax(ry), 1000)
    ndata = interper(*np.meshgrid(nx, ny))
    ndata = ndata.reshape((ny.shape[0], nx.shape[0]))
    return nx, ny, ndata
