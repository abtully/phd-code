"""
Plotting functions for angle-resolved photoemission spectroscopy (ARPES) data
@author: Alexandra Tully
@date: November 2020

Notes:
Colorscale options: https://plotly.com/python/builtin-colorscales/
"""

import plotly.graph_objects as go
from arpes_functions.arpes_dataclasses import Data2D
from typing import List, Optional
import numpy as np
import copy
import matplotlib.pyplot as plt
from .analysis_functions import get_2Dslice

DEFAULT_RENDERER = 'browser'  # this is a constant of this file

"""Formatting Functions"""


def format_FS_plot(fig, ax, slice_val=None, xlabel='kx (A-1)', ylabel='ky (A-1)', title=None, EF=None,
                   xticks=np.arange(-0.6, 0.52, 0.3), yticks=np.arange(-1.2, 0.2, 0.4), set_aspect_ratio=True,
                   aspect_ratio=1.0, fontsize_ticks=14, fontsize_labels=16, fontsize_title=18):
    if title and EF is None:
        raise ValueError(f'Title = {title} and EF = {EF}. Must specify one.')
    if xticks is not None:
        plt.xticks(xticks, fontsize=fontsize_ticks)
    ax.set_xlabel(xlabel=xlabel, fontsize=fontsize_labels)
    if yticks is not None:
        plt.yticks(yticks, fontsize=fontsize_ticks)
    ax.set_ylabel(ylabel=ylabel, fontsize=fontsize_labels)
    if title is not None:
        ax.set_title(f'{title}', fontsize=fontsize_title)
    elif title is None:
        if EF is not None:
            EF = float(EF)
            ax.set_title(f'{np.round((-EF + slice_val), 2)} eV', fontsize=fontsize_title)

    if set_aspect_ratio:
        ratio = aspect_ratio  # set aspect ratio
        x_left, x_right = ax.get_xlim()
        y_low, y_high = ax.get_ylim()
        ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)


def format_HS_plot(fig, ax, slice_constant, slice_val, ylabel='E - EF (eV)', title=None, EF=None, xticks=None,
                   yticks=np.arange(-0.6, 0.2, 0.2), set_aspect_ratio=True, aspect_ratio=1.0, fontsize_ticks=14,
                   fontsize_labels=16, fontsize_title=18):
    if title and slice_val is None:
        raise ValueError(f'Title = {title} and slice_val = {slice_val}. Must specify one.')
    if slice_constant == 'kx':
        xlabel = 'ky (A-1)'
        if title is None:
            title = f'kx={slice_val}'
    if slice_constant == 'ky':
        xlabel = 'kx (A-1)'
        if title is None:
            title = f'ky={slice_val}'
    if xticks is None:
        plt.xticks(fontsize=fontsize_ticks)
    else:
        plt.xticks(xticks, fontsize=fontsize_ticks)
    ax.set_xlabel(xlabel=xlabel, fontsize=fontsize_labels)
    if yticks is None:
        plt.yticks(fontsize=fontsize_ticks)
    else:
        plt.yticks(yticks, fontsize=fontsize_ticks)
    ax.set_ylabel(ylabel=ylabel, fontsize=fontsize_labels)
    if EF is not None:
        EF = float(EF)
        ax.set_title(f'{np.round((-EF + slice_val), 2)} eV', fontsize=fontsize_title)
    if title is not None:
        ax.set_title(f'{title}', fontsize=fontsize_title)

    if set_aspect_ratio:
        ratio = aspect_ratio  # set aspect ratio
        x_left, x_right = ax.get_xlim()
        y_low, y_high = ax.get_ylim()
        ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)


"""Plotting Functions 2D"""


def plot2D(x: np.ndarray, y: np.ndarray, data: np.ndarray, show=True, opacity=None, xlabel=None, ylabel=None,
           title=None, xrange: tuple = None, yrange: tuple = None, dark_mode=False, colorscale='Plasma',
           E_f: float = None, renderer=DEFAULT_RENDERER):
    """
    Plots 2D plotly figure

    Returns: plotly figure

    """
    fig = go.Figure()
    if E_f is not None and E_f != '':
        y = y - float(E_f)
        ylabel = 'E - EF (eV)'
    hm = heatmap(x, y, data, opacity=opacity)
    fig.add_trace(hm)
    fig.update_layout(dict(xaxis_title=xlabel, yaxis_title=ylabel, title=title), coloraxis=dict(colorscale=colorscale))
    if dark_mode:
        fig.update_layout(template='plotly_dark')  # dark mode
    if xrange:
        fig.update_layout(xaxis=dict(range=[xrange[0], xrange[1]]))
    if yrange:
        fig.update_layout(yaxis=dict(range=[yrange[0], yrange[1]]))
    if show:
        fig.show(renderer=renderer)
    return fig


def plot2D_jupyter(x: np.ndarray, y: np.ndarray, data: np.ndarray, width=500, height=500,
                   title=False, ylabel=False, xlabel=False):
    """
    Calls 2D plotting function in jupyter-friendly style

    Returns: Plotly figure
    """
    if not title:
        title = 'kx vs Energy'
    if not ylabel:
        ylabel = 'Energy (eV)'
    if not xlabel:
        xlabel = 'kx (A-1)'
    return plot2D(x=x, y=y, data=data, renderer='jupyterlab', show=False, xlabel=xlabel, ylabel=ylabel,
                  title=title).update_layout(autosize=False, width=width, height=height)


def plot_2D_mpl(x: np.ndarray, y: np.ndarray, data: np.ndarray, title=None, xlabel=None, ylabel=None, EF=None,
                cmap=None, vmin=None, vmax=None, norm=None, colorbar=False, swap_axes=False):
    """
    Plots 2D matplotlib figure

    Returns: mpl fig, ax

    """
    # xx, yy = np.meshgrid(x, y)
    fig, ax = plt.subplots(1)
    if swap_axes:
        x, y = y, x
        # xlabel, ylabel = ylabel, xlabel
        data = data.T
    if EF:
        y = y - EF
        if ylabel is None:
            ylabel = 'E - EF (eV)'
    pc = ax.pcolormesh(x, y, data, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax, norm=norm)
    if colorbar is True:
        fig.colorbar(pc, ax=ax, extend='max')
    if title is None:
        ax.set_title(label='kx vs Energy')
    else:
        ax.set_title(label=title)
    if xlabel is None:
        ax.set_xlabel(xlabel='kx (A-1)')
    else:
        ax.set_xlabel(xlabel=xlabel)
    if ylabel is None:
        ax.set_ylabel(ylabel='Energy (eV)')
    else:
        ax.set_ylabel(ylabel=ylabel)
    return fig, ax


def show_kpts(fig, material='C60', ticks=True, tickangle=0, vlines=False):
    """
    Plots plotly figure with x-xaxis replaced by k high symmetry points for C60 or Au(111) (expects k-corrected data)
    Args:
        fig: plotly figure
        material: C60, Au(111)
        ticks: can be turned off
        tickangle: 0, 30, 45, 90, etc.
        vlines: adds dashed black vertical lines at HS points

    Returns: Plotly figure

    """
    gamma = 0  # distances from gamma to high symmetry point in inverse angstroms
    m_c60 = 0.36
    m_au = 1.26
    k_c60 = 0.416
    k_au = 1.45
    if material == 'C60':
        vals = [gamma, m_c60, k_c60, -m_c60, -k_c60]
    elif material == 'Au(111)':
        vals = [gamma, m_au, k_au, -m_au, -k_au]
    else:
        raise ValueError(f'Material {material}, must be C60 or Au(111)')
    text = ['gamma', 'M', 'K', 'M', 'K']
    fig.update_layout(xaxis=dict(tickmode='array', tickvals=vals, ticktext=text))
    if ticks == True:
        fig.update_xaxes(ticks='outside', tickangle=tickangle)
        fig.update_yaxes(ticks='outside')
    if vlines == True:
        for v in vals:
            fig.add_vline(x=v, line_dash='dash', line_color='black', line_width=2)
    fig.update_layout(title=f'{material} Band Structure')
    return fig


def show_kpts_mpl(fig, ax, material='C60', tickangle=0, vlines=False, color='black', title=None):
    """
    Plots mpl figure with x-xaxis replaced by k high symmetry points for C60 or Au(111) (expects k-corrected data).
    Note that choosing C60/Au(111) will plot C60 HS points in black and Au HS points in red (if color is left default).
    Args:
        fig: mpl figure
        ax: mpl ax
        material: C60, Au(111), or C60/Au(111)
        tickangle: 0, 30, 45, 90, etc.
        vlines: adds dashed vertical lines at HS points
        color: specifies color of vertical lines
        title: can specify title of plot (otherwise reverts to default)

    Returns: mpl figure

    """
    gamma = 0  # distances from gamma to high symmetry point in inverse angstroms
    m_c60 = 0.36
    m_au = 1.26
    k_c60 = 0.416
    k_au = 1.45
    if material not in ['C60', 'Au(111)', 'C60/Au(111)']:
        raise ValueError(f'Material {material}, must be C60, Au(111), or C60/Au(111)')
    if material == 'C60/Au(111)':
        vals = [gamma, m_c60, k_c60, -m_c60, -k_c60, m_au, k_au, -m_au, -k_au]
        text = ['gamma', 'M', 'K', 'M', 'K', 'M', 'K', 'M', 'K']
    else:
        if material == 'C60':
            vals = [gamma, m_c60, k_c60, -m_c60, -k_c60]
        elif material == 'Au(111)':
            vals = [gamma, m_au, k_au, -m_au, -k_au]
        text = ['gamma', 'M', 'K', 'M', 'K']
    ax.set_xticks(vals)
    ax.set_xticklabels(text, rotation=tickangle, color=color)
    if material == 'C60/Au(111)':
        [x2.set_color('red') for x2 in ax.get_xticklabels()[-4:]]
    if vlines is True:
        [ax.axvline(x=val, ymin=0, ymax=1, linestyle='dashed', color=color, linewidth=1) for val in vals]
    if title is None:
        ax.set_title(label=f'{material} Band Structure')
    elif title:
        ax.set_title(label=title)
    return fig


def multi_plot2D(heatmaps: List[go.Heatmap], xlabel=None, ylabel=None,
                 title=None, xrange: tuple = None, yrange: tuple = None,
                 dark_mode=False, colorscale=None):  # expecting heatmaps to be a list of go.Heatmaps (class)
    fig = go.Figure()
    for hm in heatmaps:
        fig.add_trace(hm)
    fig.update_layout(xaxis_title=xlabel, yaxis_title=ylabel, title=title, coloraxis=dict(colorscale=colorscale))
    if dark_mode:
        fig.update_layout(template='plotly_dark')  # dark mode
    if xrange:
        fig.update_layout(xaxis=dict(range=[xrange[0], xrange[1]]))
    if yrange:
        fig.update_layout(yaxis=dict(range=[yrange[0], yrange[1]]))
    fig.show(renderer=DEFAULT_RENDERER)
    return fig


def filepath_plot2D(filepath: str, month, year=2020, laser='lamp', cryo_temp='RT', scan_type='HS', k_cut=None,
                    scan_number=1, filename=None, colorscale='Plasma') -> go.Figure:
    data = Data2D.single_load(month, year=year, light_source=laser, cryo_temp=cryo_temp, scan_type=scan_type,
                              k_cut=k_cut, scan_number=scan_number, filename=filename, filepath=filepath)
    return plot2D(data.xaxis, data.yaxis, data.data, show=False, colorscale=colorscale)


"""Generate Heatmaps"""


def heatmap(x, y, data, opacity=None) -> go.Heatmap:  # this function returns a heatmap
    return go.Heatmap(x=x, y=y, z=data, opacity=opacity, coloraxis="coloraxis")


def multi_heatmaps(datas: List[Data2D], E_f: float, opacity: Optional[float] = None) -> List[go.Heatmap]:
    heatmaps = []
    for data in datas:
        hm = heatmap(data.yaxis, data.xaxis - E_f, data.data, opacity=opacity)
        heatmaps.append(hm)
    return heatmaps


"""Plotting Functions 3D"""


# plot 2D slice of 3D data
def plot3D(x: np.ndarray, y: np.ndarray, z: np.ndarray, data: np.ndarray, slice_dim: str, slice_val: float,
           int_range: float = 0, show=True, opacity=None, xlabel=None, ylabel=None, title=None, xrange: tuple = None,
           yrange: tuple = None, dark_mode=False, colorscale='Plasma', renderer=DEFAULT_RENDERER):
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
    return plot2D(axes_2d[0], axes_2d[1], data2d, show=show, opacity=opacity, xlabel=xlabel, ylabel=ylabel, title=title,
                  xrange=xrange, yrange=yrange, dark_mode=dark_mode, colorscale=colorscale, renderer=renderer)


def plot3D_jupyter(x, y, z, data, slice_dim, slice_val, int_range, width=500, height=500, figtitle=False):
    """
    Calls 3D plotly plotting function in jupyter-friendly style

    Returns: Plotly figure
    """
    if slice_dim == 'y':
        yaxis = 'ky (A-1)'
    elif slice_dim == 'z':
        yaxis = 'Energy (eV)'
    if figtitle is False:
        figtitle = f'kx vs {yaxis}'
    return plot3D(x=x, y=y, z=z, data=data, slice_dim=slice_dim, slice_val=slice_val, int_range=int_range,
                  renderer='jupyterlab', show=False, xlabel='kx', ylabel=yaxis,
                  title=figtitle).update_layout(autosize=False, width=width, height=height)


def plot_3D_mpl(x, y, z, data, slice_dim, slice_val, int_range, title=None, swap_axes=False, cmap=None):
    """
    Plots 2D slice of 3D data in matplotlib

    Returns: mpl fig, ax

    """
    xaxis, yaxis, dataslice = get_2Dslice(x, y, z, data, slice_dim, slice_val, int_range)
    if slice_dim == 'y':
        new_title = f'Constant Energy {slice_val}eV (int range {int_range})'
        ylabel = 'ky (A-1)'
        xlabel = 'kx (A-1)'
    elif slice_dim == 'z':
        new_title = f'Constant phi = {slice_val} (int range {int_range})'
        ylabel = 'Energy (eV)'
        xlabel = 'kx (A-1)'
    elif slice_dim == 'x':
        new_title = f'Constant theta = {slice_val} (int range {int_range})'
        xlabel = 'ky (A-1)'
        ylabel = 'Energy (eV)'
    else:
        raise ValueError(f'Slice dimension not x, y, or z: f{slice_dim}')
    if title:
        title = title
    else:
        title = new_title
    if swap_axes:
        xaxis, yaxis = yaxis, xaxis
        # xlabel, ylabel = ylabel, xlabel
        dataslice = dataslice.T
    return plot_2D_mpl(xaxis, yaxis, dataslice, title=title, ylabel=ylabel, xlabel=xlabel, cmap=cmap)


def filepath_plot3D(filepath: str, month, year=2020, laser='lamp', cryo_temp='RT', scan_type='FS',
                    scan_number=1, slice_dim='z', slice_val=15, int_range=0.02, filename=None, colorscale='Plasma') \
        -> go.Figure:
    # print(type(int_range), type(slice_val), type(slice_dim))
    data = Data3D.single_load(month, year=year, light_source=laser, cryo_temp=cryo_temp, scan_type=scan_type,
                              scan_number=scan_number, filename=filename, filepath=filepath)
    return plot3D(x=data.yaxis, y=data.xaxis, z=data.zaxis, data=np.moveaxis(data.data, 2, 1), slice_dim=slice_dim,
                  slice_val=slice_val, int_range=int_range, show=False, colorscale=colorscale)


def mutliplot3D(x: np.ndarray, y: np.ndarray, z: np.ndarray, data: np.ndarray, slice_dim: str, slice_val: float,
                int_range: float = 0, show=True, opacity=None, xlabel=None, ylabel=None, title=None,
                xrange: tuple = None,
                yrange: tuple = None, dark_mode=False, colorscale='Plasma'):
    raise NotImplementedError
    # hm = heatmap3D(x=x, y=y, z=z, data=data, slice_dim=slice_dim, slice_val=slice_val, int_range=int_range, show=show,
    #                opacity=opacity)


def transpose_figure(fig: go.Figure) -> go.Figure:
    fig = copy.copy(fig)
    fig_data = fig.data[0]
    x, y, data = fig_data['x'], fig_data['y'], fig_data['z']
    fig_data['x'], fig_data['y'], fig_data['z'] = y, x, data.T
    fig.data = [fig_data]
    return fig


if __name__ == '__main__':
    from arpes_functions.arpes_dataclasses import Data3D, Data2D
    # from misc_functions import multi_load
    #
    # d = Data2D.single_load('April', '2021', light_source='XUV', filename='OMBE_XUV_2D0016_.ibw')
    # plot2D(d.xaxis, d.yaxis, d.data)
    # d = Data3D.single_load('October', cryo_temp='RT', scan_subtype='Calibrate')
    # plot3D(x=d.yaxis, y=d.xaxis, z=d.zaxis, data=np.moveaxis(d.data, 2, 1), slice_dim='z', slice_val=15,
    # int_range=0.02)
    # d.berend_data.show(mode='CE', val=15, Eint=0.02)

    # d = Data2D.single_load('December', year='2020', filename='UPS_20K0001_001.ibw')
    # plot2D(d.xaxis, d.yaxis-16.8, d.data)
    # d = Data2D.single_load(month='January', year='2021', light_source='XUV', scan_number=4)
    # plot2D(d.xaxis, d.yaxis, d.data)
    #
    # raster = Data2D.single_load('October', year='2020', cryo_temp='RT', scan_number=15, filename='Raster0015_015.ibw')
    # E_f = 16.8
    # plot2D(raster.yaxis, raster.xaxis - E_f, raster.data, opacity=0.7, xrange=(-20, 17))
    #
    # rasters = multi_load(i for i in range(10, 14))
    # heatmaps = multi_heatmaps(rasters, E_f=16.8, opacity=0.5)
    # multi_plot2D(heatmaps, title='Test')
