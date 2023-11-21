"""
K-correction functions; modifying code written by Berend Zwartsenberg
@author: Alexandra Tully
@date: March 2021
"""

import numpy as np
from zwartsenberg.plotters import kplot2D
from zwartsenberg.plotters import plot2D as bpl2
from zwartsenberg.zwartsenberg_kcorrection_new_new import kfromangles
from arpes_functions.HDF5_loader import data_from_hdf
from typing import Optional
import plotly.io as pio

pio.renderers.default = 'browser'


def fix_EkatEF(self, EF):  # 16.8 is the EF for Au(111) with He lamp
    if self.berend_dataclass.p['EkatEF'] is None:
        self.berend_dataclass.p['EkatEF'] = EF
        self.berend_dataclass.p['type'] = 'corrected'
        # self.setXScale(self.xaxis[0] - EF, self.xaxis[1] - self.xaxis[0])


def kcorrect2D(self, kmode='edep'):
    """applies Berend's k-correction to data, so that Ali's plotting functions can be used; see kplot2D funciton in
    Berend's plotters.py file for original function. Takes a class.



    Args:
        kmode (string): 'edep' (dependent on energy) or 'simple' (taken at 0 energy).
                    Always use 'edep' in normal operation
    Returns:
        kmesh[0, :]: xaxis for kcorrected data
        Emesh[:, 0]: yaxis for kcorrected data (Ek not binding energy)
        self.data: dataset

    """
    Aa2D = self.berend_dataclass

    if Aa2D.p['type'] == 'rawdata':
        raise RuntimeError('k plots only work on corrected data')
    if Aa2D.p['type'] is None:
        raise RuntimeError('k plots only work on corrected data')
    if (kmode == 'simple'):
        kx, ky = kfromangles(Aa2D.yaxis, 0., Aa2D.p['EkatEF'], theta_m=Aa2D.p['theta_m'], phi_m=Aa2D.p['phi_m'],
                             theta0=Aa2D.p['theta0'], phi0=Aa2D.p['phi0'], alpha0=Aa2D.p['alpha0'])  # phi is set to
        # zero, because manipulator offset is already captured by phi_m
        dkx = np.zeros_like(kx)
        dky = np.zeros_like(ky)
        dkx[1:] = kx[1:] - kx[:-1]
        dky[1:] = ky[1:] - ky[:-1]
        dk = np.sqrt(dkx * dkx + dky * dky)
        kax = np.cumsum(dk)
        argzero = np.argmin(np.sqrt(kx ** 2 + ky ** 2))
        kax -= kax[argzero]
        kmesh, Emesh = np.meshgrid(kax, Aa2D.xaxis)
    elif (kmode == 'edep'):
        thmesh, Emesh = np.meshgrid(Aa2D.yaxis, Aa2D.xaxis + Aa2D.p['EkatEF'])
        kx, ky = kfromangles(thmesh, 0., Emesh, theta_m=Aa2D.p['theta_m'], phi_m=Aa2D.p['phi_m'],
                             theta0=Aa2D.p['theta0'], phi0=Aa2D.p['phi0'], alpha0=Aa2D.p['alpha0'])
        dkx = np.zeros_like(kx).astype('float64')  # this is to prevent a bug in np that raises an error taking the
        # sqrt of 0
        dky = np.zeros_like(ky).astype('float64')
        dkx[:, 1:] = kx[:, 1:] - kx[:, :-1]
        dky[:, 1:] = ky[:, 1:] - ky[:, :-1]

        dk = np.sqrt(dkx * dkx + dky * dky)
        kmesh = np.cumsum(dk, axis=1)

        argzero = np.argmin(np.sqrt(kx ** 2 + ky ** 2), axis=1)

        for i in range(kmesh.shape[0]):
            kmesh[i] -= kmesh[i, argzero[i]]

        Emesh -= Aa2D.p['EkatEF']
        return kmesh[0, :], Emesh[:, 0], self.data


def kcorrect2D_general(data, xaxis, yaxis, EF=16.8, theta_m: float = 0, phi_m: float = 0, theta0: float = 0,
                       phi0: float = 0, alpha0: float = 0):
    """More general version of Berend's k-correction function for 2D data.
    Args:
        kmode (string): 'edep' (dependent on energy) or 'simple' (taken at 0 energy).
                    Always use 'edep' in normal operation
    Returns:
        kmesh[0, :]: xaxis for kcorrected data
        Emesh[:, 0]: yaxis for kcorrected data (Ek not binding energy)
        self.data: dataset

    """
    thmesh, Emesh = np.meshgrid(xaxis, yaxis + EF)
    kx, ky = kfromangles(thmesh, 0., Emesh, theta_m=theta_m, phi_m=phi_m, theta0=theta0, phi0=phi0, alpha0=alpha0)
                        # phi is set to zero, because manipulator offset is already captured by phi_m
    dkx = np.zeros_like(kx).astype('float64')  # this is to prevent a bug in np that raises an error taking the
    # sqrt of 0
    dky = np.zeros_like(ky).astype('float64')
    dkx[:, 1:] = kx[:, 1:] - kx[:, :-1]
    dky[:, 1:] = ky[:, 1:] - ky[:, :-1]

    dk = np.sqrt(dkx * dkx + dky * dky)
    kmesh = np.cumsum(dk, axis=1)

    argzero = np.argmin(np.sqrt(kx ** 2 + ky ** 2), axis=1)

    for i in range(kmesh.shape[0]):
        kmesh[i] -= kmesh[i, argzero[i]]

    Emesh -= EF
    return kmesh[0, :], Emesh[:, 0], data


def get2Dslice(x: np.ndarray, y: np.ndarray, z: np.ndarray, data: np.ndarray, slice_dim: str, slice_val: float,
               int_range: Optional[float] = None):
    if int_range is None:
        int_range = 0
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


def kcorrect3D(data_class, num=None, val=None, Eint_n=1, Eint=None, fixEF=False, EF: float = None,
               slice_dim: str = 'z', scan_type: str = 'deflector_scan', theta_m: float = 0, phi_m: float = 0,
               theta0: float = 0, phi0: float = 0, alpha0: float = 0):
    """k-corrected constant energy plot function, based on Berend's plotCEk3D from plotters.py


    Args:
        num (int): the xaxis index at which energy the slice is taken
        val (float): energy at which the slice is taken, overrides num
        Eint_n (int): number of slices to average over
        Eint (float): energy range to average over, overrides Eint_n
        fixEF (bool): optional parameter, if true, calls fix_EkatEF function
        EF (float): optional parameter, sets EF
        slice_dim (str): define axis to slice along, typically z for deflector voltage scans (energy axis)
        scan_type: deflector_scan
    Returns:
        kmeshx[0, :]: kx axis
        kmeshy[:, 0]: ky axis
        np.mean(Aa3D.data[num - Eint_n:num + Eint_n], axis=0).T: dataset for plotting

    """
    # print(slice_dim)
    Aa3D = data_class
    # Aa3D = data_class.berend_data

    # if fixEF is True:
    #     fix_EkatEF(self, EF)

    # if not (Aa3D.p['type'] == 'corrected'):
    if EF is None:
        raise RuntimeError(f'k plots only work on corrected data, specify EF. (EF = {EF})')

    choose_slice = {
        'x': data_class.xaxis, 'y': data_class.yaxis, 'z': data_class.zaxis
    }
    ax = choose_slice[slice_dim]
    ax = ax - EF
    choose_axval = {
        'x': 2, 'y': 1, 'z': 0
        # 'x': 0, 'y': 1, 'z': 2
    }
    axval = choose_axval[slice_dim]
    start = ax[0]
    end = ax[-1]
    delta = np.mean(np.diff(ax))

    if val is not None:
        num = int(round((val - start) / delta))
    elif num is None:
        raise RuntimeError('Provide either num or val')

    # print(f'{num}, {axval}, {data_class.data.shape}, {data_class.data.shape[axval]}, {val}, {start}, {end}, {delta}')
    if (num < 0) or (num >= data_class.data.shape[axval]):
        raise ValueError('Val or num outside of plot range')

    if Eint is not None:
        Eint_n = int(Eint / delta)

    kmeshx, kmeshy = ali_get_kmeshCE(data_class, ax[num], extend=True, EF=EF, slice_dim=slice_dim,
                                     scan_type=scan_type, theta_m=theta_m, phi_m=phi_m, theta0=theta0, phi0=phi0,
                                     alpha0=alpha0)  # FIXME: This only works for cuts in z (energy), make it general!!
    # print(ax.shape)
    # _, _, dd = get2Dslice(x=data_class.xaxis, y=data_class.yaxis, z=data_class.zaxis, data=data_class.data, slice_dim='z', slice_val=val, int_range=Eint)

    return kmeshx[0, :], kmeshy[:, 0], np.mean(Aa3D.data[num - Eint_n:num + Eint_n], axis=0).T
    # return kmeshx[0, :], kmeshy[:, 0], dd


def kcorrect_phimotor(fp=None, fn=None, data=None, ss=None, cs=None, p=None, slice_dim='y', EF=16.8, num=None,
                      val: float = None, Eint_n=1, Eint: float = None, theta_m: float = 0, phi_m: float = 0,
                      theta0: float = 0, phi0: float = 0, alpha0: float = 0):
    """k-corrected constant energy plot function for data taken with the phi motor (hdf5 files), either from raw file
    or from numpy arrays (data and axes)


    Args:
        fp: filepath for HDF5 file
        fn: filename, e.g. 'FS_averaged.h5'
        data: instead of raw file, can input numpy dataset and axes
        ss: slice_scale axis
        cs: channel_scale axis (y = energy axis)
        p: p axis
        slice_dim: dimension of slice through data cube (typically y = energy for CE plot)
        EF: Fermi Energy (typically ~16.8 for C60 on gold)
        num (int): the xaxis index at which energy the slice is taken
        val (float): energy at which the slice is taken, overrides num, (binding energy!!!)
        Eint_n (int): number of slices to average over
        Eint (float): energy range to average over, overrides Eint_n
        theta_m: offset angle for manipulator (see Berend's kfromangles function)
        phi_m: offset angle for manipulator (see Berend's kfromangles function)
        theta0: offset angle for sample (see Berend's kfromangles function)
        phi0: offset angle for sample (see Berend's kfromangles function)
        alpha0: offset angle for sample (see Berend's kfromangles function)
    Returns:
        kmeshx[0, :]: kx axis
        kmeshy[:, 0]: ky axis
        np.mean(Aa3D.data[num - Eint_n:num + Eint_n], axis=0).T: dataset for plotting

    """
    if fp is None and fn is None:
        if data is None:
            raise ValueError('If no filepath is provided, dataset (and ss, cs, and p axes) is required')
    if data is None:
        data, ss, cs, p = data_from_hdf(fp, fn)
    choose_slice = {
        'x': ss, 'y': cs, 'z': p
    }
    ax = choose_slice[slice_dim]
    if slice_dim == 'y':
        ax = ax - EF
    start = ax[0]
    end = ax[-1]
    delta = np.mean(np.diff(ax))

    choose_axval = {
        'x': 0, 'y': 1, 'z': 2
        # 'x': 2, 'y': 1, 'z': 0
    }
    axval = choose_axval[slice_dim]

    # if val is None:
    #     raise ValueError(f'Provide slice value ({val})')
    # elif val is not None:
    #     val_range = [start > val, end < val]
    #     if any(val_range):
    #         raise ValueError(f'{val} is not in range {start} to {end}')
    if val is not None:
        num = int(round((val - start) / delta))
    elif num is None:
        raise RuntimeError('Provide either num or val')

    print(f'{num} is not None')
    if (num < 0) or (num >= data.shape[axval]):
        raise ValueError('Val or num outside of plot range')

    if Eint is not None:
        Eint_n = int(Eint / delta)

    def get_kmeshCE(atE, EF, slice_dim, extend=True):
        x, y, z = ss, cs, p
        xdelta = np.mean(np.diff(ss))
        ydelta = np.mean(np.diff(cs))
        zdelta = np.mean(np.diff(p))
        if slice_dim == 'x':
            if extend:
                pyaxis = np.linspace(y[0] - 0.5 * ydelta, y[-1] + 0.5 * ydelta, (y.shape[0] + 1))
                pzaxis = np.linspace(z[0] - 0.5 * zdelta, z[-1] + 0.5 * zdelta, (z.shape[0] + 1))
            else:
                pyaxis = y
                pzaxis = z
            ameshth, ameshphi = np.meshgrid(pyaxis, pzaxis)
        elif slice_dim == 'y':
            if extend:
                pxaxis = np.linspace(x[0] - 0.5 * xdelta, x[-1] + 0.5 * xdelta, (x.shape[0] + 1))
                pzaxis = np.linspace(z[0] - 0.5 * zdelta, z[-1] + 0.5 * zdelta, (z.shape[0] + 1))
            else:
                pxaxis = x
                pzaxis = z
            ameshth, ameshphi = np.meshgrid(pxaxis, pzaxis)
        elif slice_dim == 'z':
            if extend:
                pxaxis = np.linspace(x[0] - 0.5 * xdelta, x[-1] + 0.5 * xdelta, (x.shape[0] + 1))
                pyaxis = np.linspace(y[0] - 0.5 * ydelta, y[-1] + 0.5 * ydelta, (y.shape[0] + 1))
            else:
                pxaxis = x
                pyaxis = y
            ameshth, ameshphi = np.meshgrid(pxaxis, pyaxis)
        else:
            raise ValueError(f'{slice_dim} is not valid')
        kmeshx, kmeshy = kfromangles(ameshth, ameshphi, (EF + atE), theta_m=theta_m, phi_m=phi_m, theta0=theta0,
                                     phi0=phi0, alpha0=alpha0)
        return kmeshx, kmeshy

    _, _, dd = get2Dslice(x=ss, y=cs, z=p, data=data, slice_dim=slice_dim, slice_val=val + EF, int_range=Eint)

    kmeshx, kmeshy = get_kmeshCE(atE=ax[num], EF=EF, slice_dim=slice_dim, extend=True)

    return kmeshx[0, :], np.flip(kmeshy[:, 0]), dd


def ali_get_kmeshCE(data_class, atE, extend=True, EF: float = None, slice_dim: str = None, scan_type: str = None,
                    theta_m: float = 0, phi_m: float = 0, theta0: float = 0, phi0: float = 0, alpha0: float = 0):
    """Get kmesh at energy from angles in Aa3D
    Args:
        data_class: Aa3D object
        atE: energy to get the kmesh at
        extend: if True, return the dimensions+1 as kmesh, for use with pcolormesh
        EF: optional parameter, fermi energy
        slice_dim: dimension of slice of data cube
        scan_type: deflector_scan or phi_motor_scan

    Returns:
        kmeshx, kmeshy, a tuple of arrays with the plot mesh

        """
    if scan_type == 'deflector_scan':
        x, y, z = data_class.xaxis, data_class.yaxis, data_class.zaxis
        xdelta = np.mean(np.diff(data_class.xaxis))
        ydelta = np.mean(np.diff(data_class.yaxis))
        zdelta = np.mean(np.diff(data_class.zaxis))

    elif scan_type == 'phi_motor_scan':
        x, y, z = ss, cs, p
        xdelta = np.mean(np.diff(ss))
        ydelta = np.mean(np.diff(cs))
        zdelta = np.mean(np.diff(p))

    else:
        return ValueError(f'scan_type is {scan_type}, must be deflector_scan or phi_motor_scan')

    if slice_dim == 'z':
        if extend:
            pyaxis = np.linspace(y[0] - 0.5 * ydelta, y[-1] + 0.5 * ydelta, (y.shape[0] + 1))
            pxaxis = np.linspace(x[0] - 0.5 * zdelta, x[-1] + 0.5 * xdelta, (x.shape[0] + 1))
        else:
            pyaxis = y
            pxaxis = x
    ameshth, ameshphi = np.meshgrid(pyaxis, pxaxis)
    # print(pyaxis, pzaxis)
    kmeshx, kmeshy = kfromangles(ameshth, ameshphi, (EF + atE), theta_m=theta_m, phi_m=phi_m, theta0=theta0, phi0=phi0,
                                 alpha0=alpha0)
                                 # kmeshx, kmeshy = kfromangles(ameshth,ameshphi,(16.8+atE),
                                 # **{k: data_class.berend_dataclass.p[k] for k in
                                 #    ['theta_m', 'phi_m', 'theta0', 'phi0', 'alpha0']})
    return kmeshx, kmeshy


if __name__ == '__main__':
    from arpes_functions.arpes_dataclasses import Data2D
    from plotting_functions import plot2D
    import matplotlib.pyplot as plt

    """2D Test (compare with Berend's results)"""
    # load data
    d = Data2D.single_load('December', year='2020', filename='UPS_20K0001_001.ibw')
    # plot regular 2D data
    d.berend_dataclass.show()  # Berend's
    bpl2(d.berend_dataclass)  # Berend's
    plot2D(d.xaxis, d.yaxis - 16.8, d.data)  # mine
    # correct EF
    fix_EkatEF(d, 16.8)
    # generate and plot k corrected data
    ax = kplot2D(d.berend_dataclass)  # Berend's
    plt.show()
    kx, ky, kdata = kcorrect2D(d)  # mine
    plot2D(kx, ky - 16.8, kdata)  # mine

    """3D Test (compare with Berend's results)"""
    # # # load data
    # # d = Data3D.single_load('January', scan_number=1)
    # d = Data3D.single_load('October', year='2020', scan_number=4, cryo_temp='RT')
    # d2 = AaData3D(r'C:\Users\atully\Code\ARPES Code Python\analysis_data\October_2020\RT\Lamp\3D\deflector_scan\OMBE_Lamp_3D0004', datatype='APE', zaxis='theta_y')
    # # plot regular CE slice
    # # d.berend_dataclass.show(mode='CE', val=15.6, Eint=0.02)  # Berend's method
    # # plotCE3D(d2, val=15.7, Eint=0.05, size=7)
    # # plot3D(x=d.yaxis, y=d.xaxis, z=d.zaxis, data=np.moveaxis(d.data, 2, 1), slice_dim='z', slice_val=15.6,
    # #               int_range=0.02)
    # # # correct EF
    # fix_EkatEF(d, 16.8)
    # d2.QuickCorrect(16.8)  # FIXME: This changes energy axis to binding energies in original dataset!!!
    # # # # generate and plot k corrected data
    # kx, ky, kdata = kcorrect3D(data_class=d, val=-1.2, Eint=0.02, EF=16.8)  # mine
    # # kx, ky, kdata = kcorrect3D(data_class=d2, val=-1.2, Eint=0.02, EF=16.8)  # mine
    # # # data_new = np.where(kdata > 1.3, 1.3, kdata)
    # fig = plot2D(kx, ky, kdata, xlabel='kx [A-1]', ylabel='ky [A-1]', title='Constant Energy Slice')  # mine
    # # plotCEk3D(d.berend_dataclass, val=15.7)  # Berend's
    # plotCEk3D(d2, val=-1.2, Eint=0.05, size=5, asp=True, pltbzn=False)  # Berend's
    # d.berend_dataclass.QuickCorrect(16.8)
    # plotCEk3D(d.berend_dataclass, val=-1.2, Eint=0.05, size=5, asp=True, pltbzn=False)  # Berend's
    # # # d2.setangles(0., -22.25, -3., 0., -5.5)
    # # plotCEk3D(d2, val=-1.2, Eint=0.05, size=5, asp=True, pltbzn=False)  # Berend's

    """Add Hexagons"""
    # coords = gen_polygon(6, radius=0.42, rotation=30, translation=(-0.07, 0.1))
    # new_coords, coords_bl, coords_tr, coords_l, coords_tl, coords_r = gen_tiled_hexagons(coords, radius=0.42,
    #                                                                                      rotation=30,
    #                                                                                      translation=(-0.07, 0.1))
    # plot_polygons([coords, new_coords, coords_bl, coords_tr, coords_l, coords_tl, coords_r], fig=fig)

    """3D Phi Motor Test"""
    # fp = r'C:\Users\atully\Code\ARPES Code Python\analysis_data\January_2021\LT\Lamp\3D\phi_motor_scan\FS_Y2021M01D25'
    # fn = r'FS_averaged.h5'
    # kx, ky, kdata = kcorrect_phimotor(fp=fp, fn=fn, slice_dim='y', val=-2.2, Eint=0.02, EF=16.9, theta0=0.5, phi0=-4)
    # fig = plot2D(kx, ky, kdata, xlabel='kx [A-1]', ylabel='ky [A-1]', title='Constant Energy Slice')
    #
    # """Add Hexagons"""
    # coords = gen_polygon(6, radius=0.42, rotation=15, translation=(0.2, -0.7))
    # # translation = (-0.14, -0.35)  # original (no vertical flip of ky)
    # # translation = (0.23, -0.5)   # flipped ky but no theta0 or phi0 offset angles
    # # translation = (0.2, -0.7) for theta0=0.5, phi0=-4   # flipped ky, hexagon #7 at gamma
    # # translation = (-0.5, -0.5) for theta0=0.5, phi0=-4   # flipped ky, hexagon #1 at gamma
    # new_coords, coords_bl, coords_tr, coords_l, coords_tl, coords_r = gen_tiled_hexagons(coords, radius=0.42,
    #                                                                                      rotation=15,
    #                                                                                      translation=(0.2, -0.7))
    # plot_polygons([coords, new_coords, coords_bl, coords_tr, coords_l, coords_tl, coords_r], fig=fig)
    # fig.add_vline(x=0)
    # fig.add_hline(y=0)
    # fig.show()
