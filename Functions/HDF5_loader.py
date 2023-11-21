"""
HDF5 file loader (required for phi_motor ARPES data)
@author: Alexandra Tully
@date: April 2021
"""

import h5py
import os
import numpy as np
from igor.binarywave import load


def array_from_hdf(fp_complete: str, dataname: str) -> np.ndarray:
    with h5py.File(fp_complete, 'r') as hdf:  # makes sure hdf file is closed at the end
        dataset = hdf.get(dataname)
        data = dataset[:]
    return data


# def array_from_hdf(fp_complete: str, dataname) -> np.ndarray:
#     with h5py.File(fp_complete, 'r') as hdf:  # makes sure hdf file is closed at the end
#         dataset = hdf.get(d for d in dataname)
#         datas = dataset[:]
#     return datas


def avg_array_from_hdfs(fp: str, fns: list) -> np.ndarray:
    datas = []
    for f in fns:
        filename = os.path.join(fp, f)
        data = array_from_hdf(filename, 'data')
        datas.append(data)
    return np.mean(datas, axis=0)


def data_from_hdf(fp: str, dataname: str):  # returns 4 ndarrays
    p = array_from_hdf(os.path.join(fp, dataname), 'p')  # ky (phi motor slices -- 102 steps, typically)
    ss = array_from_hdf(os.path.join(fp, dataname), 'slice_scale')  # kx
    cs = array_from_hdf(os.path.join(fp, dataname), 'channel_scale')  # energy
    data = array_from_hdf(os.path.join(fp, dataname), 'data')
    return data, ss, cs, p


def data_from_hdf_2022(fp: str, dataname: str):  # returns 4 ndarrays
    p = array_from_hdf(os.path.join(fp, dataname), 'entry1/ScanValues').squeeze()  # ky (phi motor slices -- 102 steps, typically)
    ss = array_from_hdf(os.path.join(fp, dataname), 'angles')[:, 0]  # theta (kx)
    cs = array_from_hdf(os.path.join(fp, dataname), 'energies')[:, 0]  # energy
    data = array_from_hdf(os.path.join(fp, dataname), 'data')
    return data, ss, cs, p


def data_from_hdf_2D_2022(fp: str, dataname: str):  # returns 4 ndarrays
    ss = array_from_hdf(os.path.join(fp, dataname), 'angles').squeeze()  # theta (kx)
    cs = array_from_hdf(os.path.join(fp, dataname), 'energies').squeeze()  # energy
    data = array_from_hdf(os.path.join(fp, dataname), 'data').squeeze()
    return data, ss, cs


def avg_data_hdf(fp: str, fn: str, data_avg: np.ndarray, p: np.ndarray, slice_scale: np.ndarray,
                 channel_scale: np.ndarray):
    filepath = os.path.join(fp, f'{fn}.h5')
    with h5py.File(filepath, 'w') as hdf:
        hdf['data'] = data_avg
        hdf['p'] = p
        hdf['slice_scale'] = slice_scale
        hdf['channel_scale'] = channel_scale


def ibw_to_hdf5(fp, fn, h5file='test.h5', export=False):
    """Loads .ibw file, converts to h5. h5file is string --> desired filename + location"""
    w = load(os.path.join(fp, fn))
    data = w['wave']['wData']
    # print(w)  # To see the full contents of the igorwave (it is a dictionary of information)

    # Axis info is saved as start value and delta value only (e.g. np.arange(100)*sfA+sfB to get 100 values starting at sfB and
    # stepping by sfA)
    axis_delta = w['wave']['wave_header']['sfA']  # Each of these has 4 values (for the 4 possible dims of Igor waves)
    axis_start = w['wave']['wave_header']['sfB']  # ""
    axis_shape = w['wave']['wave_header']['nDim']

    axes = []
    for delta, start, shape in zip(axis_delta, axis_start, axis_shape):
        if shape > 0:  # Ignore unused axes
            axis = np.arange(shape) * delta + start
            axes.append(axis)

    # printing first few values from each so you can figure out which is which
    for axis in axes:
        print(axis[:10])

    axes_names = ['theta', 'energy', 'phi']  # Change these to match your axes labels

    if export is True:
        fn = fn if h5file == 'test.h5' else h5file
        h5file = os.path.join(fp, fn.split('.')[0]+'.h5')
    with h5py.File(h5file,
                   'w') as f:  # Note: 'w' creates a new empty file (or overwrites), use 'r+' to modify an existing file
        f['data'] = data
        for axis, name in zip(axes, axes_names):
            f[name] = axis


def load_hdf5(fp, fn=None):  # THIS FUNCTION IS IN LOADING_FUNCTIONS TOO!
    """Loads HDF5 file and returns the following numpy arrays: data, theta axis, phi axis, energy axis"""
    if fn is not None:
        filepath = os.path.join(fp, fn)
    else:
        filepath = fp

    def _new_loader(fp, fn):
        with h5py.File(filepath, "r") as f:  # Read only
            loaded_data = f["data"][:].squeeze()  # Squeeze removes any axes with size 1
            # [:] to convert h5py.Dataset to numpy array (otherwise it dissapears outside of the with statement)
            #     loaded_axes = [f[key][:] for key in axes_names]
            energy_ax = f["energies"][:, 0]
            theta_ax = f["angles"][:, 0]

            phi_or_time_ax = (
                f["entry1"]["ScanValues"][:, 0]
                if "ScanValues" in f["entry1"].keys()
                else None
            )
        if phi_or_time_ax is not None:
            return loaded_data.T, theta_ax, phi_or_time_ax, energy_ax
        else:
            return loaded_data.T, theta_ax, energy_ax

    def _old_loader(fp, fn):
        with h5py.File(filepath, "r") as f:  # Read only
            loaded_data = f["data"][:]
            # [:] to convert h5py.Dataset to numpy array (otherwise it dissapears outside of the with statement)
            #     loaded_axes = [f[key][:] for key in axes_names]
            #             axes_names = ['theta', 'energy', 'phi']  # Change these to match your axes labels
            #             axes_dict = {key: f[key] for key in axes_names if key in f.keys()}
            energy_ax = f["energy"][:]
            theta_ax = f["theta"][:]
            phi_ax = f["phi"][:] if "phi" in f.keys() else None
        if phi_ax is not None:
            return loaded_data.T, theta_ax, phi_ax, energy_ax
        else:
            return loaded_data.T, theta_ax, energy_ax

    with h5py.File(filepath, "r") as f:
        if "energies" in f.keys():
            new = True
        else:
            new = False

    if new:
        return _new_loader(fp, fn)
    else:
        return _old_loader(fp, fn)


# def load_hdf5(fp, fn=None):
#     """Loads HDF5 file and returns the following numpy arrays: data, theta axis, phi axis, energy axis"""
#     if fn is not None:
#         filepath = os.path.join(fp, fn)
#     else:
#         filepath = fp
#     with h5py.File(filepath, 'r') as f:  # Read only
#         loaded_data = f['data'][
#                       :]  # [:] to convert h5py.Dataset to numpy array (otherwise it dissapears outside of the with statement)
#         #     loaded_axes = [f[key][:] for key in axes_names]
#         axes_names = ['theta', 'energy', 'phi']  # Change these to match your axes labels
#         axes_dict = {key: f[key] for key in axes_names if key in f.keys()}
#         energy_ax = f['energy'][:]
#         theta_ax = f['theta'][:]
#         phi_ax = f['phi'][:] if 'phi' in f.keys() else None
#     if phi_ax is not None:
#         return loaded_data.T, theta_ax, phi_ax, energy_ax
#     if not phi_ax:
#         return loaded_data.T, theta_ax, energy_ax


if __name__ == '__main__':
    path = r'C:\Users\atully\Code\ARPES Code Python\analysis_data\January_2021\LT\Lamp\3D\phi_motor_scan' \
           r'\HOMO15_Y2021M01D27\HOMO15_Y2021M01D28dT08h36m19s.h5 '
    hdf = h5py.File(path, 'r')
    hdf.keys()  # shows groups and datasets
    hdf.attrs.keys()  # shows attributes
    hdf['channel_scale'].attrs.keys()
    cs = hdf.get('channel_scale')  # nice way of getting things from my hdf file
    cs  # cs is a dataset, if close hdf file this dataset will close as well (can't do anything with it)
    cs_data = cs[:]  # numpy array, this will stay open after I close hdf
    cs_data  # energy scale
    cs_data.shape  # (1064,)
    hdf.close()
