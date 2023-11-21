"""
Loading functions
@author: Alexandra Tully
@date: February 2023
"""

import h5py
import os
import numpy as np
import pandas as pd
from igor.binarywave import load
import re


def load_hdf5(fp, fn=None):
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


def load_denoised_data(file_path: str, filename: str):
    data_path = os.path.join(file_path, filename)
    df = pd.read_csv(data_path, skiprows=3, header=None, sep='\s+', skipfooter=2)
    data = np.array(df).T
    with open(data_path, 'r') as f:
        lines = f.readlines()
    last = lines[-1]

    x_start, x_step = [float(v) for v in re.search('x\s*(-?\d*.\d+),\s*(-?\d+.\d+)', last).groups()]
    y_start, y_step = [float(v) for v in re.search('y\s*(-?\d*.\d+),\s*(-?\d+.\d+)', last).groups()]

    x = np.linspace(x_start, x_start+data.shape[1]*x_step, data.shape[1])
    y = np.linspace(y_start, y_start+data.shape[0]*y_step, data.shape[0])
    return x, y, data

