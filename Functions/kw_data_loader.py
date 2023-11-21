"""
K-corrected data file loader (using Sean's IGOR program to export kw data)
@author: Alexandra Tully
@date: November 2021
"""

import numpy as np
import os
import igor.binarywave


def load2D_k(ddir, fn, summary=True):
    """
    Notes: When creating axes, use data.shape[1] for x because numpy reads right to left
    (I tranzposed data matrix before that).
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
    xaxis, yaxis = (np.arange(data.shape[1], dtype='float32') * deltas[0]) + offsets[0], \
                   (np.arange(data.shape[0], dtype='float32') * deltas[1]) + offsets[1]  # create x and y axes

    params = wave['note'].decode('utf-8').split('\r')
    orig_filename = params[2][12:]
    orig_filepath = params[20][5:]
    comments = params[27][9:]
    datetime = params[28][5:] + ' ' + params[29][5:]
    if summary:
        summary = f'\033[1mFile Summary\033[0m\nfilename: {fn}\n' \
                  f'xaxis (kx): {len(xaxis)}, x_min: {np.min(xaxis):.2f}, x_max: {np.max(xaxis):.2f}\n' \
                  f'yaxis (energy): {len(yaxis)}, y_min: {np.min(yaxis):.2f}, y_max: {np.max(yaxis):.2f}\n' \
                  f'datetime: {datetime}\ncomments: {comments}\n' \
                  f'current filepath: {path}\nraw data filepath: {orig_filepath}'

        print(summary)
    return xaxis, yaxis, data


def load3D_k(ddir, fn, summary=True):
    """
    Notes: When creating axes, recall that numpy reads right to left (also, I transpose data matrix before that).
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
    # create x (=ss=kx)and y (=cs=energy) axes
    xaxis, yaxis = (np.arange(data.shape[2], dtype='float32') * deltas[0]) + offsets[0], \
                   (np.arange(data.shape[1], dtype='float32') * deltas[1]) + offsets[1]
    zaxis = np.arange(data.shape[0], dtype='float32') * deltas[2] + offsets[2]  # create zaxis (=p=ky)

    params = wave['note'].decode('utf-8').split('\r')
    theta_m = int(params[1][3:-4])
    theta_0 = int(params[2][4:-4])
    alpha_0 = int(params[3][7:-4])
    phi_0 = int(params[4][6:-4])

    if summary:
        summary = f'\033[1mFile Summary\033[0m\nfilename: {fn}\n' \
                  f'xaxis (kx): {len(xaxis)}, x_min: {np.min(xaxis):.2f}, x_max: {np.max(xaxis):.2f}\n' \
                  f'yaxis (energy): {len(yaxis)}, y_min: {np.min(yaxis):.2f}, y_max: {np.max(yaxis):.2f}\n' \
                  f'zaxis (ky): {len(zaxis)}, z_min: {np.min(zaxis):.2f}, z_max: {np.max(zaxis):.2f}\n' \
                  f'cryostat rotation (theta motor): {theta_m}\ntheta 0: {theta_0}\n' \
                  f'azimuth rotaton (alpha 0): {alpha_0}\nphi 0: {phi_0}\ncurrent filepath: {path}'

        print(summary)
    return xaxis, yaxis, zaxis, data
