"""
Filtering functions
@author: Alexandra Tully
@date: June 2021
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from copy import deepcopy
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import os
import h5py

from arpes_functions.plotting_functions import plot2D
from arpes_functions.arpes_dataclasses import Data2D
from arpes_functions.analysis_functions import get_data_region, get_2Dslice


def fourier_2d(data: np.ndarray, xaxis: np.ndarray, yaxis: np.ndarray):
    fft = np.fft.fft2(data)
    lx, ly = np.shape(xaxis)[0], np.shape(yaxis)[0]
    dx = (xaxis[-1] - xaxis[0]) / lx
    dy = (yaxis[-1] - yaxis[0]) / ly
    x_fft = 1 / dx * np.linspace(-0.5, 0.5, lx)
    y_fft = 1 / dy * np.linspace(-0.5, 0.5, ly)
    dat_fft = 2 * np.abs(np.fft.fftshift(fft))
    return dat_fft, x_fft, y_fft


def generate_FFT_filtered_dataset(
    theta, energy, phi, data, fp, fn, int_range=0.0, overwrite=False, new_fn=None
):
    new_fn = (
        new_fn
        if new_fn
        else f"{os.path.splitext(fn)[0]}_filteredFFT_{int_range:.2f}int.h5"
    )
    new_fn = os.path.join(fp, new_fn)
    if not overwrite and os.path.exists(new_fn):
        raise FileExistsError(f"{new_fn} already exists")
    new_data = []
    for p in phi:
        val = p
        xaxis, yaxis, dataslice = get_2Dslice(
            x=theta,
            y=energy,
            z=phi,
            data=data,
            slice_dim="z",
            slice_val=val,
            int_range=int_range,
        )
        fft_data = fft2d_mask(dataslice, plot=False)
        new_data.append(fft_data)
    new_data = np.array(new_data).T
    with h5py.File(
        new_fn, "w"
    ) as f:  # Note: 'w' creates a new empty file (or overwrites), use 'r+' to modify an existing file
        f["data"] = new_data
        axes_names = [
            "theta",
            "energy",
            "phi",
        ]  # Change these to match your axes labels
        axes = [theta, energy, phi]
        for axis, name in zip(axes, axes_names):
            f[name] = axis
    return new_fn


def fft2d_mask(Aa2D, MT='Zero', MS=[46,20], ER=[45,55], plot=True):
    '''
    Takes 2D data and finds its fft. Output is modified data
    params:
        Aa2D.data(array): 2D array of input data
        MT(str): Mask Type
        MS(list): Mask size
        ER(list): exclusion region - region to exclude from mask
        plot(bool): true generates plots, False does not generate plots
    returns: new modified data
    '''
    Aa2D = np.nan_to_num(Aa2D)
    Aa2D=np.flipud(Aa2D)
    #Fourier transform
    Aa2D_fft=np.fft.fft2(Aa2D)
    Aa2D_fft_shift=np.fft.fftshift(Aa2D_fft)
    dim=Aa2D_fft.shape
    ind0y=round(dim[0]/2)+1
    ind0x=round(dim[1]/2)+1
    y=np.arange(0, dim[0]) - ind0y+1
    x=np.arange(0, dim[1]) - ind0x+1
    #peak placements --> hard coded god for scans 1064by1000
    A=abs(Aa2D_fft_shift)
    Aa2D_fft_mod = deepcopy(Aa2D_fft_shift)
    dx=43
    dy=70
    xpk1=np.arange(-3*dx,3*dx+dx,dx)
    ypk1=-2*xpk1
    xpk2=np.arange(-3*dy,3*dy+dy, dy)
    ypk2=0.75*xpk2
    xpk3=xpk2-dx
    ypk3=0.75*xpk3+120
    xpk4=xpk2+dx
    ypk4=0.75*xpk4-120
    xpk5=xpk1+(dy-dx)
    ypk5=-2*xpk5+192.5
    xpk6=xpk1-(dy-dx)
    ypk6=-2*xpk6-192.5
    indx=np.concatenate((xpk1,xpk2,xpk3,xpk4,xpk5,xpk6))+ind0x #list of x indices
    indy=np.concatenate((ypk1,ypk2,ypk3,ypk4,ypk5,ypk6))+ind0y #list of y indices
    ind=np.array([indx,indy]) #array [0] has x indices, [1] has y indices
    ind=np.round(ind) #make them all int
    dim2=ind.shape
    dxind=MS[0]
    dyind=MS[1]
    #make mask
    for i in range(0,dim2[1]):
        indx=ind[0,i]
        indy=ind[1,i]
        if indx<=-ER[0]+ind0x or indx>=ER[0]+ind0x or indy<=-ER[1]+ind0y or indy>=ER[1]+ind0y:  # either apecifically applies to certain region or doesn't apply to those regions. How did you come up with the hard coded values?
            Masky=np.arange(indy-dyind/2, indy+dyind/2+1).astype(int)
            Maskx=np.arange(indx-dxind/2, indx+dxind/2+1).astype(int)
            if MT=='Zero':
                Aa2D_fft_mod[Masky[0]:Masky[-1]+1, Maskx[0]:Maskx[-1]+1]=1e-10
    #unshift
    Aa2D_fft_mod_unshift = np.fft.ifftshift(Aa2D_fft_mod)
    #ifft
    Aa2D_ifft_mod = np.fft.ifft2(Aa2D_fft_mod_unshift)
    if plot:
        plt.figure()
        plt.imshow(Aa2D)
        plt.title('test.txt')
        plt.figure()
        plt.imshow(abs(Aa2D_fft_shift))
        plt.title('fft of test.txt')
        plt.figure()
        plt.imshow(abs(Aa2D_fft_mod))
        plt.scatter(xpk1+ind0x, ypk1+ind0y, color='green', s=10)
        plt.scatter(xpk2+ind0x, ypk2+ind0y, color='green', s=10)
        plt.scatter(xpk3+ind0x, ypk3+ind0y, color='green', s=10)
        plt.scatter(xpk4+ind0x, ypk4+ind0y, color='green', s=10)
        plt.scatter(xpk5+ind0x, ypk5+ind0y, color='green', s=10)
        plt.scatter(xpk6+ind0x, ypk6+ind0y, color='green', s=10)
        plt.title('fft of test.txt and peak positions')
        plt.figure()
        plt.imshow(abs(Aa2D_fft_mod))
        plt.title('Modified fft')
        plt.figure()
        plt.imshow(abs(Aa2D_ifft_mod))
        plt.title('Modified test')
        plt.figure()
        plt.imshow(Aa2D-abs(Aa2D_ifft_mod))
        plt.title('Difference')
    return abs(np.flipud(Aa2D_ifft_mod))
#     return abs(Aa2D_ifft_mod)


def enhance_image(fp, old_fn, new_fn, factor = 1.6):
    im = os.path.join(fp, old_fn)
    im3 = Image.open(im)  #im is full filepath
    enhancer = ImageEnhance.Contrast(im3)  #image brightness enhancer
    factor = factor #increase contrast
    im_output = enhancer.enhance(factor)
    im_output.save(os.path.join(fp, new_fn))  # new image filepath
    return im_output


if __name__ == '__main__':

    """ Load Raw Data """

    # d = Data2D.single_load('January', year='2021', light_source='XUV', filename='OMBE_XUV_2D0004_.ibw')
    data = Data2D.single_load(month='January', year='2021', light_source='XUV', scan_number=4)
    plot2D(data.xaxis, data.yaxis, data.data, title='OMBE_XUV_2D0004_.ibw, January 2021', xlabel='Theta', ylabel='KE')

    # zoom in on cone
    d, x, y = get_data_region(data.data, data.xaxis, data.yaxis, xbounds=(-12, 8), ybounds=(17.1, 18.4), EB=False)
    fig = plot2D(x, y, d, title='OMBE_XUV_2D0004_.ibw, January 2021', xlabel='Theta', ylabel='KE')

    """ Apply Filters """

    # gaussian mask
    sigma = 2
    gauss_data = gaussian_filter(d, sigma=sigma)
    fig = plot2D(x, y, gauss_data,
                 title=f'OMBE_XUV_2D0004_.ibw, January 2021, Gaussian mask (sigma: {sigma})',
                 xlabel='Theta', ylabel='KE')