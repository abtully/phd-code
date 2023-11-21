"""
General fitting functions
@author: Alexandra Tully
@date: June 2021

Note: Includes linear, lorentzian, and hyperbolic fits, as well an offset function (constant, linear, or quadratic).
"""

import numpy as np
import lmfit as lm
from scipy.special import erf
from typing import Union, Optional

from arpes_functions import analysis_functions, tr_functions


def offset_model(offset_type: str, a: float = None, b: float = None, c: float = None) -> Union[lm.models.QuadraticModel, lm.models.PolynomialModel]:
    """

    Args:
        offset_type: constant, linear, or quadratic
        a: quadratic coefficient (ax^2 + bx + c)
        b: linear coefficient
        c: constant coefficient

    Returns:
        offset model to be added to lmfit model (e.g. model + offset_model)

    """
    if offset_type in ('constant', 'linear', 'quadratic'):
        model = lm.models.QuadraticModel()

        model.set_param_hint('a', value=a)
        model.set_param_hint('b', value=b)
        model.set_param_hint('c', value=c)

        if offset_type == 'quadratic':
            pass
        elif offset_type == 'linear':
            model.set_param_hint('a', value=0, vary=False)  # not allowed to vary parameter 'a' while fitting
        elif offset_type == 'constant':
            model.set_param_hint('a', value=0, vary=False)
            model.set_param_hint('b', value=0, vary=False)
        return model
    elif offset_type.startswith('degree='):
        degree = int(offset_type[7:])
        model = lm.models.PolynomialModel(degree=degree)
        for i, v in enumerate([c, b, a]):  # Set the first three with initial values
            if v is not None:
                model.set_param_hint(f'c{i}', value=v)
        if degree >= 3:
            for i in range(3, degree+1):  # Set the rest to zero
                model.set_param_hint(f'c{i}', value=0)
        return model
    else:
        raise ValueError(f'offset_type: {offset_type} is not quadratic, linear, constant, or degree=int')


def make_line(num, a, b, pos_slope=True) -> lm.models.LinearModel:
    """

    Args:
        num: index of line (0, 1, 2)
        a: initial guess (y = ax + b)
        b: initial guess

    Returns:
        linear model for fit

    """
    pref = f'i{num}_'
    model = lm.models.LinearModel(prefix=pref)
    if pos_slope:
        model.set_param_hint(pref + 'a', value=a, min=0, max=5 * a)
    elif not pos_slope:
        model.set_param_hint(pref + 'a', value=a, min=5 * a, max=0)
    model.set_param_hint(pref + 'b', value=b, min=b - 50, max=b + 50)
    return model


def make_n_lines(num, aes: Union[list, float], bes: Union[list, float]) -> lm.Model:
    """

    Args:
        num: number of lines for fit
        aes: initial guess(es)
        bes: initial guesses(es)

    Returns:
        lmfit model

    """
    if not isinstance(aes, list):
        aes = [aes] * num
    if not isinstance(bes, list):
        bes = [bes] * num

    if any([len(arr) != num for arr in [aes, bes]]):
        raise ValueError(f'length of {aes}, {bes}, not all equal to {num}')

    model = None
    for i, a, b in zip(range(num), aes, bes):
        if a >= 0:
            this_model = make_line(i, a, b, pos_slope=True)
        elif a < 0:
            this_model = make_line(i, a, b, pos_slope=False)
        if not model:
            model = this_model
        else:
            model += this_model
    return model


def fit_linear_data(x: np.ndarray, data: np.ndarray,
                        num: int,
                        aes: Union[list, float] = None, bes: Union[list, float] = None,
                        offset_type: Optional[str] = None, a: float = None, b: float = None, c: float = None) \
        -> lm.model.ModelResult:
    """

    Args:
        x: 1D numpy array xdata
        data: 1D numpy array ydata
        num: number of lines
        amplitudes: initial guess(es)
        centers: initial guesses
        sigmas: initial guess(es)
        offset_type: constant, linear, or quadratic
        a: quadratic coefficient (ax^2 + bx + c)
        b: linear coefficient
        c: constant coefficient

    Returns:
        lmfit model result

    """
    if c is None:
        c = np.mean(data)
    if b is None:
        b = (data[-1] - data[0]) / (x[-1] - x[0])
    if a is None:
        a = 0

    if bes is None:
        bes = np.mean(data)
    if aes is None:
        aes = (data[-1] - data[0]) / (x[-1] - x[0])

    lines = make_n_lines(num, aes, bes)
    if offset_type:
        offset = offset_model(offset_type, a, b, c)
        model = lines + offset
    else:
        model = lines

    fit = model.fit(data.astype(np.float32), x=x.astype(np.float32))

    return fit


def make_lorentzian(num, amplitude, center, sigma) -> lm.models.LorentzianModel:
    """

    Args:
        num: index of lorentzian (0, 1, 2)
        amplitude: initial guess
        center: initial guess
        sigma: initial guess

    Returns:
        lorentzian model for fit

    """
    pref = f'i{num}_'
    model = lm.models.LorentzianModel(prefix=pref)
    model.set_param_hint(pref + 'amplitude', value=amplitude, min=0, max=5 * amplitude)
    model.set_param_hint(pref + 'center', value=center, min=center - 10, max=center + 10)
    model.set_param_hint(pref + 'sigma', value=sigma, min=0, max=10)
    return model


def make_n_lorentzians(num, amplitudes: Union[list, float], centers: list, sigmas: Union[list, float]) -> lm.Model:
    """

    Args:
        num: number of lorentzians for fit
        amplitudes: initial guess(es)
        centers: initial guesses
        sigmas: initial guess(es)

    Returns:
        lmfit model

    """
    if not isinstance(amplitudes, list):
        amplitudes = [amplitudes] * num
    if not isinstance(sigmas, list):
        sigmas = [sigmas] * num

    if any([len(arr) != num for arr in [amplitudes, centers, sigmas]]):
        raise ValueError(f'length of {amplitudes}, {centers}, {sigmas} not all equal to {num}')

    model = None
    for i, amp, cent, sig in zip(range(num), amplitudes, centers, sigmas):
        this_model = make_lorentzian(i, amp, cent, sig)
        if not model:
            model = this_model
        else:
            model += this_model
    return model


def fit_lorentzian_data(x: np.ndarray, data: np.ndarray,
                        num_peaks: int,
                        amplitudes: Union[list, float], centers: list, sigmas: Union[list, float] = None,
                        offset_type: Optional[str] = 'linear', a: float = None, b: float = None, c: float = None,
                        method: str = 'leastsq',
                        params = None) \
        -> lm.model.ModelResult:
    """

    Args:
        x: 1D numpy array xdata
        data: 1D numpy array ydata
        num_peaks: number of lorentzian peaks
        amplitudes: initial guess(es) for lorentzian peaks
        centers: initial guesses for lorentzian peaks
        sigmas: initial guess(es) for lorentzian peaks
        offset_type: constant, linear, or quadratic
        a: quadratic coefficient (ax^2 + bx + c)
        b: linear coefficient
        c: constant coefficient
        method: minimization methosd (e.g. leastsq, powell, etc.)

    Returns:
        lmfit model result

    """
    if c is None:
        c = np.mean(data)
    if b is None:
        b = (data[-1] - data[0]) / (x[-1] - x[0])
    if a is None:
        a = 0
    if sigmas is None:
        sigmas = 1

    lorentzians = make_n_lorentzians(num_peaks, amplitudes, centers, sigmas)
    if offset_type:
        offset = offset_model(offset_type, a, b, c)
        model = lorentzians + offset
    else:
        model = lorentzians

    if params is None:
        params = model.make_params()
    for p in params.keys():
        if 'center' in p:
            par = params[p]
            par.min = np.nanmin(x)
            par.max = np.nanmax(x)

    fit = model.fit(data.astype(np.float32), x=x.astype(np.float32), method=method, params=params)

    return fit


def hyperbola(x, a, b, h, k, pos=False):
    # return -np.sqrt(a**2 * b**2 + a**2 * (x-h)**2) / b + k
    if pos:
        return np.sqrt(a ** 2 * (1 + (x - h) ** 2 / b ** 2)) + k
    else:
        return -np.sqrt(a**2 * (1 + (x-h)**2 / b**2)) + k


def make_hyperbola(num, a, b, h, k) -> lm.Model:
    """

    Args:
        num: index of hyperbola (0, 1, 2)
        a: initial guess
        b: initial guess
        h: initial guess for center (h, k)
        k: initial guess for center (h, k)

    Returns:
        lmfit model of hyperbola

    """
    pref = f'i{num}_'
    model = lm.Model(hyperbola, prefix=pref)
    model.set_param_hint(pref + 'a', value=a)
    model.set_param_hint(pref + 'b', value=b)
    model.set_param_hint(pref + 'h', value=h, min=h - 5, max=h + 5)
    model.set_param_hint(pref + 'k', value=k, min=k-4, max=k+4)
    return model


def make_n_hyperbolas(num,
                      aes: Union[list, float], bes: Union[list, float], hes: Union[list, float], kes: Union[list, float]) \
        -> lm.Model:
    """

    Args:
        num: number of hyperbolas for fit
        aes: initial guess(es)
        bes: initial guess(es)
        hes: initial guess(es) for center (h, k)
        kes: initial guess(es) for center (h, k)

    Returns:
        lmfit model

    """
    if not isinstance(aes, list):
        aes = [aes] * num
    if not isinstance(bes, list):
        bes = [bes] * num
    if not isinstance(hes, list):
        hes = [hes] * num
    if not isinstance(kes, list):
        kes = [kes] * num

    if any([len(arr) != num for arr in [aes, bes, hes, kes]]):
        raise ValueError(f'length of {aes}, {bes}, {hes}, {kes} not all equal to {num}')

    model = None
    for i, a, b, h, k in zip(range(num), aes, bes, hes, kes):
        this_model = make_hyperbola(i, a, b, h, k)
        if not model:
            model = this_model
        else:
            model += this_model
    return model


def fit_hyperbola_data(x: np.ndarray, data: np.ndarray,
                        num: int,
                        aes: Union[list, float], bes: Union[list, float], hes: Union[list, float], kes: Union[list, float],
                        offset_type: Optional[str] = 'linear', a: float = None, b: float = None, c: float = None) \
        -> lm.model.ModelResult:
    """

    Args:
        x: 1D numpy array xdata
        data: 1D numpy array ydata
        num: number of hyperbolas to fit
        aes: initial guess(es)
        bes: initial guess(es)
        hes: initial guess(es) for center (h, k)
        kes: initial guess(es) for center (h, k)
        offset_type: constant, linear, or quadratic
        a: quadratic coefficient (ax^2 + bx + c)
        b: linear coefficient
        c: constant coefficient

    Returns:
        lmfit model result

    """
    if c is None:
        c = np.mean(data)
    if b is None:
        b = (data[-1] - data[0]) / (x[-1] - x[0])
    if a is None:
        a = 0
    # if sigmas is None:
    #     sigmas = 1

    hyperbolas = make_n_hyperbolas(num, aes, bes, hes, kes)
    if offset_type:
        offset = offset_model(offset_type, a, b, c)
        model = hyperbolas + offset
    else:
        model = hyperbolas

    fit = model.fit(data.astype(np.float32), x=x.astype(np.float32))

    return fit


def make_hyperbola_asymptotes(fit, x):
    """

    Args:
        fit: lmfit model fit
        x: 1D numpy array xdata

    Returns:
        y1, y2 lines for hyperbola

    """
    a, b, h, k = fit.best_values['i0_a'], fit.best_values['i0_b'], fit.best_values['i0_h'], fit.best_values['i0_k']
    y1 = a / b * (x - h) + k
    y2 = - a / b * (x - h) + k
    return y1, y2


def convolved_gaussian_exponential(x, I, t0, sigma, gamma):  # using analytical formula
    # sigma = tr_functions.fwhm_to_sig(fwhm)
    # gamma = 1 / tau
    return (
        (I * gamma / 2)
        * np.exp(gamma * (t0 - x + gamma * sigma**2 / 2))
        * (1 - erf((t0 + gamma * sigma**2 - x) / (sigma * np.sqrt(2))))
    )


def make_gaussian(num, amplitude, center, sigma, include_exp_decay=False, gamma=None, lock_sigma=False) -> lm.models.GaussianModel:
    """

    Args:
        num: index of gaussian (0, 1, 2)
        amplitude: initial guess
        center: initial guess
        sigma: initial guess
        include_exp_decay: if True, model returned is lm.models.ExponentialGaussianModel, where Gaussian is convolved
                        with an exponential
        gamma: initial guess for lifetime if using the convolved exponential model

    Returns:
        lmfit model

    """
    pref = f'i{num}_'
    if include_exp_decay:
        model = lm.models.ExponentialGaussianModel(prefix=pref)
        # model = lm.models.Model(convolved_gaussian_exponential, prefix=pref)
        model.set_param_hint(pref + 'gamma', value=gamma, min=1e-6, max=1e3)
    else:
        model = lm.models.GaussianModel(prefix=pref)
    model.set_param_hint(pref + 'amplitude', value=amplitude, min=0, max=5 * amplitude)
    model.set_param_hint(pref + 'center', value=center, min=center - 10, max=center + 10)
    if lock_sigma:
        model.set_param_hint(pref + 'sigma', value=sigma, min=0, max=10, vary=False)
    else:
        model.set_param_hint(pref + 'sigma', value=sigma, min=0, max=10)
    return model


def make_n_gaussians(num, amplitudes: Union[list, float], centers: list, sigmas: Union[list, float],
                     include_exp_decay=False, gammas: Union[list, float] = None, lock_sigma=False) -> lm.Model:
    """

    Args:
        num: number of gaussians for fit
        amplitudes: initial guess(es)
        centers: initial guesses
        sigmas: initial guess(es)
        include_exp_decay: if True, model returned is lm.models.ExponentialGaussianModel, where Gaussian is covolved
                        with an exponential
        gammas: initial guess(es) for lifetime if using the convolved exponential model

    Returns:
        lmfit model

    """
    if include_exp_decay:
        if gammas is None:
            raise ValueError(f'gamma={gammas}, but must have an initial guess for lifetime of state if convolving with '
                             f'exponential. try gamma=20.')
        else:
            if not isinstance(amplitudes, list):
                amplitudes = [amplitudes] * num
            if not isinstance(sigmas, list):
                sigmas = [sigmas] * num
            if not isinstance(gammas, list):
                gammas = [gammas] * num

            if any([len(arr) != num for arr in [amplitudes, centers, sigmas, gammas]]):
                raise ValueError(f'length of {amplitudes}, {centers}, {sigmas}, {gammas} not all equal to {num}')

            model = None
            for i, (amp, cent, sig, gamma) in enumerate(zip(amplitudes, centers, sigmas, gammas)):
                this_model = make_gaussian(i, amp, cent, sig, include_exp_decay=True, gamma=gamma, lock_sigma=lock_sigma)
                if not model:
                    model = this_model
                else:
                    model += this_model
        return model

    else:
        if not isinstance(amplitudes, list):
            amplitudes = [amplitudes] * num
        if not isinstance(sigmas, list):
            sigmas = [sigmas] * num

        if any([len(arr) != num for arr in [amplitudes, centers, sigmas]]):
            raise ValueError(f'length of {amplitudes}, {centers}, {sigmas} not all equal to {num}')

        model = None
        for i, amp, cent, sig in zip(range(num), amplitudes, centers, sigmas):
            this_model = make_gaussian(i, amp, cent, sig)
            if not model:
                model = this_model
            else:
                model += this_model
        return model


def fit_gaussian_data(x: np.ndarray, data: np.ndarray,
                        num_peaks: int,
                        amplitudes: Union[list, float], centers: list, sigmas: Union[list, float] = None,
                        include_exp_decay = False, gammas: Union[list, float] = None, lock_sigma = False,
                        offset_type: Optional[str] = 'linear', a: float = None, b: float = None, c: float = None,
                        method: str = 'leastsq',
                        params = None) \
        -> lm.model.ModelResult:
    """

    Args:
        x: 1D numpy array xdata
        data: 1D numpy array ydata
        num_peaks: number of gaussian peaks
        amplitudes: initial guess(es) for gaussian peaks
        centers: initial guesses for gaussian peaks
        sigmas: initial guess(es) for gaussian peaks
        include_exp_decay: if True, model returned is lm.models.ExponentialGaussianModel, where Gaussian is covolved
                        with an exponential
        gammas: initial guess(es) for lifetime if using the convolved exponential model
        offset_type: constant, linear, or quadratic
        a: quadratic coefficient (ax^2 + bx + c)
        b: linear coefficient
        c: constant coefficient
        method: minimization methosd (e.g. leastsq, powell, etc.)

    Returns:
        lmfit model result

    """
    if c is None:
        c = np.mean(data)
    if b is None:
        b = (data[-1] - data[0]) / (x[-1] - x[0])
    if a is None:
        a = 0
    if sigmas is None:
        sigmas = 1
    if gammas is None:
        gammas = 0

    gaussians = make_n_gaussians(num_peaks, amplitudes, centers, sigmas, include_exp_decay, gammas,
                                 lock_sigma=lock_sigma)
    if offset_type:
        offset = offset_model(offset_type, a, b, c)
        model = gaussians + offset
    else:
        model = gaussians

    if params is None:
        params = model.make_params()
    for p in params.keys():
        if 'center' in p:
            par = params[p]
            par.min = np.nanmin(x)
            par.max = np.nanmax(x)

    fit = model.fit(data.astype(np.float32), x=x.astype(np.float32), method=method, params=params)

    return fit


def fit_partial_cone(x, y, data, xlim, ylim, centers=[0.25, 0.4], num_peaks=2, window=0.02, steps=20,
                     offset_type='quadratic', plot=True):
    x_fit, y_fit, d_fit = analysis_functions.limit_dataset(x, y, data, xlim=(xlim[0], xlim[1]), ylim=(ylim[0], ylim[1]))

    # fit top of cone
    if num_peaks == 1:
        fits = []
        coords = []

        for yval in np.linspace(ylim[0], ylim[1], steps):
            row = analysis_functions.get_averaged_slice(
                analysis_functions.get_horizontal_slice(d_fit, y_fit, yval, window), axis='y')

            fit = fit_lorentzian_data(x=x_fit, data=row, num_peaks=num_peaks, amplitudes=0.5, centers=[centers[0]],
                                      sigmas=0.05, offset_type=offset_type)
            fits.append(fit)
            coords.extend([(fit.best_values[f'i{i}_center'], yval) for i in range(num_peaks)])
        return coords

    if num_peaks == 2:
        fits1 = []
        coords1 = []

        for yval in np.linspace(ylim[0], ylim[1], steps):
            row = analysis_functions.get_averaged_slice(
                analysis_functions.get_horizontal_slice(d_fit, y_fit, yval, window), axis='y')
            fit1 = fit_lorentzian_data(x=x_fit, data=row, num_peaks=num_peaks, amplitudes=0.5,
                                       centers=[centers[0], centers[1]], sigmas=0.05, offset_type=offset_type)
            fits1.append(fit1)
            coords1.extend([(fit1.best_values[f'i{i}_center'], yval) for i in range(num_peaks)])
        return coords1

    else:
        return ValueError(f'num_peaks is {num_peaks}; must be 1 or 2')


def compile_peaks(coords_top, coords_middle, coords_bottom):
    x_line1, y_line1 = np.array([c[0] for c in coords_top]), np.array([c[1] for c in coords_top])

    x_line2 = np.append(x_line1, np.array([c[0] for c in coords_middle]), axis=0)
    x_line_all = np.append(x_line2, np.array([c[0] for c in coords_bottom]), axis=0)

    y_line2 = np.append(y_line1, np.array([c[1] for c in coords_middle]), axis=0)
    y_line_all = np.append(y_line2, np.array([c[1] for c in coords_bottom]), axis=0)

    return x_line_all, y_line_all


def fit_cone_lines(x_line, y_line, cone_center=None):
    if cone_center is None:
        cone_center = np.mean(x_line)
    # Fits all data points (RIGHT SIDE OF CONE)
    fit_line1 = fit_linear_data(x=x_line[np.where(x_line > cone_center)],
                                                  data=y_line[np.where(x_line > cone_center)],
                                                  num=1,
                                                  aes=1, bes=1,
                                                  offset_type=None)

    # Fits all data points (LEFT SIDE OF CONE)
    fit_line2 = fit_linear_data(x=x_line[np.where(x_line < cone_center)],
                                                  data=y_line[np.where((x_line < cone_center))],
                                                  num=1,
                                                  aes=1, bes=1,
                                                  offset_type=None)

    return fit_line1, fit_line2


def fit_peaks(
    x,
    y,
    data,
    slice_dim,
    xlim,
    ylim,
    centers,
    slice_window,
    steps,
    num_peaks=1,
    offset_type="linear",
    model="Gaussian",
    amplitudes=1,
    sigmas=0.05,
    include_exp_decay=True,
    gammas=19,
    get_coords=False,
):
    """
    - Returns a list of fits and (optinally) coordinates (tuple) of center of peak based on a series of horizontal or
    vertical slices of a 2D array (x, y, data). Useful for plotting the center locations of a peak through energy or
    time.
    """
    data = data.astype(np.float32)
    x = x.astype(np.float32)
    y = y.astype(np.float32)

    if xlim is None:
        xlim = (np.min(x), np.max(x))
    if ylim is None:
        ylim = (np.min(y), np.max(y))

    x_fit, y_fit, d_fit = analysis_functions.limit_dataset(
        x, y, data, xlim=(xlim[0], xlim[1]), ylim=(ylim[0], ylim[1])
    )

    fits = []
    coords = []

    if num_peaks == 1:
        center_vals = [centers[0]]
    elif num_peaks == 2:
        center_vals = [centers[0], centers[1]]
    else:
        raise ValueError(f"num_peaks is {num_peaks}; must be 1 or 2")

    if slice_dim == "horizontal":
        for yval in np.linspace(ylim[0], ylim[1], steps):
            y_window = (yval - slice_window / 2, yval + slice_window / 2)
            x_1d, row = tr_functions.get_1d_y_slice(
                x=x_fit, y=y_fit, data=d_fit, xlims=xlim, y_range=y_window
            )

            if model == "Lorentzian":
                fit = fit_lorentzian_data(
                    x=x_1d,
                    data=row,
                    num_peaks=num_peaks,
                    amplitudes=amplitudes,
                    centers=center_vals,
                    sigmas=sigmas,
                    offset_type=offset_type,
                )
            elif model == "Gaussian":
                fit = fit_gaussian_data(
                    x=x_1d,
                    data=row,
                    num_peaks=num_peaks,
                    amplitudes=amplitudes,
                    centers=center_vals,
                    sigmas=sigmas,
                    include_exp_decay=include_exp_decay,
                    gammas=gammas,
                    offset_type=offset_type,
                )
            else:
                raise ValueError(
                    f"model {model} is not an option. Use Lorentzian or Gaussian."
                )
            fits.append(fit)
            coords.extend(
                [(fit.best_values[f"i{i}_center"], yval) for i in range(num_peaks)]
            )
        if get_coords:
            return fits, coords
        else:
            return fits

    elif slice_dim == "vertical":
        for xval in np.linspace(xlim[0], xlim[1], steps):
            x_window = (xval - slice_window / 2, xval + slice_window / 2)
            y_1d, col = tr_functions.get_1d_x_slice(
                x=x_fit, y=y_fit, data=d_fit, ylims=ylim, x_range=xval
            )

            if model == "Lorentzian":
                fit = fit_lorentzian_data(
                    x=y_1d,
                    data=col,
                    num_peaks=num_peaks,
                    amplitudes=amplitudes,
                    centers=center_vals,
                    sigmas=sigmas,
                    offset_type=offset_type,
                )
            elif model == "Gaussian":
                fit = fit_gaussian_data(
                    x=y_1d,
                    data=col,
                    num_peaks=num_peaks,
                    amplitudes=amplitudes,
                    centers=center_vals,
                    sigmas=sigmas,
                    include_exp_decay=include_exp_decay,
                    gammas=gammas,
                    offset_type=offset_type,
                )
            else:
                raise ValueError(
                    f"model {model} is not an option. Use Lorentzian or Gaussian."
                )
            fits.append(fit)
            coords.extend(
                [(fit.best_values[f"i{i}_center"], xval) for i in range(num_peaks)]
            )
        if get_coords:
            return fits, coords
        else:
            return fits