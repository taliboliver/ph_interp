import numpy as np

def chen_interp(arr, radius, fill_value=0.0):
    '''
    Interpolate pixels based on distance from valid pixels
    Based on Chen, J., Zebker, H. A., & Knight, R. (2015). 
    "A persistent scatterer interpolation for retrieving 
    accurate ground deformation over InSAR-decorrelated 
    agricultural fields." Geophysical Research Letters, 
    42(21), 9294–9301. https://doi.org/10.1002/2015GL065031
    Parameters
    ----------
    arr: numpy.ndarray
        Array containing invalid pixel locations to fill
    radius: int
        Radius of the sampling/filling window
    fill_value: int or np.nan
        Value used to fill outlier locations
    Returns
    -------
    arr_filt: numpy.ndarray
        Array with interpolated values at invalid pixel locations
    '''
    arr_filt = np.copy(arr)
    # Get center locations
    x_cent, y_cent = np.where(arr == fill_value)

    for xc, yc in zip(x_cent, y_cent):
        # Find the coordinates of all nonzero pixels
        x, y = np.where(arr_filt != 0)
        # Compute distance between center pixel and valid pixels
        ps_dist = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)
        # Compute weights based on distance and selected radius
        w = np.exp(-ps_dist ** 2 / 2 * radius)

        # Compute Eq. 2 of Chen's paper
        weighted_arr = arr_filt[x, y].flatten() * w
        interp_pix = np.nansum(weighted_arr) / np.nansum(w)
        if np.isnan(interp_pix):
            interp_pix = 0.0
        arr_filt[xc, yc] = interp_pix

    return arr_filt

def chen_interp_v2(arr, radius, fill_value=0.0):
    '''
    Modified after Liangs suggestion
    Interpolate pixels based on distance from valid pixels
    Based on Chen, J., Zebker, H. A., & Knight, R. (2015). 
    "A persistent scatterer interpolation for retrieving 
    accurate ground deformation over InSAR-decorrelated 
    agricultural fields." Geophysical Research Letters, 
    42(21), 9294–9301. https://doi.org/10.1002/2015GL065031
    Parameters
    ----------
    arr: numpy.ndarray
        Array containing invalid pixel locations to fill
    radius: int
        Radius of the sampling/filling window
    fill_value: int or np.nan
        Value used to fill outlier locations
    Returns
    -------
    arr_filt: numpy.ndarray
        Array with interpolated values at invalid pixel locations
    '''
    arr_filt = np.copy(arr)
    # Get center locations
    x_cent, y_cent = np.where(arr == fill_value)
    x, y = np.where(arr_filt != 0)
    ps_dist = np.sqrt((x[..., np.newaxis] - x_cent) ** 2 + (y[..., np.newaxis] - y_cent) ** 2)
    w = np.exp(-ps_dist ** 2 / 2 * radius)
    weighted_arr = arr_filt[x, y][..., np.newaxis] * w
    interp_pix = np.nansum(weighted_arr, axis=0) / np.nansum(w, axis=0) 
    interp_pix[np.isnan(interp_pix)] = 0.0
    arr_filt[x_cent, y_cent] = interp_pix

    return arr_filt
