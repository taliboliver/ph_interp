#!/usr/bin/env python

"""
Copyright (c) 2023-, California Institute of Technology ("Caltech"). U.S.

Government sponsorship acknowledged.
All rights reserved.
Author (s): Emre Havazli
"""

import argparse
import glob
import os
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal

EXAMPLE = '''
interpolate_phase.py <path_to_ifg_folder> <coherence_threshold>
<interpolation_method>
'''


def create_parser():
    """Create parser for command line arguments."""
    parser = argparse.ArgumentParser(
        description='Interpolate phase',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=EXAMPLE)

    parser.add_argument(
        'folder', type=str, help='Interferogram folder path')

    parser.add_argument(
        'coh_thresh', type=float, help='Coherence threshold to select '
        'reliable pixels')

    parser.add_argument(
        'interp_method', type=str, help='Interpolation method: chen, chen_v2, numba')

    return parser


def get_intf(intf_file):
    """Get interferogram and coherence from files."""
    f = intf_file.split('/')
    f[-1] = 'interp_filt_fine.int'
    interp_file = os.path.join('/', *f)
    shutil.copy(intf_file, interp_file)
    shutil.copy(intf_file + '.xml', interp_file + '.xml')
    ds = gdal.Open(interp_file, gdal.GA_ReadOnly)
    print('Raster size=', ds.RasterXSize, ds.RasterYSize)
    w_int = ds.GetRasterBand(1).ReadAsArray()
    ds = None
    print('interferogram dtype', w_int.dtype)

    c = intf_file.split('/')
    c[-1] = 'filt_fine.cor'
    coh_file = os.path.join('/', *c)
    ds = gdal.Open(coh_file, gdal.GA_ReadOnly)
    coh = ds.GetRasterBand(1).ReadAsArray()
    ds = None

    # Plot Igram overview
    print('Plotting interferogram overview...')
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1, 2, 1)
    cax = ax.imshow(np.angle(w_int), vmin=-3, vmax=3, cmap='hsv',
                    interpolation='nearest')
    ax.set_title(r'ifg phase')
    cbar = fig.colorbar(cax, orientation='horizontal')
    ax = fig.add_subplot(1, 2, 2)
    cax = ax.imshow(coh, cmap='jet', interpolation='nearest')
    ax.set_title(r'coh')
    cbar = fig.colorbar(cax, orientation='horizontal')
    plt.savefig('intf_overview.png', dpi=300, facecolor='white')
    plt.close()

    return interp_file, w_int, coh


def get_ps_pixels(w_int, coh, coh_thresh):
    """Define reliable pixels."""
    ps_msk = (coh > coh_thresh).astype('bool')

    # Get the coordinates of the pixels to be interpolated and generate
    # a circular sample to interpolate from.
    centers = np.where(ps_msk == 0)
    cents_num = len(centers[0])
    print('A total of ', cents_num, 'pixels require interpolation')

    # Plot PS pixels overview
    print('Plotting PS pixels overview...')
    fig = plt.figure(figsize=(7, 5))  # New figure
    ax = fig.add_subplot(1, 2, 1)  # subplot
    cax = ax.imshow(ps_msk, cmap='jet', interpolation='nearest')
    ax.set_title(r'ps_mask')
    cbar = fig.colorbar(cax, orientation='horizontal')
    ax = fig.add_subplot(1, 2, 2)  # subplot
    cax = ax.imshow(np.ma.masked_where(np.angle(w_int) * ps_msk == 0,
                                       np.angle(w_int) * ps_msk),
                    cmap='hsv', interpolation='nearest')
    ax.set_title(r'ps_pixels')
    cbar = fig.colorbar(cax, orientation='horizontal')
    plt.savefig('ps_pixels_overview.png', dpi=300, facecolor='white')
    plt.close()

    return ps_msk


def run_chen_interp(w_int, ps_msk):
    """Run interpolation for test array."""
    from dev import phase_interp as phint

    t_all = time.time()  # track processing time

    # Run interpolation here !!!
    # Modify radius to increase resampling valid region.
    ifg_re = np.copy(np.angle(w_int) * ps_msk)  # array to be resampled
    # run interpolation
    ifg_re = phint.chen_interp(ifg_re, radius=20, fill_value=0.0)

    # Track processing time
    t_all_elapsed = time.time() - t_all  # track processing time
    hours, rem = divmod(t_all_elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Successfully interpolated phase in {hours:0>2}:{minutes:0>2}:"
          f"{seconds:05.2f} hours:min:secs")  # track processing time

    return ifg_re


def run_chen_interp_v2(w_int, ps_msk):
    """Run interpolation for test array."""
    from dev import phase_interp as phint

    t_all = time.time()  # track processing time

    # Run interpolation here !!!
    # Modify radius to increase resampling valid region.
    ifg_re = np.copy(np.angle(w_int) * ps_msk)  # array to be resampled
    # run interpolation
    ifg_re = phint.chen_interp_v2(ifg_re, radius=20, fill_value=0.0)

    # Track processing time
    t_all_elapsed = time.time() - t_all  # track processing time
    hours, rem = divmod(t_all_elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Successfully interpolated phase in {hours:0>2}:{minutes:0>2}:"
          f"{seconds:05.2f} hours:min:secs")  # track processing time

    return ifg_re


def run_interp_numba(w_int, ps_msk):
    """Run interpolation for test array."""
    print('Running interpolation with "interp_numba"')
    from dev import interp_numba

    t_all = time.time()  # track processing time

    # run interpolation
    ifg_re = interp_numba.interp(w_int, ps_msk,
                                 20, 20, 0, 0.75, 5)

    # Track processing time
    t_all_elapsed = time.time() - t_all  # track processing time
    hours, rem = divmod(t_all_elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Successfully interpolated phase in {hours:0>2}:{minutes:0>2}:"
          f"{seconds:05.2f} hours:min:secs")  # track processing time

    return ifg_re


def make_interp_plots(w_int, ps_msk, ifg_re):
    """Plot interpolation overview."""
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 3, 1)
    cax = ax.imshow(np.angle(w_int), vmin=-3, vmax=3, cmap='hsv',
                    interpolation='nearest')
    ax.set_title(r'ifg orig')
    cbar = fig.colorbar(cax, orientation='horizontal')
    ax = fig.add_subplot(1, 3, 2)
    cax = ax.imshow(ps_msk, cmap='jet', interpolation='nearest')
    ax.set_title(r'Ps mask')
    cbar = fig.colorbar(cax, orientation='horizontal')
    ax = fig.add_subplot(1, 3, 3)
    cax = ax.imshow(np.angle(ifg_re), vmin=-3, vmax=3, cmap='hsv',
                    interpolation='nearest')
    ax.set_title(r'interpolated phase')
    cbar = fig.colorbar(cax, orientation='horizontal')
    plt.savefig('interpolation_overview.png', dpi=300, facecolor='white')
    plt.close()


def interpolate_ifg(args, ifg_file):
    """Interpolate a single interferogram."""
    interp_file, w_int, coh = get_intf(ifg_file)
    ps_msk = get_ps_pixels(w_int, coh, args.coh_thresh)

    if args.interp_method == 'chen':
        ifg_re = run_chen_interp(w_int, ps_msk)
    elif args.interp_method == 'chen_v2':
        ifg_re = run_chen_interp_v2(w_int, ps_msk)
    elif args.interp_method == 'numba':
        ifg_re = run_interp_numba(w_int, ps_msk)
    else:
        raise ValueError(f'Unknown interpolation method: {args.interp_method}')

    ds = gdal.Open(interp_file, gdal.GA_Update)
    ds.GetRasterBand(1).WriteArray(ifg_re)
    ds.FlushCache()
    ds = None

    print(f'Wrote: {interp_file}')


def main():
    """Run the program."""
    parser = create_parser()
    args = parser.parse_args()
    args.folder = os.path.abspath(args.folder)
    file_list = glob.glob(os.path.join(args.folder, '*/filt_fine.int'))
    print('Found the files below in the given folder:')
    print(*file_list, sep='\n')

    if args.interp_method not in ['chen', 'chen_v2', 'numba']:
        raise KeyError('The interpolation methods are "chen", "chen_v2" '
                       'and "numba"')

    for ifg_file in file_list:
        print('\n')
        print(f'Interpolating: {ifg_file}')
        interpolate_ifg(args, ifg_file)


if __name__ == '__main__':
    main()
