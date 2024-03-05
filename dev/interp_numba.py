import warnings

import numpy as np
import numba
from numba.typed import List
import pymp


def interp(
    ifg: np.ndarray,
    weights: np.ndarray,
    num_neighbors: int,
    max_radius: int,
    min_radius: int = 0,
    alpha: float = 0.75,
    weight_cutoff: float = 0.0,
    n_workers: int = 5,
):
    """Persistent scatterer interpolation.

    Parameters
    ----------
    ifg : np.ndarray, 2D complex array
        wrapped interferogram to interpolate
    weights : 2D boolean array
        Array of weights from 0 to 1 indicating how strongly to weigh
        the ifg values when interpolating.
        A special case of this is a PS mask where
            weights[i,j] = True if radar pixel (i,j) is a PS
            weights[i,j] = False if radar pixel (i,j) is not a PS
        Can also pass a coherence image to use as weights.
    num_neighbors: int (optional)
        number of nearest PS pixels used for interpolation
        num_neighbors = 20 by default
    max_radius : int (optional)
        maximum radius (in pixel) for PS searching
        max_radius = 51 by default
    alpha : float (optional)
        hyperparameter controlling the weight of PS in interpolation: smaller
        alpha means more weight is assigned to PS closer to the center pixel.
        alpha = 0.75 by default
    weight_cutoff: float
        Threshold to use on `weights` so that pixels where
        `weight[i, j] < weight_cutoff` have phase values replaced by
        an interpolated value.
        If `weight_cutoff = 0` (default),  All pixels are replaced with a
        smoothed version of the surrounding pixels.
        If `weight_cutoff = 1`, only pixels with exactly weight=1
        are kept, and the rest are replaced with an interpolated value.

    Returns
    -------
    interpolated_ifg : 2D complex array
        interpolated interferogram with the same amplitude, but different
        wrapped phase at non-ps pixels.

    References
    ----------
    "A persistent scatterer interpolation for retrieving accurate ground
    deformation over InSAR‐decorrelated agricultural fields"
    Chen et al., 2015, https://doi.org/10.1002/2015GL065031
    """

    nrow, ncol = weights.shape

    if np.all(
        np.logical_or(
            np.logical_or(weights == 0, weights == 1),
            np.logical_or(weights == True, weights == False),
        )
    ):
        print("Binary weights, using PS-like interpolation.")
    else:
        print("Range of values as weights, using weight_cutoff =", weight_cutoff)

    # Ensure weights are between 0 and 1
    if np.any(weights.astype(np.float32) > 1):
        warnings.warn("weights array has values greater than 1. Clipping to 1.")
    if np.any(weights.astype(np.float32) < 0):
        warnings.warn("weights array has negative values. Clipping to 0.")
    # Make shared versions of the input arrays to avoid copying in each thread
    weights_shared = pymp.shared.array(weights.shape, dtype=np.float32)
    weights_shared[:] = np.clip(weights.astype(np.float32), 0, 1)

    # Make shared versions of the input arrays to avoid copying in each thread
    ifg_shared = pymp.shared.array(ifg.shape, dtype=np.complex64)
    ifg_shared[:] = ifg[:]

    # Make shared output array
    interpolated_ifg = pymp.shared.array((nrow, ncol), dtype=np.complex64)

    indices = np.array(_get_circle_idxs(max_radius, min_radius=min_radius))
    indices_arr = pymp.shared.array(indices.shape, dtype=indices.dtype)
    indices_arr[:] = indices

    with pymp.Parallel(n_workers) as p:
        for idx in p.range(nrow * ncol):
            # convert linear idx to row, col
            r0, c0 = np.unravel_index(idx, (nrow, ncol))
            _interp_inner_loop(
                ifg_shared,
                weights_shared,
                weight_cutoff,
                num_neighbors,
                alpha,
                indices_arr,
                r0,
                c0,
                interpolated_ifg,
            )
    return interpolated_ifg


@numba.njit
def _interp_inner_loop(
    ifg, weights, weight_cutoff, num_neighbors, alpha, indices, r0, c0, interpolated_ifg
):
    if weights[r0, c0] >= weight_cutoff:
        interpolated_ifg[r0, c0] = ifg[r0, c0]
        return

    nrow, ncol = weights.shape
    nindices = len(indices)
    counter = 0
    csum = 0.0 + 0j
    r2 = np.zeros(num_neighbors, dtype=np.float64)
    cphase = np.zeros(num_neighbors, dtype=np.complex128)

    for i in range(nindices):
        idx = indices[i]
        r = r0 + idx[0]
        c = c0 + idx[1]

        if (
            (r >= 0)
            and (r < nrow)
            and (c >= 0)
            and (c < ncol)
            and weights[r, c] >= weight_cutoff
        ):
            # calculate the square distance to the center pixel
            r2[counter] = idx[0] ** 2 + idx[1] ** 2

            cphase[counter] = np.exp(1j * np.angle(ifg[r, c]))
            counter += 1
            if counter >= num_neighbors:
                break

    # TODO : why use the "counter - 1" here to normalize?
    r2_norm = (r2[counter - 1] ** alpha) / 2
    for i in range(counter):
        csum += np.exp(-r2[i] / r2_norm) * cphase[i]

    interpolated_ifg[r0, c0] = np.abs(ifg[r0, c0]) * np.exp(1j * np.angle(csum))


@numba.njit
def _get_circle_idxs(max_radius: int, min_radius: int = 0) -> np.ndarray:
    # using the mid-point cirlce drawing algorithm to search for neighboring PS pixels
    # # code adapated from "https://www.geeksforgeeks.org/mid-point-circle-drawing-algorithm/"
    visited = np.zeros((max_radius, max_radius), dtype=numba.bool_)
    visited[0][0] = True

    indices = List()
    for r in range(1, max_radius):
        x = r
        y = 0
        p = 1 - r
        if r > min_radius:
            indices.append([r, 0])
            indices.append([-r, 0])
            indices.append([0, r])
            indices.append([0, -r])

        visited[r][0] = True
        visited[0][r] = True
        # flag > 0 means there are holes between concentric circles
        flag = 0
        while x > y:
            # do not need to fill holes
            if flag == 0:
                y += 1
                if p <= 0:
                    # Mid-point is inside or on the perimeter
                    p += 2 * y + 1
                else:
                    # Mid-point is outside the perimeter
                    x -= 1
                    p += 2 * y - 2 * x + 1

            else:
                flag -= 1

            # All the perimeter points have already been visited
            if x < y:
                break

            while not visited[x - 1][y]:
                x -= 1
                flag += 1

            visited[x][y] = True
            visited[y][x] = True
            if r > min_radius:
                indices.append([x, y])
                indices.append([-x, -y])
                indices.append([x, -y])
                indices.append([-x, y])

                if x != y:
                    indices.append([y, x])
                    indices.append([-y, -x])
                    indices.append([y, -x])
                    indices.append([-y, x])

            if flag > 0:
                x += 1

    return indices
