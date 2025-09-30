import numpy as np
from numba import njit


@njit
def get_dzdh(z, V, a):
    # change in level per unit of volume (m/m)
    dz = np.diff(z)
    # change in volume (normalized to meters)
    dh = np.maximum(np.diff(V) / a, 0.001)
    return dz / dh


@njit
def isclose(a, b, rtol=1e-05, atol=1e-08):
    return abs(a - b) <= (atol + rtol * abs(b))


@njit
def subgrid_v_table(
    elevation: np.ndarray,
    dx: float,
    dy: float,
    nlevels: int,
    zvolmin: float,
    max_gradient: float,
):
    """
    map vector of elevation values into a hypsometric volume - depth relationship
    for one grid cell

    Parameters
    ----------
    elevation: np.ndarray
        subgrid elevation values for one grid cell [m]
    dx: float
        x-directional cell size (typically not known at this level) [m]
    dy: float
        y-directional cell size (typically not known at this level) [m]
    nlevels: int
        number of levels to use for the hypsometric curve
    zvolmin: float
        minimum elevation value to use for volume calculation (typically -20 m)
    max_gradient: float
        maximum gradient to use for volume calculation

    Return
    ------
    z, V: np.ndarray
        sorted elevation values, volume per elevation value
    zmin, zmax: float
        minimum, and maximum elevation values
    """

    # Cell area
    a = float(elevation.size * dx * dy)

    # Set minimum elevation to -20 (needed with single precision), and sort
    ele_sort = np.sort(np.maximum(elevation, zvolmin).flatten())

    # Make sure each consecutive point is larger than previous
    for j in range(1, ele_sort.size):
        if ele_sort[j] <= ele_sort[j - 1]:
            ele_sort[j] += 1.0e-6

    depth = ele_sort - ele_sort.min()

    volume = np.zeros_like(depth)
    volume[1:] = np.cumsum((np.diff(depth) * dx * dy) * np.arange(1, depth.size))

    # Resample volumes to discrete levels
    steps = np.arange(nlevels) / (nlevels - 1)
    V = steps * volume.max()
    dvol = volume.max() / (nlevels - 1)
    # scipy not supported in numba jit
    # z = interpolate.interp1d(volume, ele_sort)(V)
    z = np.interp(V, volume, ele_sort)
    dzdh = get_dzdh(z, V, a)
    n = 0
    while (
        dzdh.max() > max_gradient and not (isclose(dzdh.max(), max_gradient))
    ) and n < nlevels:
        # reshape until gradient is satisfactory
        idx = np.where(dzdh == dzdh.max())[0]
        z[idx + 1] = z[idx] + max_gradient * (dvol / a)
        dzdh = get_dzdh(z, V, a)
        n += 1
    return z, V, elevation.min(), z.max()


@njit
def subgrid_q_table(
    elevation: np.ndarray,
    rgh: np.ndarray,
    nlevels: int,
    huthresh: float,
    option: int = 2,
    z_zmin_a: float = -99999.0,
    z_zmin_b: float = -99999.0,
    weight_option: str = "min",
    roughness_type: str = "manning",
):
    """
    map vector of elevation values into a hypsometric hydraulic radius - depth relationship for one u/v point
    Parameters
    ----------
    elevation : np.ndarray (nr of pixels in one cell) containing subgrid elevation values for one grid cell [m]
    rgh : np.ndarray (nr of pixels in one cell) containing subgrid roughness values for one grid cell [s m^(-1/3)]
    nlevels : int, number of vertical levels [-]
    huthresh : float, threshold depth [m]
    option : int, option to use "old" or "new" method for computing conveyance depth at u/v points
    z_zmin_a : float, elevation of lowest pixel in neighboring cell A [m]
    z_zmin_b : float, elevation of lowest pixel in neighboring cell B [m]
    weight_option : str, weight of q between sides A and B ("min" or "mean")
    roughness_type : str, "manning" or "chezy"

    Returns
    -------
    zmin : float, minimum elevation [m]
    zmax : float, maximum elevation [m]
    havg : np.ndarray (nlevels) grid-average depth for vertical levels [m]
    nrep : np.ndarray (nlevels) representative roughness for vertical levels [m1/3/s] ?
    pwet : np.ndarray (nlevels) wet fraction for vertical levels [-] ?
    navg : float, grid-average Manning's n [m 1/3 / s]
    ffit : float, fitting coefficient [-]
    zz   : np.ndarray (nlevels) elevation of vertical levels [m]
    """
    # Initialize output arrays
    havg = np.zeros(nlevels)
    nrep = np.zeros(nlevels)
    pwet = np.zeros(nlevels)
    zz = np.zeros(nlevels)

    n = int(np.size(elevation))  # Nr of pixels in grid cell
    n05 = int(n / 2)  # Nr of pixels in half grid cell

    # Sort elevation and manning values by side A and B
    dd_a = elevation[0:n05]
    dd_b = elevation[n05:]
    rgh_a = rgh[0:n05]
    rgh_b = rgh[n05:]

    # Ensure that pixels are at least as high as the minimum elevation in the neighbouring cells
    # This should always be the case, but there may be errors in the interpolation to the subgrid pixels
    dd_a = np.maximum(dd_a, z_zmin_a)
    dd_b = np.maximum(dd_b, z_zmin_b)

    # Determine min and max elevation
    zmin_a = np.min(dd_a)
    zmax_a = np.max(dd_a)
    zmin_b = np.min(dd_b)
    zmax_b = np.max(dd_b)

    # Add huthresh to zmin
    zmin = max(zmin_a, zmin_b) + huthresh
    zmax = float(max(zmax_a, zmax_b))

    # Make sure zmax is at least 0.01 m higher than zmin
    zmax = max(zmax, zmin + 0.01)

    # Determine bin size
    dlevel = (zmax - zmin) / (nlevels - 1)

    # Option can be either 1 ("old") or 2 ("new")
    # Should never use option 1 !
    option = option

    # Loop through levels
    for ibin in range(nlevels):
        # Top of bin
        zbin = zmin + ibin * dlevel
        zz[ibin] = zbin

        h = np.maximum(zbin - elevation, 0.0)  # water depth in each pixel

        # Depth of all pixels (but set min pixel height to zbot). Can be negative, but not zero (because zmin = zbot + huthresh, so there must be pixels below zb).
        h_a = np.maximum(zbin - dd_a, 0.0)
        # Depth of all pixels (but set min pixel height to zbot). Can be negative, but not zero (because zmin = zbot + huthresh, so there must be pixels below zb).
        h_b = np.maximum(zbin - dd_b, 0.0)

        if roughness_type == "manning":
            manning_a = rgh_a
            manning_b = rgh_b
            manning = rgh
        elif roughness_type == "chezy":
            manning_a = (1.0 / rgh_a) * h_a ** (1.0 / 6.0)
            manning_b = (1.0 / rgh_b) * h_b ** (1.0 / 6.0)
            manning = (1.0 / rgh) * h ** (1.0 / 6.0)
            # Set minimum value to avoid division by zero
            manning_a = np.maximum(manning_a, 0.001)
            manning_b = np.maximum(manning_b, 0.001)
            manning = np.maximum(manning, 0.001)

        # Side A
        q_a = h_a ** (5.0 / 3.0) / manning_a  # Determine 'flux' for each pixel
        q_a = np.mean(q_a)  # Grid-average flux through all the pixels
        h_a = np.mean(h_a)  # Grid-average depth through all the pixels

        # Side B
        q_b = h_b ** (5.0 / 3.0) / manning_b  # Determine 'flux' for each pixel
        q_b = np.mean(q_b)  # Grid-average flux through all the pixels
        h_b = np.mean(h_b)  # Grid-average depth through all the pixels

        # Compute q and h
        # Determine grid average 'flux' for each pixel
        q_all = np.mean(h ** (5.0 / 3.0) / manning)
        h_all = np.mean(h)  # grid averaged depth of A and B combined
        q_min = np.minimum(q_a, q_b)
        h_min = np.minimum(h_a, h_b)

        if option == 1:
            # Use old 1 option (weighted average of q_ab and q_all) option (min at bottom bin, mean at top bin)
            w = (ibin) / (
                nlevels - 1
            )  # Weight (increase from 0 to 1 from bottom to top bin)
            q = (1.0 - w) * q_min + w * q_all  # Weighted average of q_min and q_all
            hmean = h_all
            # Wet fraction
            pwet[ibin] = (zbin > elevation + huthresh).sum() / n

        elif option == 2:
            # Use newer 2 option (minimum of q_a an q_b, minimum of h_a and h_b increasing to h_all, using pwet for weighting) option
            # This is done by making sure that the wet fraction is 0.0 in the first level on the shallowest side (i.e. if ibin==0, pwet_a or pwet_b must be 0.0).
            # As a result, the weight w will be 0.0 in the first level on the shallowest side.

            pwet_a = (zbin > dd_a).sum() / int(n / 2)
            pwet_b = (zbin > dd_b).sum() / int(n / 2)

            if ibin == 0:
                # Ensure that at bottom level, either pwet_a or pwet_b is 0.0
                if pwet_a < pwet_b:
                    pwet_a = 0.0
                else:
                    pwet_b = 0.0
            elif ibin == nlevels - 1:
                # Ensure that at top level, both pwet_a and pwet_b are 1.0
                pwet_a = 1.0
                pwet_b = 1.0

            if weight_option == "mean":
                # Weight increases linearly from 0 to 1 from bottom to top bin use percentage wet in sides A and B
                w = 2 * np.minimum(pwet_a, pwet_b) / max(pwet_a + pwet_b, 1.0e-9)
                q = (1.0 - w) * q_min + w * q_all  # Weighted average of q_min and q_all
                # Weighted average of h_min and h_all
                hmean = (1.0 - w) * h_min + w * h_all

            else:
                # Take minimum of q_a and q_b
                if q_a < q_b:
                    q = q_a
                    hmean = h_a
                else:
                    q = q_b
                    hmean = h_b

            pwet[ibin] = 0.5 * (pwet_a + pwet_b)  # Combined pwet_a and pwet_b

        havg[ibin] = hmean  # conveyance depth
        nrep[ibin] = hmean ** (5.0 / 3.0) / q  # Representative n for qmean and hmean

    nrep_top = nrep[-1]
    havg_top = havg[-1]

    ### Fitting for nrep above zmax

    # Determine nfit at zfit
    zfit = float(zmax + zmax - zmin)
    # mean water depth in cell as computed in SFINCS (assuming linear relation between water level and water depth above zmax)
    hfit = havg_top + zmax - zmin
    # Compute q and navg
    if weight_option == "mean":
        # Use entire uv point
        h = np.maximum(zfit - elevation, 0.0)  # water depth in each pixel
        q = np.mean(h ** (5.0 / 3.0) / manning)  # combined unit discharge for cell
        navg = np.mean(manning)

    else:
        if roughness_type == "manning":
            manning_a = rgh_a
            manning_b = rgh_b
            manning = rgh
        elif roughness_type == "chezy":
            manning_a = (1.0 / rgh_a) * h_a ** (1.0 / 6.0)
            manning_b = (1.0 / rgh_b) * h_b ** (1.0 / 6.0)
            manning = (1.0 / rgh) * h ** (1.0 / 6.0)
            # Set minimum value to avoid division by zero
            manning_a = np.maximum(manning_a, 0.001)
            manning_b = np.maximum(manning_b, 0.001)
            manning = np.maximum(manning, 0.001)

        # Use minimum of q_a and q_b
        if q_a < q_b:
            h = np.maximum(zfit - dd_a, 0.0)  # water depth in each pixel

            if roughness_type == "manning":
                manning_a = rgh_a
            elif roughness_type == "chezy":
                manning_a = (1.0 / rgh_a) * h ** (1.0 / 6.0)
                # Set minimum value to avoid division by zero
                manning_a = np.maximum(manning_a, 0.001)

            q = np.mean(
                h ** (5.0 / 3.0) / manning_a
            )  # combined unit discharge for cell
            navg = np.mean(manning_a)
        else:
            h = np.maximum(zfit - dd_b, 0.0)
            if roughness_type == "manning":
                manning_b = rgh_b
            elif roughness_type == "chezy":
                manning_b = (1.0 / rgh_b) * h ** (1.0 / 6.0)
                # Set minimum value to avoid division by zero
                manning_b = np.maximum(manning_b, 0.001)
            q = np.mean(h ** (5.0 / 3.0) / manning_b)
            navg = np.mean(manning_b)

    nfit = hfit ** (5.0 / 3.0) / q

    # Actually apply fit on gn2 (this is what is used in sfincs)
    gnavg2 = float(9.81 * navg**2)
    gnavg_top2 = float(9.81 * nrep_top**2)

    if gnavg2 / gnavg_top2 > 0.99 and gnavg2 / gnavg_top2 < 1.01:
        # gnavg2 and gnavg_top2 are almost identical
        ffit = 0.0
    else:
        if navg > nrep_top:
            if nfit > navg:
                nfit = nrep_top + 0.9 * (navg - nrep_top)
            if nfit < nrep_top:
                nfit = nrep_top + 0.1 * (navg - nrep_top)
        else:
            if nfit < navg:
                nfit = nrep_top + 0.9 * (navg - nrep_top)
            if nfit > nrep_top:
                nfit = nrep_top + 0.1 * (navg - nrep_top)
        gnfit2 = float(9.81 * nfit**2)
        zfit = max(zfit, zmax + 1.0e-6)
        gnavg2 = max(gnavg2, gnfit2 + 1.0e-8)
        ffit = (((gnavg2 - gnavg_top2) / (gnavg2 - gnfit2)) - 1) / (zfit - zmax)

    return zmin, zmax, havg, nrep, pwet, ffit, navg, zz
