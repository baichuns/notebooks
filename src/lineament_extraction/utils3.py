
import gc
import os
import re
# import dask
import dask.array as da
# import fiona
import numpy as np
import scipy as sp
import pandas as pd
import rasterio
from itertools import compress
from dask.diagnostics import ProgressBar
# from osgeo import gdal
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree
from skimage import morphology
from shapely.geometry import LineString, mapping, Point, MultiLineString
from shapely.strtree import STRtree
from shapely.ops import linemerge
import rdp
# from skimage.measure import marching_cubes
# from sklearn.neighbors import KernelDensity
from sklearn.utils import check_X_y, check_array, column_or_1d

import matplotlib.colors as clr



def read_geotif(tif_path, band_index=1):
    tif = rasterio.open(tif_path)
    print(f"--- raster band count = {tif.count}.")

    if tif.count > 1:
        print(f"--- band count = {tif.count}.")
    if not tif._has_band(band_index):
        raise Exception(f"*** incorrect band index {band_index}, band count = {tif.count}!")
    band = tif.read(band_index)
    crs = tif.crs

    # x, y coordinates
    cols, rows = np.meshgrid(np.arange(tif.width), np.arange(tif.height))
    x, y = tif.xy(rows, cols)
    x = np.array(x)
    y = np.array(y)

    dx = np.mean(np.diff(np.unique(x.ravel())))
    dy = np.mean(np.diff(np.unique(y.ravel())))
    if abs(dx-dy)<1e-5:
        ds = dx
        if ds < 1:
            print(f" ds~={round(ds*6373, 2)}m is an average resolution!")
        else:
            print(f" ds={round(ds, 2)} is an average resolution!")
    else:
        ds = (dx+dy)*0.5
        if ds > 3:
            print(f" dx={round(dx,2)} and dy={round(dy, 2)} resolution value diffs, ds={round(ds, 2)} is an average resolution!")
        else:
            print(f" dx={round(dx*6373,2)} and dy={round(dy*6373, 2)} resolution value diffs, ds={round(ds*6373, 2)} is an average resolution!")

    tif.close()

    print(f'--- loaded! crs = {crs}')
    return band, x, y, ds, crs, tif.transform, cols, rows




# footprint type of intersection
T1 = np.array([[0, 0, 0], [1, 1, 1], [0, 1, 0]])
T2 = np.rot90(T1)
T3 = np.rot90(T2)
T4 = np.rot90(T3)
T5 = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 1]])
T6 = np.rot90(T5)
T7 = np.rot90(T6)
T8 = np.rot90(T7)
Y1 = np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]])
Y2 = np.rot90(Y1)
Y3 = np.rot90(Y2)
Y4 = np.rot90(Y3)
Y5 = np.array([[0, 1, 0], [0, 1, 1], [1, 0, 0]])
Y6 = np.rot90(Y5)
Y7 = np.rot90(Y6)
Y8 = np.rot90(Y7)

def sort_xy_array(arr):
    """
    sort coordinate array by distance

    Args:
        arr (_type_): _description_

    Returns:
        _type_: _description_
    """
    lbl_tree = cKDTree(arr)
    lbl_tree2 = cKDTree(arr)

    idx = lbl_tree.query_ball_tree(lbl_tree2, r=1.5)
    for i in np.arange(len(idx)):
        idx[i] = np.delete(idx[i], [i==j for j in idx[i]])

    m = np.where([len(id)==1 for id in idx])[0]
    if len(m) < 1:
        return arr
    m = m[0] if len(m) > 1 else m
    this_end = arr[m].ravel()

    lbl_xy = [list(this_end)]
    used_idx = [m]
    add_pixel=True
    while add_pixel:
        this_nbr = lbl_tree.query_ball_point(this_end, r=1.5)
        this_nbr = np.delete(this_nbr, [i in used_idx for i in this_nbr])
        if len(this_nbr) == 1:
            this_end = list(arr[this_nbr].ravel())
            lbl_xy.append(this_end)
            used_idx.append(this_nbr.item())
        elif len(this_nbr) > 1:
            this_nbr = lbl_tree.query_ball_point(this_end, r=1)
            this_nbr = np.delete(this_nbr, [i in used_idx for i in this_nbr])
            if len(this_nbr) == 1:
                this_end = list(arr[this_nbr].ravel())
                lbl_xy.append(this_end)
                used_idx.append(this_nbr.item())
            else:
                add_pixel=False
        else:
            add_pixel=False
    
    return np.array(lbl_xy)

def binarr2lines(binary_arr, len_threshold = 2, affine=None):
    bintx = sp.ndimage.binary_hit_or_miss(binary_arr, T1)

    for s in [T2, T3, T4, T5, T6, T7, T8, Y1, Y2, Y3, Y4, Y5, Y6, Y7, Y8]:
        bintx |= sp.ndimage.binary_hit_or_miss(binary_arr, s)

    introws, intcols = np.where(bintx)
    intxn_xy = np.c_[introws.ravel(), intcols.ravel()] # intersection points
    intx_idx_tree = cKDTree(intxn_xy)
    bintx_dila = morphology.dilation(bintx, footprint=sp.ndimage.generate_binary_structure(2,2))

    disjoint_lines = binary_arr & np.invert(bintx_dila)

    # label lines
    s = sp.ndimage.generate_binary_structure(2,2)
    label_obj, nb_labels = sp.ndimage.label(disjoint_lines, structure=s)

    # construct lines
    all_lines = []
    for lbl in np.arange(nb_labels)+1:
        lbl_rows, lbl_cols = np.where(label_obj == lbl)
        lbl_row_col = np.c_[lbl_rows.ravel(), lbl_cols.ravel()]

        # sort array
        lbl_row_col = sort_xy_array(lbl_row_col)

        # check and append ends distance to intersections
        itx0 = intx_idx_tree.query_ball_point(lbl_row_col[0], 4.4)
        if len(itx0) == 1:
            lbl_row_col = np.append(intxn_xy[itx0], lbl_row_col, axis=0)
        itx1 = intx_idx_tree.query_ball_point(lbl_row_col[-1], 4.4)
        if len(itx1) == 1:
            lbl_row_col = np.append(lbl_row_col, intxn_xy[itx1], axis=0)

        if len(lbl_row_col) > len_threshold:
            lbl_row_col[:, [0, 1]] = lbl_row_col[:, [1, 0]]
            # this_line = LineString(lbl_row_col)
            # all_lines.append(this_line)
            if affine is not None:
                all_lines.append(LineString([affine*i for i in rdp.rdp(lbl_row_col)]))
            else:
                all_lines.append(LineString(rdp.rdp(lbl_row_col)))


    return all_lines, intxn_xy


def calc_linestring_azimuth(this_linestring):
    """[summary]

    Args:
        this_linestring (shapely geometry): [description]

    Returns:
        [float]: azimuth between -90 to +90 degree
    """
    this_lines = np.array([[this_linestring.coords[i+1][0]-this_linestring.coords[i][0], this_linestring.coords[i+1][1]-this_linestring.coords[i][1]] \
        for i in range(len(this_linestring.coords) - 1)])
    # plot for debugging
    # fig, ax=plt.subplots(1); ax.plot(this_linestring.xy[0].tolist(), this_linestring.xy[1].tolist()); ax.set_aspect('equal')

    this_azi = 90 - np.arctan2(this_lines[:,1], this_lines[:,0])*57.29577951308232

    this_lengths = [LineString([this_linestring.coords[i], this_linestring.coords[i+1]]).length/this_linestring.length for i in range(len(this_linestring.coords) - 1)]
    azi = sum(this_azi*this_lengths)
    # limit the azimuth range
    if azi > 90 and azi <= 270:
        azi -= 180
    elif azi < -90 and azi >= -180:
        azi += 180

    return azi

def filter_line_by_curvature(lines):
    dm = np.median([i.length for i in lines])
    is_straight = []
    for this_line in lines:
        if isinstance(this_line, LineString):
            dv = np.linalg.norm(np.array(this_line.coords[0])-np.array(this_line.coords[-1]))
            ds = this_line.length

            rto = 0.6 if ds>dm else 0.7
            if dv/ds < rto:
                is_straight.append(False)
            else:
                is_straight.append(True)
    return list(compress(lines, is_straight))


def merge_connected_lines(lines, intxn_pts_list, eps=0, length_to_merge_threshold = None):
    assert isinstance(lines, list), "lines should be a list type!"
    idx_to_del = []
    new_lines =[]
    if length_to_merge_threshold is None:
        all_length = [i.length for i in lines]
        length_to_merge_threshold = np.median(all_length)


    lines_tree = STRtree(lines)
    for this_pt in intxn_pts_list:
        this_intxn = Point(this_pt)
        other_lines = lines_tree.query(this_intxn.buffer(3))
        # other_lines_idx = np.where([i.intersects(this_intxn.buffer(0.2)) for i in lines])[0]

        if len(other_lines) == 2 and other_lines[0].intersects(other_lines[1]) and \
            (other_lines[0].length<length_to_merge_threshold or other_lines[1].length<length_to_merge_threshold):
            # new_lines.append(linemerge(MultiLineString([lines[other_lines_idx[0]], lines[other_lines_idx[1]]])))
            new_lines.append(linemerge(MultiLineString(other_lines)))
            n1 = np.where([other_lines[0]==i for i in lines])[0].item()
            idx_to_del.append(n1)
            # del lines[n1]
            n2 = np.where([other_lines[1]==i for i in lines])[0].item()
            idx_to_del.append(n2)
            # del lines[n2]
        elif len(other_lines) == 3:
            azis = [calc_linestring_azimuth(i) for i in other_lines]
            a12 = abs(azis[1]-azis[0])
            a23 = abs(azis[1]-azis[2])
            a13 = abs(azis[2]-azis[0])
            a_min = np.min([a12, a23, a13])
            a_idx = []
            if a12 == a_min and a12<45 and other_lines[0].intersects(other_lines[1]):
                a_idx=[0, 1]
            elif a23 == a_min and a23<45 and other_lines[2].intersects(other_lines[1]):
                a_idx=[1, 2]
            elif a13 == a_min and a13<45 and other_lines[2].intersects(other_lines[0]):
                a_idx=[0, 2]
            if len(a_idx)==2:
                new_lines.append(linemerge(MultiLineString([other_lines[a_idx[0]], other_lines[a_idx[1]]])))
                n1 = np.where([other_lines[a_idx[0]]==i for i in lines])[0].item()
                idx_to_del.append(n1)
                n2 = np.where([other_lines[a_idx[1]]==i for i in lines])[0].item()
                idx_to_del.append(n2)

    # clean lines list
    for i, this_line in enumerate(lines):
        if i not in idx_to_del:
            new_lines.append(this_line)
    
    if eps > 0:
        for i in range(len(new_lines)):
            new_lines[i] = LineString(rdp.rdp([(x, y) for x, y in new_lines[i].coords], epsilon=eps))

    return new_lines



class PrepareImage():

    def __init__(self):
        self.img = None

    def fit(self, imgarr):

        self.img = check_array(imgarr)


    def normalize(self):
        """ normalize as pre-processing for PCA. flat z scores (i.e. mean = 0, std = 1, shape = flat)"""
        
        X = []
        for band in self.img:
            # z-score = mean of zero and standard deviation of 1
            zscore = (band - band.mean())/band.std()
            
            # flatten the bands into 1D arrays (required by np.cov)
            X.append(zscore.flatten())
            
        return np.array(X)

    def principal_components(self):
        """principal components of input array"""

        # flat z-scores
        X = self.normalize()
        
        # covariance matrix
        cov = np.cov(X)

        # eigenvalues and vectors of covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # sort high to low
        sort_order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort_order]
        eigenvectors = eigenvectors[sort_order]

        # Linear transformation of X coordinate system into PC coordinate system
        PC = np.matmul(X.transpose(),eigenvectors.transpose()).transpose()
        
        return (PC, eigenvalues, eigenvectors)




def weighted_average(
    xyz_in: np.ndarray,
    xyz_out: np.ndarray,
    values: list,
    max_distance: float = np.inf,
    n: int = 8,
    return_indices: bool = False,
    threshold: float = 1e-1,
) -> list:
    """
    Perform a inverse distance weighted averaging on a list of values.

    :param xyz_in: shape(*, 3) Input coordinate locations.
    :param xyz_out: shape(*, 3) Output coordinate locations.
    :param values: Values to be averaged from the input to output locations.
    :param max_distance: Maximum averaging distance, beyond which values are assigned nan.
    :param n: Number of nearest neighbours used in the weighted average.
    :param return_indices: If True, return the indices of the nearest neighbours from the input locations.
    :param threshold: Small value added to the radial distance to avoid zero division.
        The value can also be used to smooth the interpolation.

    :return avg_values: List of values averaged to the output coordinates
    """
    assert isinstance(values, list), "Input 'values' must be a list of numpy.ndarrays"

    assert all(
        [vals.shape[0] == xyz_in.shape[0] for vals in values]
    ), "Input 'values' must have the same shape as input 'locations'"

    tree = cKDTree(xyz_in)
    rad, ind = tree.query(xyz_out, n)
    rad[rad > max_distance] = np.nan
    avg_values = []
    for value in values:
        values_interp = np.zeros(xyz_out.shape[0])
        weight = np.zeros(xyz_out.shape[0])

        for ii in range(n):
            values_interp += value[ind[:, ii]] / (rad[:, ii] + threshold)
            weight += 1.0 / (rad[:, ii] + threshold)

        avg_values += [values_interp / weight]

    if return_indices:
        return avg_values, ind

    return avg_values


def filter_xy(
    x: np.array, y: np.array, distance: float, window: dict = None
) -> np.array:
    """
    Function to extract and down-sample xy locations based on minimum distance and window parameters.

    :param x: Easting coordinates
    :param y: Northing coordinates
    :param distance: Minimum distance between neighbours
    :param window: Window parameters describing a domain of interest.
        Must contain the following keys and values:
        window = {
            "center": [X: float, Y: float],
            "size": [width: float, height: float],
            "azimuth": float
        }

    :return selection: Array of 'bool' of shape(x)
    """
    mask = np.ones_like(x, dtype="bool")
    if window is not None:
        x_lim = [
            window["center"][0] - window["size"][0] / 2,
            window["center"][0] + window["size"][0] / 2,
        ]
        y_lim = [
            window["center"][1] - window["size"][1] / 2,
            window["center"][1] + window["size"][1] / 2,
        ]
        xy_rot = rotate_xy(
            np.c_[x.ravel(), y.ravel()], window["center"], window["azimuth"]
        )
        mask = (
            (xy_rot[:, 0] > x_lim[0])
            * (xy_rot[:, 0] < x_lim[1])
            * (xy_rot[:, 1] > y_lim[0])
            * (xy_rot[:, 1] < y_lim[1])
        ).reshape(x.shape)

    if x.ndim == 1:
        filter_xy = np.ones_like(x, dtype="bool")
        if distance > 0:
            mask_ind = np.where(mask)[0]
            xy = np.c_[x[mask], y[mask]]
            tree = cKDTree(xy)

            nstn = xy.shape[0]
            # Initialize the filter
            for ii in range(nstn):
                if filter_xy[mask_ind[ii]]:
                    ind = tree.query_ball_point(xy[ii, :2], distance)
                    filter_xy[mask_ind[ind]] = False
                    filter_xy[mask_ind[ii]] = True

    elif distance > 0:
        filter_xy = np.zeros_like(x, dtype="bool")
        d_l = np.max(
            [
                np.linalg.norm(np.c_[x[0, 0] - x[0, 1], y[0, 0] - y[0, 1]]),
                np.linalg.norm(np.c_[x[0, 0] - x[1, 0], y[0, 0] - y[1, 0]]),
            ]
        )
        dwn = int(np.ceil(distance / d_l))
        filter_xy[::dwn, ::dwn] = True
    else:
        filter_xy = np.ones_like(x, dtype="bool")
    return filter_xy * mask


def rotate_xy(xyz: np.ndarray, center: list, angle: float):
    """
    Perform a counterclockwise rotation on the XY plane about a center point.

    :param xyz: shape(*, 3) Input coordinates
    :param center: len(2) Coordinates for the center of rotation.
    :param  angle: Angle of rotation in degree
    """
    R = np.r_[
        np.c_[np.cos(np.pi * angle / 180), -np.sin(np.pi * angle / 180)],
        np.c_[np.sin(np.pi * angle / 180), np.cos(np.pi * angle / 180)],
    ]

    locs = xyz.copy()
    locs[:, 0] -= center[0]
    locs[:, 1] -= center[1]

    xy_rot = np.dot(R, locs[:, :2].T).T

    return np.c_[xy_rot[:, 0] + center[0], xy_rot[:, 1] + center[1], locs[:, 2:]]


def running_mean(
    values: np.array, width: int = 1, method: str = "centered"
) -> np.array:
    """
    Compute a running mean of an array over a defined width.

    :param values: Input values to compute the running mean over
    :param width: Number of neighboring values to be used
    :param method: Choice between 'forward', 'backward' and ['centered'] averaging.

    :return mean_values: Averaged array values of shape(values, )
    """
    # Averaging vector (1/N)
    weights = np.r_[np.zeros(width + 1), np.ones_like(values)]
    sum_weights = np.cumsum(weights)

    mean = np.zeros_like(values)

    # Forward averaging
    if method in ["centered", "forward"]:
        padd = np.r_[np.zeros(width + 1), values]
        cumsum = np.cumsum(padd)
        mean += (cumsum[(width + 1) :] - cumsum[: (-width - 1)]) / (
            sum_weights[(width + 1) :] - sum_weights[: (-width - 1)]
        )

    # Backward averaging
    if method in ["centered", "backward"]:
        padd = np.r_[np.zeros(width + 1), values[::-1]]
        cumsum = np.cumsum(padd)
        mean += (
            (cumsum[(width + 1) :] - cumsum[: (-width - 1)])
            / (sum_weights[(width + 1) :] - sum_weights[: (-width - 1)])
        )[::-1]

    if method == "centered":
        mean /= 2.0

    return mean




def string_2_list(string):
    """
    Convert a list of numbers separated by comma to a list of floats
    """
    return [float(val) for val in string.split(",") if len(val) > 0]



def hex_to_rgb(hex):
    """
    Convert hex color code to RGB
    """
    code = hex.lstrip("#")
    return [int(code[i : i + 2], 16) for i in (0, 2, 4)]


def symlog(values, threshold):
    """
    Convert values to log with linear threshold near zero
    """
    return np.sign(values) * np.log10(1 + np.abs(values) / threshold)


def inv_symlog(values, threshold):
    """
    Compute the inverse symlog mapping
    """
    return np.sign(values) * threshold * (-1.0 + 10.0 ** np.abs(values))




def ij_2_ind(coordinates, shape):
    """
    Return the index of ij coordinates
    """
    return [ij[0] * shape[1] + ij[1] for ij in coordinates]


def ind_2_ij(indices, shape):
    """
    Return the index of ij coordinates
    """
    return [[int(np.floor(ind / shape[1])), ind % shape[1]] for ind in indices]


def get_neighbours(index, shape):
    """
    Get all neighbours of cell in a 2D grid
    """
    j, i = int(np.floor(index / shape[1])), index % shape[1]
    vec_i = np.r_[i - 1, i, i + 1]
    vec_j = np.r_[j - 1, j, j + 1]

    vec_i = vec_i[(vec_i >= 0) * (vec_i < shape[1])]
    vec_j = vec_j[(vec_j >= 0) * (vec_j < shape[0])]

    ii, jj = np.meshgrid(vec_i, vec_j)

    return ij_2_ind(np.c_[jj.ravel(), ii.ravel()].tolist(), shape)


def get_active_neighbors(index, shape, model, threshold, blob_indices):
    """
    Given an index, append to a list if active
    """
    out = []
    for ind in get_neighbours(index, shape):
        if (model[ind] > threshold) and (ind not in blob_indices):
            out.append(ind)
    return out


def get_blob_indices(index, shape, model, threshold, blob_indices=[]):
    """
    Function to return indices of cells inside a model value blob
    """
    out = get_active_neighbors(index, shape, model, threshold, blob_indices)

    for neigh in out:
        blob_indices += [neigh]
        blob_indices = get_blob_indices(
            neigh, shape, model, threshold, blob_indices=blob_indices
        )

    return blob_indices



def input_string_2_float(input_string):
    """
    Function to input interval and value as string to a list of floats.

    Parameter
    ---------
    input_string: str
        Input string value of type `val1:val2:ii` and/or a list of values `val3, val4`


    Return
    ------
    list of floats
        Corresponding list of values in float format

    """
    if input_string != "":
        vals = re.split(",", input_string)
        cntrs = []
        for val in vals:
            if ":" in val:
                param = np.asarray(re.split(":", val), dtype="float")
                if len(param) == 2:
                    cntrs += [np.arange(param[0], param[1] + 1)]
                else:
                    cntrs += [np.arange(param[0], param[1] + param[2], param[2])]
            else:
                cntrs += [float(val)]
        return np.unique(np.sort(np.hstack(cntrs)))

    return None


# Function to make MatplotLib colormap
def gen_cmap(name, array, start, end):
    b3 = array[:,2] # value of blue at sample n
    b2 = array[:,2] # value of blue at sample n
    b1 = np.linspace(start, end, len(b2)) # position of sample n - ranges from 0 to 1
    
    # Setting up columns for tuples
    g3 = array[:,1]
    g2 = array[:,1]
    g1 = np.linspace(start, end, len(g2))
    
    r3 = array[:,0]
    r2 = array[:,0]
    r1 = np.linspace(start, end, len(r2))
    
    # Creating tuples
    R = sorted(zip(r1,r2,r3))
    G = sorted(zip(g1,g2,g3))
    B = sorted(zip(b1,b2,b3))
    
    # Transposing
    RGB = zip(R,G,B)
    rgb = zip(*RGB)
    
    # Creating dictionary
    k = ['red', 'green', 'blue']
    Cube1 = dict(zip(k,rgb))
    
    return clr.LinearSegmentedColormap(name, Cube1)
    
    
    
    
    
    
def join_two_lines_by_closest_ends(line1, line2):
    ends = [line1.boundary[0], line1.boundary[-1], line2.boundary[0], line2.boundary[-1]]
    pts = [[geom.xy[0][0], geom.xy[1][0]] for geom in ends]
    m = [(pts[0], pts[2]), (pts[0], pts[3]), (pts[1], pts[2]), (pts[1], pts[3])]
    dists = [math.dist(i[0], i[1]) for i in m]
    jline = LineString(m[np.argmin(dists)])
    return linemerge([line1, line2, jline])
    
    
    
    
    
