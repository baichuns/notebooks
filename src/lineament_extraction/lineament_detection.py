
from cgitb import small
from matplotlib.ft2font import HORIZONTAL
import numpy as np
import scipy as sp
from sklearn.utils import check_X_y, check_array, column_or_1d
import rasterio
import rasterio.features
import rasterio.warp
from rasterio.plot import show
import geopandas as gpd
import rdp
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point, shape, Polygon, MultiLineString
import matplotlib.pyplot as plt
from matplotlib import collections
from skimage.feature import canny
from skimage import filters, morphology
from skimage.transform import probabilistic_hough_line
from itertools import compress
# from centerline.geometry import Centerline
from .utils3 import *
from .potential_field_tool import *


def make_tiles(d2, tile_size):
    d2 = check_array(d2, ensure_2d=True)
    shp = d2.shape
    # Cycle through tiles of square size
    max_l = np.min([tile_size, shp[0], shp[1]])
    half = np.floor(max_l / 2)
    overlap = 1.25

    n_cell_y = (shp[0] - 2 * half) * overlap / max_l
    n_cell_x = (shp[1] - 2 * half) * overlap / max_l

    if n_cell_x > 0:
        cnt_x = np.linspace(
            half, shp[1] - half, 2 + int(np.round(n_cell_x)), dtype=int
        ).tolist()
        half_x = half
    else:
        cnt_x = [np.ceil(shp[1] / 2)]
        half_x = np.ceil(shp[1] / 2)

    if n_cell_y > 0:
        cnt_y = np.linspace(
            half, shp[0] - half, 2 + int(np.round(n_cell_y)), dtype=int
        ).tolist()
        half_y = half
    else:
        cnt_y = [np.ceil(shp[0] / 2)]
        half_y = np.ceil(shp[0] / 2)
    
    return cnt_x, half_x, cnt_y, half_y


def prob_hough_line(d2, x, y, window_size=64, line_length=1, threshold=1, line_gap=1):

    d2 = check_array(d2, ensure_2d=True)

    cnt_x, half_x, cnt_y, half_y = make_tiles(d2, tile_size=window_size)
    # shape = d2.shape
    # # Cycle through tiles of square size
    # max_l = np.min([window_size, shape[0], shape[1]])
    # half = np.floor(max_l / 2)
    # overlap = 1.25

    # n_cell_y = (shape[0] - 2 * half) * overlap / max_l
    # n_cell_x = (shape[1] - 2 * half) * overlap / max_l

    # if n_cell_x > 0:
    #     cnt_x = np.linspace(
    #         half, shape[1] - half, 2 + int(np.round(n_cell_x)), dtype=int
    #     ).tolist()
    #     half_x = half
    # else:
    #     cnt_x = [np.ceil(shape[1] / 2)]
    #     half_x = np.ceil(shape[1] / 2)

    # if n_cell_y > 0:
    #     cnt_y = np.linspace(
    #         half, shape[0] - half, 2 + int(np.round(n_cell_y)), dtype=int
    #     ).tolist()
    #     half_y = half
    # else:
    #     cnt_y = [np.ceil(shape[0] / 2)]
    #     half_y = np.ceil(shape[0] / 2)

    coords = []
    for cx in cnt_x:
        for cy in cnt_y:
            i_min, i_max = int(cy - half_y), int(cy + half_y)
            j_min, j_max = int(cx - half_x), int(cx + half_x)
            lines = probabilistic_hough_line(
                d2[i_min:i_max, j_min:j_max],
                line_length=line_length,
                threshold=threshold,
                line_gap=line_gap,
                seed=0,
            )

            if np.any(lines):
                coord = np.vstack(lines)
                coords.append(
                    np.c_[
                        x[i_min:i_max, j_min:j_max][
                            coord[:, 1], coord[:, 0]
                        ],
                        y[i_min:i_max, j_min:j_max][
                            coord[:, 1], coord[:, 0]
                        ],
                    ]
                )
    hough_lines = []
    if coords:
        coords = np.vstack(coords)
        for i, j in zip(coords[::2], coords[1::2]):
            hough_lines.append(LineString([i, j]))


    return hough_lines

def filter_hough_lines(
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
    # true mask for all x vector
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
        # filter mask for x vector
        filter_xy = np.ones_like(x, dtype="bool")
        if distance > 0: 
            # mask index
            mask_ind = np.where(mask)[0]
            # concatenate along second axis
            xy = np.c_[x[mask], y[mask]]
            tree = cKDTree(xy)
            nstn = xy.shape[0] # nrow
            # Initialize the filter
            for ii in range(nstn):
                if filter_xy[mask_ind[ii]]:  # if this row is true
                    ind = tree.query_ball_point(xy[ii, :2], distance) # find nearest to this row coord
                    filter_xy[mask_ind[ind]] = False  #  turn nearest point to false, including itself
                    filter_xy[mask_ind[ii]] = True    #  keep the current point to true

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

#  binary pixel to merge into linestrings
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# find single ends points
# def nbr_colrow(x0, y0):  # not used, use cKDTree algo instead
#     xy=[[x0-1, y0], [x0-1, y0-1], [x0, y0-1], [x0+1, y0-1], [x0+1, y0], [x0+1, y0+1], [x0, y0+1], [x0-1, y0+1]]
#     b = [not any(i)<0 for i in xy]
#     return xy

def find_endpoints(xy):
    xy = check_array(xy)
    assert xy.shape[1] == 2, '***only two columns, x and y allowed!'

    tree = cKDTree(xy)
    # def xy_nbr_count(xy0, xy):
    def xy_nbr_count(xy0, xy):
        xy0=np.array(xy0)
        xy=np.array(xy)

        assert len(xy0) == xy.shape[1]
        # xy_nbr = nbr_colrow(xy0[0], xy0[1])

        this_point=cKDTree(xy0.reshape(-1, 2))
        ncnt=this_point.count_neighbors(tree, 1.415)
        return ncnt-1

        # count = 0
        # for i in xy_nbr:
        #     b=np.array([(i==j).all() for j in xy])
        #     if any(b):
        #         count+=1
        # print(f"neight count={neighbor_count}, count={count}")
        # return count

    xy_ends = []
    points = []
    for i in xy:
        if xy_nbr_count(i, xy) == 1:
            xy_ends.append(i)
        elif xy_nbr_count(i, xy) == 2:
            i2 = tree.query_ball_point(i, 1.415)
            p2 = np.array([xy[i2a] for i2a in i2 if not (xy[i2a]==i).all()])
            assert p2.shape == (2, 2)
            dist= np.linalg.norm(p2[1]-p2[0])
            if dist < 1.0001:
                xy_ends.append(i)
        elif xy_nbr_count(i, xy) == 3:
            i2 = tree.query_ball_point(i, 1.415)
            p2 = np.array([xy[i2a] for i2a in i2 if not (xy[i2a]==i).all()])
            assert p2.shape == (3, 2)
            if (p2[:, 0] == p2[0, 0]).all() or (p2[:, 1] == p2[0, 1]).all():
                xy_ends.append(i)
        elif xy_nbr_count(i, xy) == 0:
            points.append(i)

    # return end points xy, and point only xy
    return np.array(xy_ends), np.array(points)


def new_nbr_idx(tree, this_end, this_line, xy):
    # neighbor index
    all_nbr_ind = ckd(tree, this_end)
    # remove neighbor index which is already in the line
    ind2 = []
    for i in all_nbr_ind:
        b2=[]
        for j in this_line:
            b2.append((j==xy[i]).all())
        ind2.append(not any(b2))
    new_nbr_ind = np.where(ind2)[0]  # index of all_nbr_ind
    return all_nbr_ind, new_nbr_ind



# canny edges to index
def ckd(tree, this_xy):
    ind = tree.query_ball_point(this_xy, 1.415, workers=-1) # find nearest to this row coord
    # if len(ind) > 2:
    #     ind2 = tree.query_ball_point(this_xy, 1, workers=-1) # find nearest to this row coord
    return ind

def join_pixel_to_lines(ends, tree, xy):
    lines = []
    used_ends = np.array([])
    for end in ends:
        this_end = end.copy()
        # check this end in used-ends or not
        this_end_used = any([(i==this_end).all() for i in used_ends])

        if not this_end_used:
            this_line=np.array([end])   # initialise this line loop
            ha = True
            while ha:
                ind, this_ind = new_nbr_idx(tree, this_end, this_line, xy)
                # # neighbor index
                # ind = ckd(tree, this_end)
                # # remove neighbor index which is already in the line
                # ind2 = []
                # for i in ind:
                #     b2=[]
                #     for j in this_line:
                #         b2.append((j==xy[i]).all())
                #     ind2.append(not any(b2))
                # this_ind = np.where(ind2)[0]

                last_slope = this_line[-1] - this_line[-2] if len(this_line) > 1 else np.array([0, 0])  # last line segment slope
                if len(this_ind) ==1:
                    current_end = np.array(xy[ind[this_ind[0]]]).ravel()
                    this_end = current_end
                    this_line = np.append(this_line, [this_end], axis=0)
                elif len(this_ind)>1:
                    these_nbr = np.array([xy[j] for j in [ind[i] for i in this_ind]])
                    # nearest by distance
                    nbr_dist = np.array([np.linalg.norm(this_end-br) for br in these_nbr])
                    min_dist = np.min(nbr_dist)

                    nm = np.where(nbr_dist == min_dist)[0]
                    if len(nm) == 1:
                        this_end = np.array(these_nbr[nm]).ravel()
                        this_line = np.append(this_line, [this_end], axis=0)
                    elif len(nm) > 1:
                        these_nbr = these_nbr[nm] # only use min distance nbr
                        # compute slopes
                        these_slopes = np.abs(these_nbr - this_end)
                        # compare slopes
                        ang = [abs(np.arctan2(these_slopes[i2, 1], these_slopes[i2, 0]) - np.arctan2(last_slope[1], last_slope[0]))*180/np.pi for i2 in range(len(nm))]
                        ang = [abs(i2-180) if i2 > 90. else i2 for i2 in ang]
                        ia = np.where(ang == np.min(ang))[0]
                        if len(ia) > 1:
                            if all([abs(ag-90)<0.01 for ag in ang]): # if T intersection
                                n_nbr=[]
                                for nbr1 in these_nbr:
                                    _, this_ind2 = new_nbr_idx(tree, nbr1, this_line, xy)
                                    n_nbr.append(len(this_ind2))
                                m_nbr = np.max(n_nbr)
                                if m_nbr>0 and sum(n_nbr==m_nbr)==1:
                                    ia=np.where(n_nbr==m_nbr)[0]
                                    this_end = these_nbr[ia].ravel()
                                    this_line = np.append(this_line, [this_end], axis=0)
                                else:
                                    ha=False
                            else:
                                ha=False
                        else:
                            # ia = np.argmin(ang)
                            this_end = these_nbr[ia].ravel()
                            this_line = np.append(this_line, [this_end], axis=0)
                        # ind_a = np.where([(i==last_slope).all() for i in this_slopes])[0]
                        # if len(ind_a) == 1:
                        #     this_end = np.array(these_nbr[ind_a]).ravel()
                        #     this_line = np.append(this_line, [this_end], axis=0)
                        # elif len(ind_a) < 1:
                        #     ha = False
                    # ind_slope = np.argmin([abs(i[1]/i[0]) if abs(i[0])>1e-6 else 0 for i in slopes])
                    # this_end = these_ends[ind_slope]
                elif len(this_ind)==0:
                    ha = False
            lines.append(this_line)

        
    return lines
 

def edge_arr2lines(edges, transform, epsilon=1, line_cleanup=False):
    """_summary_

    Args:
        edges (_type_): 2D array
        transform (_type_): _description_

    Returns:
        _type_: _description_
    """
    print("--- pixel to form lines ...")
    y, x = np.where(edges)
    xy = np.c_[x.ravel(), y.ravel()]
    nxy = len(xy)

    ends, points = find_endpoints(xy)

    # build linestring from pixel True

    tree = cKDTree(xy)
    lines = join_pixel_to_lines(ends, tree, xy)
    edge_lines = []
    for ij in lines:
        # edge_lines.append(LineString([transform*i for i in ij]))
        edge_lines.append(ij)

    xy_set = set([tuple(i) for i in xy])
    def do_remaining_pixels(used_xy, xy_set):
        # lines = check_array(np.vstack(lines))
        # used_xy = set([tuple(i) for i in np.unique(lines, axis=0)])
        # xy2 = np.array([i for i in xy_set if i not in used_xy])
        xy2 = np.array([i for i in xy_set])

        ends, _ = find_endpoints(xy2)
        tree = cKDTree(xy2)
        plines = join_pixel_to_lines(ends, tree, xy2)
        if plines:
            lines2_xy = check_array(np.vstack(plines))
            used_xy2 = set([tuple(i) for i in np.unique(lines2_xy, axis=0)])
        else:
            used_xy2 = xy_set
        # xy_set = set([i for i in xy_set if i not in used_xy2])
        
        return plines, used_xy2

    lines_xy = check_array(np.vstack(lines))
    used_xy = set([tuple(i) for i in np.unique(lines_xy, axis=0)])
    remaining_pixels = nxy-len(used_xy)

    while remaining_pixels / nxy > 0.01 and len(lines) > 2:
        xy_set = set([i for i in xy_set if i not in used_xy])
        # lines, used_xy = do_remaining_pixels(lines_xy, xy, xy_set)
        lines, used_xy = do_remaining_pixels(used_xy, xy_set)
        for ij in lines:
            # edge_lines.append(LineString([transform*i for i in ij]))
            edge_lines.append(ij)

        # lines_xy = check_array(np.vstack(lines))
        # used_xy = set([tuple(i) for i in np.unique(lines, axis=0)])
        remaining_pixels -= len(used_xy)

    if line_cleanup:  # seems not need to do this
        # remove overlapped lines - clean up, may not be optimal
        print('--- clean up lines ...')
        idx_overlap = []
        for idx in range(len(edge_lines)):
            this_set = set([tuple(i) for i in edge_lines[idx]])
            for jdx in range(idx, len(edge_lines)):
                jd = edge_lines[jdx]
                if len(jd) >= len(this_set) and idx!=jdx:
                    jd_set = set([tuple(i) for i in jd])
                    ndiff = len([i for i in this_set & jd_set])
                    if ndiff/len(jd_set) > 0.7:
                        idx_overlap.append(idx)
                        break
        idx_unique = [i for i in range(len(edge_lines)) if i not in idx_overlap]
        edge_lines = [edge_lines[i] for i in idx_unique]

    # rdp
    rdp_lines = [rdp.rdp(line, epsilon=epsilon) for line in edge_lines]

    # to linestring
    edge_lines=[]
    for rdp_line in rdp_lines:
        if not (rdp_line[0] == rdp_line[-1]).all():
            edge_lines.append(LineString([transform*xy for xy in rdp_line]))


    # if edge_lines:

    #     lines = join_others(lines, xy)
    #     # lines = check_array(np.vstack(lines))
    #     # used_xy = set([tuple(i) for i in np.unique(lines, axis=0)])
    #     # xy_set = set([tuple(i) for i in xy])
    #     # xy2 = np.array([i for i in xy_set if i not in used_xy])

    #     # ends, _ = find_endpoints(xy2)
    #     # tree = cKDTree(xy2)
    #     # lines = join_pixel_to_lines(ends, tree, xy2)
    #     for ij in lines:
    #         edge_lines.append(LineString([transform*i for i in ij]))
    
    return edge_lines

def pixel_intersection_type(this_end, this_line, all_nbr_ind, new_nbr_ind, xy, xytree):
    this_grid_nbr = np.array([xy[j] for j in [all_nbr_ind[i] for i in new_nbr_ind]]) # current grid nbr
    this_grid_nbr_dist = np.array([np.linalg.norm(this_end-br) for br in this_grid_nbr])
    min_dist_4_this_grid = np.min(this_grid_nbr_dist)
    min_idx = np.where(this_grid_nbr_dist == min_dist_4_this_grid)[0]

    if len(min_idx) == 1:
        current_end = np.array(this_grid_nbr[min_idx]).ravel()
        all_ind2, new_nbr_ind2 = new_nbr_idx(xytree, current_end, np.append(this_line, [current_end], axis=0), xy)
        that_grid_nbr = np.array([xy[j] for j in [all_ind2[i] for i in new_nbr_ind2]])
        is_new_nbr = [] # remove both nbr for this and that grid
        for that_grid in that_grid_nbr:
            is_old_nbr = []
            for this_grid in this_grid_nbr:
                is_old_nbr.append(np.array_equal(that_grid, this_grid))
            is_new_nbr.append(not any(is_old_nbr))
        that_grid_nbr = that_grid_nbr[is_new_nbr]
        that_grid_nbr_dist = np.array([np.linalg.norm(this_grid_nbr[min_idx].ravel()-br) for br in that_grid_nbr])
        if len(that_grid_nbr_dist) > 0:
            min_dist_4_that_grid = np.min(that_grid_nbr_dist)
            min_idx2 = np.where(that_grid_nbr_dist == min_dist_4_that_grid)[0]
            if len(min_idx2) > 1 and min_dist_4_that_grid==1:
                pix_intersection_type = 'T'
            elif len(min_idx2) > 1 and min_dist_4_that_grid>1:
                pix_intersection_type = 'Y'
            elif len(min_idx2) == 1:
                dist2this_end = np.array([np.linalg.norm(that_grid_nbr[min_idx2].ravel()-this_end)])
                dist2current_end = np.array([np.linalg.norm(that_grid_nbr[min_idx2].ravel()-current_end)])

                # if dist2this_end == 2 and dist2current_end == 1 and (this_end[0] < 10 or this_end[1]<10):
                # if dist2this_end == 2 and dist2current_end == 1 and this_end[0]<10:
                if dist2this_end < 2:
                    dist2current_end

                if abs(dist2this_end-2*dist2current_end) <1e-6:
                    pix_intersection_type = 'I'
                else:
                    pix_intersection_type = 'Y'
            else:
                pix_intersection_type = 'Y'
        else:
            pix_intersection_type = 'T'

    else:
        pix_intersection_type = 'T'

    return pix_intersection_type


def prune_branches(branches, small_branch_threshold = 10):
    """_summary_
        prune small branches fo 2D binary image array
    Args:
        branches (_type_): _description_
        small_branch_threshold (int, optional): _description_. Defaults to 10.

    Returns:
        _type_: _description_
    """

    branches = check_array(branches, ensure_2d=True)

    # shp = d2a.shape
    shp = branches.shape
    y, x = np.where(branches)
    xy = np.c_[x.ravel(), y.ravel()]
    xytree = cKDTree(xy)

    ends = find_endpoints(xy)[0]
    bd = [0]+[i-1 for i in shp] # border coord in list

    bd2 = []
    for i in ends:
        bd2.append(not any(i1 in bd for i1 in i))
        
    ends = ends[bd2] # update ends coord


    used_ends = np.array([])
    small_branches_xy = np.array([])
    for end in ends:
        to_trim = False
        this_end = end.copy()
        # check this end in used-ends or not
        this_end_used = any([(i==this_end).all() for i in used_ends])

        end_tree = cKDTree(np.reshape(end, (-1, 2)))
        nbr = end_tree.count_neighbors(xytree, 2.5) # further nbrs

        # fig, ax = plt.subplots(1, figsize=(12, 12))
        # ax.imshow(branches)
        # ax.plot(end[0], end[1], 'r+')
        # plt.show()


        if not this_end_used and nbr<=3:
            this_line=np.array([end])   # initialise this line loop

            ha = True
            while ha:
                ind, nbr_ind = new_nbr_idx(xytree, this_end, this_line, xy)
                # last_slope = this_line[-1] - this_line[-2] if len(this_line) > 1 else np.array([0, 0])  # last line segment slope
                if len(nbr_ind) ==1:
                    current_end = np.array(xy[ind[nbr_ind[0]]]).ravel()
                    this_end = current_end
                    this_line = np.append(this_line, [this_end], axis=0)
                elif len(nbr_ind) > 1: # it's an intersection
                    ## pixel intersection type from one point
                    pix_intersection_type = pixel_intersection_type(this_end, this_line, ind, nbr_ind, xy, xytree)

                    if pix_intersection_type != 'I':
                        to_trim=True
                    # these_nbr = np.array([xy[j] for j in [ind[i] for i in nbr_ind]])
                    # # nearest by distance
                    # nbr_dist = np.array([np.linalg.norm(this_end-br) for br in these_nbr])
                    # min_dist = np.min(nbr_dist)
                    # nm = np.where(nbr_dist == min_dist)[0]
                    # if len(nm) == 1:
                    #     nbr_dist2 = np.array([np.linalg.norm(these_nbr[nm]-br) for br in np.delete(these_nbr, nm)])
                    # elif len(nm)>1:
                    #     to_trim=True
                    # # if min_dist -1.0 > 1e-6:
                    # #     to_trim = True
                    # if len(nbr_ind) > 2:
                    #     to_trim = True

                    ha = False
                elif len(nbr_ind) < 1:
                    ha = False
            if not any(i in bd for i in this_line[-1]) and len(this_line) < small_branch_threshold and to_trim:
                small_branches_xy = np.append(small_branches_xy, this_line)

    small_branches_xy = small_branches_xy.reshape(-1, 2) 

    # d2b = d2a.copy()
    for j in small_branches_xy:
        # d2b[int(j[1]), int(j[0])] = False
        branches[int(j[1]), int(j[0])] = False

    return branches

# def trim_small_holes(d2, area_threshold=64, footprint=morphology.disk(1)):
#     d2b_area = morphology.area_closing(d2, area_threshold=area_threshold, connectivity=1)
#     d2b = morphology.dilation(morphology.erosion(d2b_area, footprint=footprint))
#     d2 &= np.invert(d2b)
#     return d2
def prune_holes(d2_amp, d2_bin, area_threshold = 128):
    d2_bin = check_array(d2_bin, ensure_2d=True)
    y, x = np.where(d2_bin)
    xy = np.c_[x.ravel(), y.ravel()]
    xytree = cKDTree(xy)
    area_fill = morphology.area_closing(d2_bin, area_threshold=area_threshold, connectivity=1)
    area_fill_only = morphology.erosion(area_fill, footprint=morphology.disk(1))
    dilated_area_fill = morphology.dilation(area_fill_only)
    area_circ = ~area_fill_only & dilated_area_fill

    s = sp.ndimage.generate_binary_structure(2,2)
    label_obj, nb_labels = sp.ndimage.label(area_circ, structure=s)

    if nb_labels > 0:
        for lbl in np.arange(1, nb_labels+1):
            this_y, this_x = np.where(label_obj==lbl)
            this_xy = np.c_[this_x.ravel(), this_y.ravel()]
            this_xytree = cKDTree(this_xy)
            these_nbr_idx = xytree.query_ball_point(this_xy, r=1.5)
            assert len(these_nbr_idx) == len(this_xy)
            connection_xy = []
            for this_nbr_idx in these_nbr_idx:
                if len(this_nbr_idx) > 3:
                    is_nbr = []
                    for this_idx in this_nbr_idx:
                        is_nbr.append(not len(this_xytree.query_ball_point(xy[this_idx], r=0)) == 1)
                    connection_xy.append(xy[list(compress(this_nbr_idx, is_nbr))])
            if connection_xy:
                connection_xy = np.unique(np.concatenate(connection_xy, axis=0), axis=0)
                connection_tree = cKDTree(connection_xy)
                con_xy = connection_tree.query_ball_tree(connection_tree, r=2.5)
                ncon = len(np.unique(con_xy))
                if ncon < 2:
                    d2_bin[this_y, this_x] = False
                elif ncon == 2:



                    pass


    return d2_bin

def clear_short_binary_lines(d2_bin, pixel_length_threshold = 5):

    s = sp.ndimage.generate_binary_structure(2,2)
    label_obj, nb_labels = sp.ndimage.label(d2_bin, structure=s)
    if nb_labels > 0:
        for lbl in np.arange(1, nb_labels+1):
            this_y, this_x = np.where(label_obj==lbl)
            this_xy = np.c_[this_x.ravel(), this_y.ravel()]
            if len(this_xy) <= pixel_length_threshold:
                    d2_bin[this_y, this_x] = False
    return d2_bin



def filter_binary_lineaments(d2, tile_size, small_branch_threshold): #, trim_holes=False):
    # window_size = 201
    # small_branch_threshold = 25
    # d2 = lineobj.ridges.copy()
    d2 = check_array(d2, ensure_2d=True)
    cnt_x, half_x, cnt_y, half_y = make_tiles(d2, tile_size=tile_size)

    d2_new = d2.copy()
    for cx in cnt_x:
        # print(cx)
        for cy in cnt_y:
            i_min, i_max = int(cy - half_y), int(cy + half_y)
            j_min, j_max = int(cx - half_x), int(cx + half_x)
            d2a = d2[i_min:i_max, j_min:j_max].copy()
            d2a = prune_branches(d2a, small_branch_threshold=small_branch_threshold)
            # if trim_holes:
            #     # remove small closed area
            #     d2a = trim_small_holes(d2a, footprint=morphology.disk(1))

            d2_new[i_min:i_max, j_min:j_max] &= d2a

    return d2_new


#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class LineamentDetection():


    def __init__(self):

        self.d2 = None
        self.x = None
        self.y = None
        self.x2 = None
        self.y2 = None

        self.resolution = None
        self.crs = None
        self.affine = None
        self.extent = None
        # self.epsilon = None

        self.edges = None  # intermediate binary product of lineament production

        self.lineaments = None  # final 2D array binary map from this LineamentDetection class
        # self.lines_filtered_binary = None  # 2D array binary map
        # self.lines = None   # linestring



    def fit(self, arr2d, affine): #, epsilon=1):
    
        self.d2 = check_array(arr2d, ensure_2d=True)
        # self.epsilon = epsilon

        rows, cols = self.d2.shape

        col, row = np.meshgrid(np.arange(cols), np.arange(rows))

        self.x2=[]
        self.y2=[]
        for c, r in zip(col, row):
            a, b = affine*(c, r)
            self.x2.append(a)
            self.y2.append(b)
        self.x2 = np.array(self.x2)
        self.y2 = np.array(self.y2)
        self.x2 = check_array(self.x2, ensure_2d=True)
        self.y2 = check_array(self.y2, ensure_2d=True)
        self.affine = affine

        self.x = column_or_1d(self.x2[0, :])
        self.y = column_or_1d(self.y2[:, 0])


        x1, y1 = affine*(0, rows)
        x2, y2 = affine*(cols, 0)
        self.extent = [x1, x2, y1, y2]


        return self
    def edge_detection(self, sigma):
        print("--- detecting canny edges ...")
        arr = self.d2
        arr -= arr.min()
        arr /= arr.max()

        self.edges = None  # set this intermediate product to None
        # Find lineament edges
        lineaments = canny(arr, sigma=sigma, use_quantiles=True)

        return lineaments

    def edges_to_lines(self):
        assert self.edges is not None, "*** run edge_detection first!"
        self.edge_lines = edge_arr2lines(self.edges, self.transform)
        return self.edge_lines

    
    # def edges_to_hough_lines(self, window_size=64, line_length=1, threshold=1, line_gap=1):
    #     assert self.edges is not None, "*** run edge_detection first!"
    #     x2, y2 = np.meshgrid(self.x, self.y)
    #     self.edge_hough_lines = prob_hough_line(self.edges, x=x2, y=y2, window_size=window_size, 
    #                                     line_length=line_length, 
    #                                     threshold=threshold, line_gap=line_gap)
        
        return self.edge_hough_lines

    def ridge_detection(self, black_ridges=False, filter_obj_size=3, block_size=None):
        """
        sto filter to detect continuous ridges
        """
        arr = self.d2
        arr -= arr.min()
        arr /= arr.max()
        self.edges = filters.sato(arr, black_ridges=black_ridges, mode='reflect')
        if block_size is None:
            thresh_im = filters.threshold_otsu(self.edges)
            ridges_binary = self.edges > thresh_im
        else:
            block_size = int(block_size)
            ridges_binary = self.edges > filters.threshold_local(self.edges, block_size=block_size)

        ridges_binary = self.__filter_small_binobj(ridges_binary, bin_size_threshold=filter_obj_size)
        ridges_binary = morphology.closing(ridges_binary, morphology.square(3))
        self.lineaments = morphology.skeletonize(ridges_binary)

        # self.have_ridges = True
        return ridges_binary
        return self

    # def ridges_to_lines(self):
    #     assert self.ridges is not None, "*** run ridge_detection first!"
    #     self.ridge_lines = edge_arr2lines(self.ridges, self.transform, line_cleanup=False)

    #     return self.ridge_lines


    # def ridges_to_hough_lines(self, window_size=64, line_length=1, threshold=1, line_gap=1):
    #     if self.have_ridges:
    #         x2, y2 = np.meshgrid(self.x, self.y)
    #         self.ridge_hough_lines = prob_hough_line(self.ridges, x=x2, y=y2, window_size=window_size, 
    #                                         line_length=line_length, 
    #                                         threshold=threshold, line_gap=line_gap)
    #     return self.ridge_hough_lines

    def image_thresholding(self, type='global', local_block_size='auto', local_offset=0, global_threshold=None):

        arr = self.d2
        arr -= arr.min()
        arr /= arr.max()

        if type == 'global':
            if global_threshold is None:
                thresh_im = filters.threshold_otsu(arr)
            else:
                assert global_threshold > 10 and global_threshold < 100
                thresh_im = np.percentile(arr.ravel(), global_threshold)
            thresh_im_binary = arr > thresh_im
        elif type == 'local':
            if local_block_size =='auto':
                block_size = max([int(min([len(self.x), len(self.y)])*0.2), 11])
            else:
                block_size=int(local_block_size)
            if block_size%2 == 0:
                block_size -= 1
            offset = max([5, int(block_size*0.33)])
            thresh_im_binary = arr > filters.threshold_local(arr, block_size=block_size, offset=local_offset)

        return thresh_im_binary

    def thresh_to_lines(self, thresh_im, filter_obj_size=1, ):
        # assert self.thresh_im is not None, "*** run image_thresholding first!!"

        barr = self.__filter_small_binobj(thresh_im, bin_size_threshold=filter_obj_size)
        barr = morphology.closing(barr, morphology.square(3))
        barr = morphology.skeletonize(barr)
        # self.thresh_lines = edge_arr2lines(barr, self.transform, line_cleanup=False)
        # self.thresh_lines = barr

        return barr

    def scharr(self, low_pct=20):

        arr = self.d2
        arr -= arr.min()
        arr /= arr.max()

        dmask = arr > filters.threshold_otsu(arr)
        dout = filters.scharr(arr)
        dout = dout < np.percentile(dout.ravel(), low_pct)

        return morphology.skeletonize(dout & dmask)
        

    def __filter_small_binobj(self, bin_arr, bin_size_threshold=1):
        s = sp.ndimage.generate_binary_structure(2,2)
        label_obj, nb_labels = sp.ndimage.label(bin_arr, structure=s)
        sizes_bin = np.bincount(label_obj.ravel())
        mask_sizes = sizes_bin > bin_size_threshold
        mask_sizes[0] = 0
        return mask_sizes[label_obj]

    # def __prune_parm(self, **kwargs):
    #     if 'tile_size' in kwargs.keys():
    #         tile_size = kwargs['tile_size']
    #         assert isinstance(tile_size, int)
    #     else:
    #         raise ValueError("tile_size not found!")

    #     if 'small_branch_threshold' in kwargs.keys():
    #         small_branch_threshold = kwargs['small_branch_threshold']
    #         assert isinstance(small_branch_threshold, int)
    #     else:
    #         raise ValueError("small_branch_threshold not found!")

    #     if 'hole_area' in kwargs.keys():
    #         hole_area = kwargs['hole_area']
    #     else:
    #         hole_area = None
        
    #     if 'repeat_prune' in kwargs.keys():
    #         repeat_prune= kwargs['repeat_prune']
    #     else:
    #         repeat_prune = True
        
    #     return tile_size, small_branch_threshold, hole_area, repeat_prune

    def transform(self, type='sato', **kwargs):

        if type.lower() == 'canny':

            if 'sigma' in kwargs.keys():
                sigma = kwargs['sigma']

            self.lineaments = self.edge_detection(sigma=sigma)

            # if 'prune' in kwargs.keys() and kwargs['prune'] is True:
            #     tile_size, small_branch_threshold, hole_area, repeat_prune = self.__prune_parm(**kwargs)
            #     self.lines_filtered_binary =filter_binary_lineaments(self.lineaments, tile_size=tile_size, small_branch_threshold=small_branch_threshold)
            #     if hole_area is not None:
            #         self.lines_filtered_binary = prune_holes(self.d2, self.lines_filtered_binary, area_threshold=hole_area)
            #     self.lines_filtered_binary = clear_short_binary_lines(self.lines_filtered_binary, small_branch_threshold)

            #     if repeat_prune:
            #         self.lines_filtered_binary =filter_binary_lineaments(self.lines_filtered_binary, tile_size=tile_size, small_branch_threshold=small_branch_threshold)
            #         if hole_area is not None:
            #             self.lines_filtered_binary = prune_holes(self.d2, self.lines_filtered_binary, area_threshold=hole_area)
            #         self.lines_filtered_binary = clear_short_binary_lines(self.lines_filtered_binary, small_branch_threshold)

        elif type.lower() == 'sato' or type.lower() == 'ridges':
            if 'black_ridges' in kwargs.keys():
                black_ridges = kwargs['black_ridges']
            else:
                black_ridges = False
            if 'block_size' in kwargs.keys():
                block_size = kwargs['block_size']
            else:
                block_size = 11
            if 'filter_obj_size' in kwargs.keys():
                filter_obj_size = kwargs['filter_obj_size']
            else:
                filter_obj_size=3


            self.ridge_detection(black_ridges=black_ridges, \
                            block_size=block_size, filter_obj_size=filter_obj_size)

            # if 'prune' in kwargs.keys() and kwargs['prune'] is True:
            #     tile_size, small_branch_threshold, hole_area, repeat_prune = self.__prune_parm(**kwargs)
            #     self.lines_filtered_binary =filter_binary_lineaments(self.lineaments, \
            #                         tile_size=tile_size, small_branch_threshold=small_branch_threshold)
            #     if hole_area is not None:
            #         self.lines_filtered_binary = prune_holes(self.d2, self.lines_filtered_binary, area_threshold=hole_area)
            #     self.lines_filtered_binary = clear_short_binary_lines(self.lines_filtered_binary, small_branch_threshold)

            #     if repeat_prune:
            #         self.lines_filtered_binary =filter_binary_lineaments(self.lines_filtered_binary, tile_size=tile_size, small_branch_threshold=small_branch_threshold)
            #         if hole_area is not None:
            #             self.lines_filtered_binary = prune_holes(self.d2, self.lines_filtered_binary, area_threshold=hole_area)
            #         self.lines_filtered_binary = clear_short_binary_lines(self.lines_filtered_binary, small_branch_threshold)

        elif type.lower() == 'thresholding' or type.lower() == 'max':

            if 'threshold_type' in kwargs.keys():
                threshold_type = kwargs['threshold_type']
            else:
                threshold_type = 'local'
            if 'local_block_size' in kwargs.keys():
                local_block_size = kwargs['local_block_size']
            else:
                local_block_size = 'auto'
            if 'local_offset' in kwargs.keys():
                local_offset = kwargs['local_offset']
            else:
                local_offset = 0
            if 'global_threshold' in kwargs.keys():
                global_threshold = kwargs['global_threshold']
            else:
                global_threshold = None
            if 'filter_obj_size' in kwargs.keys():
                filter_obj_size = kwargs['filter_obj_size']
            else:
                filter_obj_size=3
            

            self.edges = self.image_thresholding(type=threshold_type, local_block_size=local_block_size, \
                                            local_offset=local_offset, global_threshold=global_threshold)
            self.lineaments = self.thresh_to_lines(self.edges, filter_obj_size=filter_obj_size)

            # if 'prune' in kwargs.keys() and kwargs['prune'] is True:
            #     tile_size, small_branch_threshold, hole_area, repeat_prune = self.__prune_parm(**kwargs)
            #     self.lines_filtered_binary =filter_binary_lineaments(self.lineaments, \
            #                         tile_size=tile_size, small_branch_threshold=small_branch_threshold)
            #     if hole_area is not None:
            #         self.lines_filtered_binary = prune_holes(self.d2, self.lines_filtered_binary, area_threshold=hole_area)
            #     self.lines_filtered_binary = clear_short_binary_lines(self.lines_filtered_binary, small_branch_threshold)

            #     if repeat_prune:
            #         self.lines_filtered_binary =filter_binary_lineaments(self.lines_filtered_binary, tile_size=tile_size, small_branch_threshold=small_branch_threshold)
            #         if hole_area is not None:
            #             self.lines_filtered_binary = prune_holes(self.d2, self.lines_filtered_binary, area_threshold=hole_area)
            #         self.lines_filtered_binary = clear_short_binary_lines(self.lines_filtered_binary, small_branch_threshold)


        elif type.lower() == 'scharr':

            if 'low_pct' in kwargs.keys():
                low_pct = kwargs['low_pct']
            else:
                low_pct = 20

            self.lineaments = self.scharr(low_pct=low_pct)

        else:
            raise TypeError("*** Wrong type name!")

        return self.lineaments #, self.lines_filtered_binary, self.lines





    # dd = lineobj.ridges.copy()
    # dd = dd*1.0
    # masked_data = np.ma.masked_where(lineobj.ridges==True, data)
    # # b=gpd.GeoDataFrame({'geometry':lineobj.ridge_lines})
    # # b1=gpd.GeoDataFrame({'geometry':lineobj.ridge_hough_lines})
    # # show(src, ax=ax)
    # # ax.imshow(data)
    # # a1, a2, a3, a4 = list(src.bounds)
    # # ax.imshow(masked_data, extent=[a1, a3, a2, a4])
    # ax.imshow(masked_data)
    # # b.plot(ax=ax)


    # # b1=gpd.GeoDataFrame({'geometry':lineobj.ridge_hough_lines})
    # # fig, ax = plt.subplots(1, figsize=(16, 16))
    # # # ax.imshow(lineobj.ridges, extent=lineobj.extent)
    # # b1['length']=b1['geometry'].length
    # # b1.plot(ax=ax, column='length')
    # # # b1[b1['length']>1000].plot(ax=ax, column='length')




# --------------------------------------------------------------------------------------------------------------



# window_size = 201
# small_branch_threshold = 25
# d2 = lineobj.ridges.copy()
# d2 = check_array(d2, ensure_2d=True)
# cnt_x, half_x, cnt_y, half_y = make_tiles(d2, tile_size=window_size)

# d2_new = d2.copy()
# for cx in cnt_x:
#     print(cx)
#     for cy in cnt_y:
#         i_min, i_max = int(cy - half_y), int(cy + half_y)
#         j_min, j_max = int(cx - half_x), int(cx + half_x)
#         d2a = d2[i_min:i_max, j_min:j_max].copy()
#         d2a = prune_branches(d2a, small_branch_threshold=small_branch_threshold)
#         # remove small closed area
#         d2a = trim_small_holes(d2a, footprint=morphology.disk(1))

#         d2_new[i_min:i_max, j_min:j_max] &= d2a




# fig, ax = plt.subplots(1, 2, figsize=(18, 16))
# ax[0].imshow(d2)
# ax[1].imshow(d2_new)




# if False:
#     a = lineobj.edges[100:250, 0:150]
#     y1, x1 = np.where(a)
#     xy1 = np.c_[x1.ravel(), y1.ravel()]

#     xy = xy1#[200:500, :]

#     ends, points = find_endpoints(xy)

#     # build linestring from pixel True
#     xy2 = np.delete(xy, np.where([(xy==i).all() for i in points]), axis=0)
#     tree = cKDTree(xy)

#     lines = pixel_to_lines(ends, tree)

#     fig, ax = plt.subplots(1, figsize=(16, 16))
#     dd = gpd.GeoDataFrame({'geometry': lines})
#     ax.imshow(a)
#     dd.plot(ax=ax)













# %%
