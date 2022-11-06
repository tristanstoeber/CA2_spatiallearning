import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import norm



def unit_vector(vector):
    """ Returns the unit vector of the vector. 
    Code by David Wolever [1]
    [1] https://stackoverflow.com/a/13849249
    """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793

    Code by David Wolever [1]
    [1] https://stackoverflow.com/a/13849249
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def rotate(p, angle, origin=(0, 0)):
    """
    Modified after ImportanceOfBeingErnest [1]
    [1] https://stackoverflow.com/a/58781388
    """
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T-o.T) + o.T).T)


def is_on_right_side(x, y, xy0, xy1):
    """
    Copied from JohanC [1]
    [1] https://stackoverflow.com/questions/63527698/determine-if-points-
    are-within-a-rotated-rectangle-standard-python-2-7-library
    """
    x0, y0 = xy0
    x1, y1 = xy1
    a = float(y1 - y0)
    b = float(x0 - x1)
    c = - a*x0 - b*y0
    return a*x + b*y + c >= 0

def to_dict(input_ordered_dict):
    return loads(dumps(input_ordered_dict))

def convert_center_df_info(df_info, flip_y=False):
    for i, row_i in df_info.iterrows():
        pixelspermetre = float(row_i['pixelspermetre'])

        # convert maze info
        xy_center = np.array(
            [row_i['info']['centre']['x'],
             row_i['info']['centre']['y']],
            dtype=float)
        xy_center /= pixelspermetre
        if flip_y:
            xy_center[1] = -1*xy_center[1]
        row_i['info']['centre']['x'] = xy_center[0]
        row_i['info']['centre']['y'] = xy_center[1]

        for key in row_i['info']['boundingbox'].keys():
            row_i['info']['boundingbox'][key] = float(row_i['info']['boundingbox'][key])/pixelspermetre
        row_i['info']['boundingbox']['x'] -= xy_center[0]
        if flip_y:
            row_i['info']['boundingbox']['y'] = -1*float(row_i['info']['boundingbox']['y'])
        row_i['info']['boundingbox']['y'] -= xy_center[1]

        # convert predetermined platform and quadrants
        for quadr_i in ['Platform', 'NE', 'NW', 'SE', 'SW']:
            row_i[quadr_i]['centre']['x'] = (
                float(row_i[quadr_i]['centre']['x'])/pixelspermetre - xy_center[0])
            if flip_y:
                row_i[quadr_i]['centre']['y'] = -1*float(row_i[quadr_i]['centre']['y'])
            row_i[quadr_i]['centre']['y'] = (
                float(row_i[quadr_i]['centre']['y'])/pixelspermetre - xy_center[1])    

            row_i[quadr_i]['boundingbox']['x'] = (
                float(row_i[quadr_i]['boundingbox']['x'])/pixelspermetre - xy_center[0])
            if flip_y:
                row_i[quadr_i]['boundingbox']['y'] = -1*float(row_i[quadr_i]['boundingbox']['y'])
            row_i[quadr_i]['boundingbox']['y'] = (
                float(row_i[quadr_i]['boundingbox']['y'])/pixelspermetre - xy_center[1])    
            row_i[quadr_i]['boundingbox']['w'] = (
                float(row_i['Platform']['boundingbox']['w'])/pixelspermetre)
            row_i[quadr_i]['boundingbox']['h'] = (
                float(row_i[quadr_i]['boundingbox']['h'])/pixelspermetre)
    return df_info

def interpolate_xy_values(xy, t, dt):

    # interpolate values with equal dt
    t_new = np.arange(t.min(), t.max()+dt, dt)

    ls_xy_new = []
    for dim_i in range(xy.shape[1]):
        xy_i_new = np.interp(t_new, t, xy[:, dim_i])
        ls_xy_new.append(xy_i_new)
    xy_new = np.vstack(ls_xy_new).T
    
    return xy_new, t_new

def exclude_nan_values(xy, t):
    bool_nan =  np.any(np.isnan(xy), axis=1)
    t = t[~bool_nan]
    xy = xy[~bool_nan]
    return xy, t

def determine_start_end_of_blobs(a, min_len=None):
    m = np.concatenate(( [True], ~a, [True] ))  # Mask                                                                                                                                                             
    ss = np.flatnonzero(m[1:] != m[:-1]).reshape(-1,2)   # Start-stop limits                                                                                                                                       
    if min_len:
        ss_bool = (ss[:,1] - ss[:,0]) >= min_len
        ss = ss[ss_bool]
        # if no value is present modify output                                                                                                                                                                     
        if ~np.any(ss_bool):
            ss = ss.flatten()
    return ss

def detect_platform_crossings(
    xy,
    t,
    xy_pltfrm,
    radius_pltfrm):
    
    assert(len(xy)==len(t))
    
    dist = np.sqrt(np.sum((xy-xy_pltfrm)**2, axis=1))
    bool_on_platform = dist <= radius_pltfrm
    
    ss = determine_start_end_of_blobs(bool_on_platform)
    return ss

def in_annulus(
    xy,
    xy_annulus,
    radius_outer,
    radius_inner=0.):
        
    dist = np.sqrt(np.sum((xy-xy_annulus)**2, axis=1))
    bool_outer = dist <= radius_outer
    bool_inner = dist <= radius_inner
    
    bool_annulus = (bool_outer & ~bool_inner)
    return bool_annulus

def remove_trailing_positions_on_platform(
    xy,
    t,
    xy_pltfrm,
    radius_pltfrm,
    return_pltfrm_crossings=False):
    """
    Remove those positions from a trace during which
    the animal is left on the platform after a trial.
    """
    
    ss = detect_platform_crossings(
        xy,
        t,
        xy_pltfrm,
        radius_pltfrm)
    if len(ss)>0:
        if ss[-1][1] == len(t):
            xy = xy[:ss[-1][0], :]
            t = t[:ss[-1][0]]

    if return_pltfrm_crossings:
        return xy, t, ss
    else:
        return xy, t
    
def time_in_blobs(ss, t):
    dt = np.median(np.diff(t))
    # add one timestep
    t = np.append(t, np.max(t)+dt)
    ls_t = []
    for ss_i in ss:
        dt = t[ss_i[1]] - t[ss_i[0]]
        ls_t.append(dt)
    return np.sum(ls_t)


def speed(
    xy,
    t,
    ):
    
    dt = np.diff(t)
    dxy = np.diff(xy, axis=0)
    
    # calculate speed
    s = np.sqrt(np.sum(dxy**2, axis=1))/dt

    return s


def acceleration(
    xy,
    t):

    s = speed(xy, t)
    
    # calculate acceleartion in m/s**2
    dt = np.diff(t)
    a = np.diff(s)/dt[:-1]
    
    return a


def bool_in_corridor(x, corr_x_start, corr_x_stop, corr_width):
    """
    Returns wether any element of x is in corridor, 
    defined by start and stop positions as well as width.
    # TODO: avoid that corridor extends beyond start and stop
    
    Parameters
    ----------
    x : ndarray
         N*D array containing data with `float` type.
         N number of observations, D number of dimensions
    corr_x_start : ndarray
        D array containing start position of corridor
        D number of dimensions        
    corr_x_stop : ndarray
        D array containing stop position of corridor
        D number of dimensions 
    corr_width : float
        Width of corridor

    Returns
    -------
    ndarray
        Boolean array with N entries
    
    """

    # distance from corridor center
    p1 = np.array(corr_x_start)
    p2 = np.array(corr_x_stop)
    p3 = np.array(x)
    d = norm(np.cross(p2-p1, p1-p3))/norm(p2-p1)
    d = []
    for i in p3:
        d.append(norm(np.cross(p2-p1, p1-i))/norm(p2-p1))
    d = np.array(d)
    
    bool_in_corridor = d<=corr_width
    return bool_in_corridor

def create_value_map(val, x, bins, f):
    ls_idx = []
    for i in range(x.shape[1]):
        idx_i = np.digitize(x[:,i], bins[i])
        ls_idx.append(idx_i)
    idx = np.array(ls_idx).T

    # collect all values of a specific bin in val_map
    val_map = {}
    
    for idx_i in np.unique(idx, axis=0):
        val_map[tuple(idx_i)] = []
        
    for i, idx_i in enumerate(idx):          
        val_map[tuple(idx_i)].append(val[i])
    
    # apply f to collected valuees
    f_map = {}
    for idx_i in np.unique(idx, axis=0):
        f_map[tuple(idx_i)] = f(val_map[tuple(idx_i)])
    
    # fill values into bins
    ar_f = np.zeros([len(bin_i)+1 for bin_i in bins])
    for idx_i in np.unique(idx, axis=0):
        ar_f[tuple(idx_i)] = f_map[tuple(idx_i)]
    
    return ar_f
