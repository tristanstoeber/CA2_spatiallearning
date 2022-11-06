import numpy as np
import tools as tls
from scipy import signal

def latency_to_platform(
    xy,
    t,
    xy_pltfrm,
    radius_pltfrm,
    trial_has_to_end_on_pltfrm=True):
    
    # detect platform crossings
    cross = tls.detect_platform_crossings(
        xy,
        t,
        xy_pltfrm,
        radius_pltfrm)
    
    latency = np.nan # return nan if no valid crossing is detected

    if len(cross)>0:
        # if trial has to end on platform, 
        # set latency to start of last crossing
        if trial_has_to_end_on_pltfrm:
            if cross[-1][1] == len(t):
                latency = t[cross[-1][0]]
        # otherwise latency is time to first crossing
        else:
            latency = t[cross[0][0]]
    return latency


def path_length(xy):
    diff = np.diff(xy, axis=0)
    path_length = np.sum(np.sqrt(np.sum(diff**2, axis=1)))
    return path_length


def mean_distance_to_point(
    xy,
    t,
    xy_point):
      
    dist = np.sqrt(np.sum((xy-xy_point)**2, axis=1))

    # as of now, t must be equally spaced
    diff_t = np.diff(t)
    if np.allclose(diff_t, np.mean(diff_t)):
        mean_dist = np.mean(dist)
    else:
        raise NotImplementedError('t needs to be equally spaced')
        
    return mean_dist


def time_in_annulus(
    xy,
    t,
    xy_annulus,
    radius_outer,
    radius_inner=0.):
    
    bool_annulus = tls.in_annulus(
        xy,
        xy_annulus,
        radius_outer,
        radius_inner)
    
    ss = tls.determine_start_end_of_blobs(bool_annulus)
    t_annulus = tls.time_in_blobs(ss, t)

    return t_annulus


def number_of_platform_crossings(
    xy,
    t,
    xy_pltfrm,
    radius_pltfrm):
    
    # detect platform crossings
    cross = tls.detect_platform_crossings(
        xy,
        t,
        xy_pltfrm,
        radius_pltfrm)
    return len(cross)

    
def time_in_quadrant(
    xy,
    t,
    sign_quadrant):
    """
    Determine fraction of time in quandrant
    
    Quadrant is defined by sign on centered coordinates
           |
    (-1,1) | (1,1)
    --------------
    (-1,-1)| (1,-1)
           |
           
    """
    
    # determine target quadrants by sign of platform location
    bool_sign = np.array([np.array_equal(sign_quadrant, np.sign(xy_i)) for xy_i in xy[:]])
    
    ss = tls.determine_start_end_of_blobs(bool_sign)
    t_target_quadrant = tls.time_in_blobs(ss, t)

    return t_target_quadrant


def mean_speed(
    xy,
    t):
    s = tls.speed(xy, t)
    
    # as of now, t must be equally spaced
    diff_t = np.diff(t)
    if np.allclose(diff_t, np.mean(diff_t)):
        mean_s = np.mean(s)
    else:
        raise NotImplementedError('t needs to be equally spaced')
        
    return mean_s


def mean_acceleration(
    xy,
    t):
    a = tls.acceleration(xy, t)
    
    # as of now, t must be equally spaced
    diff_t = np.diff(t)
    if np.allclose(diff_t, np.mean(diff_t)):
        mean_a = np.mean(a)
    else:
        raise NotImplementedError('t needs to be equally spaced')
        
    return mean_a


def surface_coverage(
    xy,
    radius,
    dx,
    l_kernel):
    
    bins = [
        np.arange(-radius, radius, dx),
        np.arange(-radius, radius, dx)]
    a = np.ones(xy.shape[0])
    c_map = tls.create_value_map(a, xy, bins, np.mean)
    pix_kernel = int(l_kernel/dx)
    kernel = np.ones((pix_kernel, pix_kernel))
    c_map = signal.convolve2d(c_map, kernel)
    # get area 
    area_c = np.sum(c_map>0)*(dx**2)
    return area_c


def time_in_corridor(
    xy,
    t,
    xy_start, 
    xy_end,
    corr_width):
    
    bool_corr = tls.bool_in_corridor(
        xy, xy_start, xy_end, corr_width)
    ss = tls.determine_start_end_of_blobs(bool_corr)
    t_corr = tls.time_in_blobs(ss, t)

    return t_corr