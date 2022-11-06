import tools as tls
import pytest
import numpy as np
    
def test_remove_trailing_positions_on_platform():
    xy = np.array([
        [0, 0],
        [1, 1],
        [2, 2],
        [2, 2],
        [2, 2],
    ])
    
    t = np.linspace(0, 1, len(xy))
    xy_pltfrm = np.array([1.9, 1.9])
    radius_pltfrm = 0.2
    
    xy_expected = np.array([
            [0, 0],
            [1, 1],
        ])
    t_expected = t[:-3]
    ss_expected = np.array(
        [[2, 5]])
    
    xy_test, t_test, ss_test = tls.remove_trailing_positions_on_platform(
        xy,
        t,
        xy_pltfrm,
        radius_pltfrm,
        return_pltfrm_crossings=True)

    assert np.array_equal(xy_expected, xy_test)
    assert np.array_equal(t_expected, t_test)
    assert np.array_equal(ss_expected, ss_test)


def test_detect_platform_crossings():
    
    # Basic function
    # -----------------
    
    xy = np.array([
        [0, 0],
        [1, 1],
        [0, 0],
        [1.1, 0.9],
        [1, 1],
    ])
    
    t = np.linspace(0, 1, len(xy))
    xy_pltfrm = np.array([1, 1])
    radius_pltfrm = 0.2

    cross_expected = np.array(
        [[1, 2],
         [3, 5],
        ])
    cross_test = tls.detect_platform_crossings(
        xy,
        t,
        xy_pltfrm,
        radius_pltfrm)
    assert np.array_equal(cross_expected, cross_test)
    

def test_in_annulus():

    xy = np.array([
        [0, 0],
        [1, 0],
        [2, 0],
    ])
    xy_annulus = np.array([0, 0])
    radius_outer = 1.5
    radius_inner = 0.5
    
    bool_annulus_exp = np.array([False, True, False])
    
    bool_annulus_test = tls.in_annulus(
        xy,
        xy_annulus,
        radius_outer=radius_outer,
        radius_inner=radius_inner)
    
    assert np.array_equal(bool_annulus_exp, bool_annulus_test)
    
    
def test_speed():
    xy = np.array([
        [0, 0],
        [1, 0],
        [2, 0],
        [2, -1],
        [2, -2],
    ])
    t = np.array([0, 1, 2, 3, 4])
    
    speed_exp = np.array([1, 1, 1, 1])
    speed_test = tls.speed(xy, t)
    
    assert np.allclose(speed_exp, speed_test)

    
def test_acceleration():
    xy = np.array([
        [0, 0],
        [2, 0],
        [3, 0],
        [4, 0],
    ])
    t = np.array([0, 1, 2, 3])
    
    speed_exp = np.array([2, 1, 1])
    a_exp = np.array([-1, 0])
    a_test = tls.acceleration(xy, t)
    
    assert np.allclose(a_exp, a_test)
    