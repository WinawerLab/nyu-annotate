################################################################################
# proc/ventral.py
#
# Code for processing the contours of the ventral cortex into labels.

"""The ventral contours processing workflow.

This file contains the processing workflow for the ventral cortical contours, as
drawn for the NYUNEI dataset visual cortex annotation project. The code in this
file converts a set of contours into a set of labels.
"""


# Dependencies #################################################################

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import neuropythy as ny
import pimms

from .util import (
    cross_isect_2D,
    iscloser,
    fix_polygon,
    contour_endangle,
    order_nearness,
    dedup_points,
    find_crossings,
    extend_contour)


# Ventral Processing ###########################################################

# The calculations -------------------------------------------------------------
def _proc_outers(hv4_outer, vo_outer, v3v_contour):
    # Both the hV4-VO1 contour need to be re-ordered to be starting near
    # the V3-ventral contour and ending far from it. We can also remove
    # duplicate points while we're at it.
    hv4_outer = order_nearness(dedup_points(hv4_outer), v3v_contour[:,-1])
    vo_outer = order_nearness(dedup_points(vo_outer), v3v_contour[:,0])
    # If the two contours have 1 intersection, that is their new endpoint.
    (hii, vii, pts) = cross_isect_2D(hv4_outer, vo_outer)
    while len(hii) > 1:
        # Trim one point off of each end and try again.
        hv4_outer = hv4_outer[:,:-1]
        vo_outer = vo_outer[:,:-1]
        (hii, vii, pts) = cross_isect_2D(hv4_outer, vo_outer)
    if len(hii) == 1:
        # This is their new endpoint.
        hv4_outer = np.hstack([hv4_outer[:, :hii[0]+1], pts])
        vo_outer = np.hstack([vo_outer[:, :vii[0]+1], pts])
    else:
        # Check whether the vector from end to end is pointing in the wrong
        # direction---if so, we need to cut ends off.
        while contour_endangle(hv4_outer, vo_outer[:,-1]) < 0:
            hv4_outer = hv4_outer[:,:-1]
            vo_outer = vo_outer[:,:-1]
        u = 0.5*(hv4_outer[:,-1] + vo_outer[:,-1])
        hv4_outer = np.hstack([hv4_outer, u[:,None]])
        vo_outer = np.hstack([vo_outer, u[:,None]])
    # These can now be turned into an outer boundary that we start and end
    # and the V3v tip.
    outer_bound = np.hstack(
        [np.fliplr(v3v_contour), vo_outer[:, :-1],
         np.fliplr(hv4_outer), v3v_contour[:,[-1]]])
    outer_sources = np.concatenate(
        [['V3v']*v3v_contour.shape[1], ['VO_outer']*(vo_outer.shape[1] - 1),
         ['hV4_outer']*(hv4_outer.shape[1] + 1)])
    return (hv4_outer, vo_outer, outer_bound, outer_sources)
def calc_extended_contours(chirality, contours):
    """Creates an extended versions of the ventral contours.
    
    The contours are extended by adding points to either end that are a 
    distance of 100 map units from the respective endpoints at an angle of 180
    degrees to the immediate-next interior point.
    
    Parameters
    ----------
    chirality : 'lh' or 'rh'
        The hemisphere chirality.
    contours : dict-like
        The dictionary of contours.
    
    Returns
    -------
    preproc_contour : dict
        A dictionary whose keys are the same as in `contours` but that have
        been mildly preprocessed: all contours are ordered starting from the
        end nearest the V3-ventral boundary. Additionally the `'outer'` contour
        has been added, consisting of the hV4-outer, V3-ventral, and VO-outer
        contours conjoined, starting at the hV4-outer and ending with the
        VO-outer.
    ext_contours : dict
        A dictionary whose keys are the same as in `preproc_contour` but whose
        ends have been extended by 100 map units in the same directoin as the
        ending segments.
    outer_sources : array of str
        A list whose elements give the origin of each point in the `'outer'`
        contour. Each element will be either `'hV4_outer'`, `'VO_outer'`, or
        `'V3v'`.
    """
    # The code that follows expects V3v to be periphery -> fovea for historical
    # reasons (all code after the next line).
    contours = dict(contours)
    (hv4_preproc, vo_preproc, outbound, outer_sources) = _proc_outers(
        contours['hV4_outer'],
        contours['VO_outer'],
        np.fliplr(contours['V3_ventral']))
    contours['hV4_outer'] = hv4_preproc
    contours['VO_outer'] = vo_preproc
    contours['V3_ventral'] = contours['V3_ventral']
    contours['outer'] = outbound
    # Now make the extended contours:
    ext_contours = {k:extend_contour(c) for (k,c) in contours.items()}
    # And return!
    return (contours, ext_contours, outer_sources)
def _calc_normcontours(hv4_vo1, vo1_vo2, outer, chirality):
    # The extended vo1/vo2 and hv4/vo contours should each intersect this outer
    # boundary twice. We subdivide the outer boundary into top and bottom pieces
    # for each of the crossing contours.
    cp0 = -1 if chirality == 'rh' else 1
    crossings = {'hV4-VO1': hv4_vo1, 'VO1-VO2': vo1_vo2}
    outer_work = outer
    pieces = []
    for name in ('hV4-VO1', 'VO1-VO2'):
        cross = crossings[name]
        # First, make sure that cross is pointing in the right direction.
        (cii, oii, pts) = find_crossings(cross, outer)
        if len(cii) != 2:
            raise RuntimeError(
                f"{len(cii)} {name} / Outer intersections")
        if oii[0] > oii[1]:
            cross = np.fliplr(cross)
        # Next, find the crossings with the actual working outer boundary.
        (cii, oii, pts) = find_crossings(cross, outer_work)
        if len(cii) != 2:
            raise RuntimeError(
                f"{len(cii)} {name} / Working-Outer intersections")
        # We can now put together the upper and lower pieces. First, roll the
        # outer_work matrix so that the intersection is between the last two
        # columns of the matrix.
        outer_work = np.roll(outer_work[:,:-1], -oii[0] - 1, axis=1)
        outer_work = np.hstack([outer_work, outer_work[:,[0]]])
        (cii, oii, pts) = find_crossings(cross, outer_work)
        # Now build the pieces.
        cross = np.hstack(
            [pts[:, [0]], cross[:, cii[0]+1:cii[1]+1], pts[:,[1]]])
        crossings[name] = cross
        upper = np.hstack(
            [cross, outer_work[:, oii[1]+1:-1], pts[:,[0]]])
        pieces.append(upper)
        outer_work = np.hstack(
            [pts[:,[0]], outer_work[:, :oii[1]+1], np.fliplr(cross)])
    (hv4_b, vo1_b) = pieces
    vo2_b = outer_work
    hv4_vo1_norm = crossings['hV4-VO1']
    vo1_vo2_norm = crossings['VO1-VO2']
    contours = {
        'VO1_VO2': vo1_vo2_norm,
        'hV4_VO1': hv4_vo1_norm,
        'outer':   outer}
    bounds = {'hV4': hv4_b, 'VO1': vo1_b, 'VO2': vo2_b}
    return (contours, bounds)
def calc_normalized_contours(chirality, preproc_contours, ext_contours,
                             outer_sources):
    """Normalizes the raw contours and converts them into path-traces.
    """
    hv4_vo1 = ext_contours['hV4_VO1']
    vo1_vo2 = ext_contours['VO1_VO2']
    outer = preproc_contours['outer']
    (contours, bounds) = _calc_normcontours(
        hv4_vo1, vo1_vo2, outer, chirality)
    if chirality == 'lh':
        # Make the boundaries counter-clockwise.
        bounds = {k: np.fliplr(v) for (k,v) in bounds.items()}
    bounds = {k: fix_polygon(b) for (k,b) in bounds.items()}
    contours['hV4_outer'] = preproc_contours['hV4_outer']
    contours['VO_outer'] = preproc_contours['VO_outer']
    contours['V3_ventral'] = preproc_contours['V3_ventral']
    return (contours, bounds)
def proc(chirality, hv4_outer, vo_outer, hv4_vo1, vo1_vo2, v3_ventral):
    """Process a set of ventral contours into ventral area boundary traces.
    """
    contours = dict(
        hV4_outer=hv4_outer.T,
        VO_outer=vo_outer.T,
        hV4_VO1=hv4_vo1.T,
        VO1_VO2=vo1_vo2.T,
        V3_ventral=v3_ventral.T)
    (preproc_contours, ext_contours, outer_sources) = calc_extended_contours(
        chirality,
        contours)
    (contours, bounds) = calc_normalized_contours(
        chirality,
        preproc_contours,
        ext_contours,
        outer_sources)
    return bounds
