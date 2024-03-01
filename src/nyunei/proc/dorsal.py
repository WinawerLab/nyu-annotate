################################################################################
# proc/dorsal.py
#
# Pipeline for processing subjects and saving/loading the outputs to disk.

"""The dorsal contours processing workflow.

This file contains the processing workflow for the dorsal cortical contours, as
drawn for the HCP visual cortex annotation project. The code in this file
converts a set of dorsal contours into visual area labels.
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
    extend_contour,
    seg_nearest)


# Dorsal Processing ###########################################################

# The calculations -------------------------------------------------------------
def calc_extended_contours(chirality, contours):
    """Creates and extends the dorsal boundary contour.
    
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
        A dictionary whose keys are the same as in `contours` but that have been
        mildly preprocessed: contours `IPS0_outer` and `V3ab_outer` are ordered
        starting from the most anterior end, `LO1_outer` is ordered from
        posterior to anterior, and the V3-dorsal contour is ordered from
        posterior end to anterior end. Additionally, the contour `V3ab_inner` is
        ordered from its most posterior/ventral end to its most dorsal end.
    ext_contours : dict
        A dictionary whose keys are the same as in `preproc_contour` but whose
        ends have been extended by 100 map units in the same directoin as the
        ending segments.
    """
    contours = dict(contours)
    v3d_contour = np.fliplr(contours.pop('V3_dorsal'))
    # V3ab and LO1 must be in the right order due to their fixed starting
    # positions. V3d is also in the right order. IPS0 could be in the wrong
    # order, though, so check that.
    contours['IPS0_outer'] = order_nearness(
        contours['IPS0_outer'],
        v3d_contour[:,-1])
    # Make sure V3ab_inner is ordered starting from the end closest to the
    # V3_dorsal contour.
    contours['V3ab_inner'] = order_nearness(
        contours['V3ab_inner'],
        v3d_contour[:,0])
    # Now make the extended contours:
    ext_contours = {k:extend_contour(c) for (k,c) in contours.items()}
    # For V3ab and LO1 we don't want the front extended.
    ext_contours['V3ab_outer'] = ext_contours['V3ab_outer'][:, 1:]
    ext_contours['LO1_outer'] = ext_contours['LO1_outer'][:, 1:]
    # Add V3d in as well.
    contours['V3_dorsal'] = v3d_contour
    # And return!
    return (contours, ext_contours)
def calc_normalized_contours(chirality, preproc_contours, ext_contours):
    """Normalizes the raw contours and converts them into path-traces.
    """
    v3ab_outer = ext_contours['V3ab_outer']
    v3ab_inner = ext_contours['V3ab_inner']
    ips0_outer = ext_contours['IPS0_outer']
    lo1_outer  = ext_contours['LO1_outer']
    v3d = preproc_contours['V3_dorsal']
    # Start with the V3ab_outer contour:                                                                                                                         
    # It starts at the end of V3d and intersects V3d somewhere; find that                                                                                        
    # intersection first.                                                                                                                                        
    (ii_v3ab, ii_v3d, pts) = cross_isect_2D(v3ab_outer, v3d[:,:-1])
    if len(ii_v3ab) == 2 and ii_v3ab[0] == 0:
        # We can fix this case--the rater just clicked on the initial point.
        ii_v3ab = [ii_v3ab[1]]
        ii_v3d = [ii_v3d[1]]
        pts = pts[:,[1]]
    elif len(ii_v3ab) == 0:
        # We just use the closest point instead. We use v3d[:,-2] to avoid the
        # contour extension.
        v3d_segs = (v3d[:,:-1], v3d[:,1:])
        (pts, ii_v3d) = seg_nearest(v3d_segs, v3ab_outer[:,-2], argmin=1)
        ii_v3ab = [v3ab_outer.shape[1] - 2]
    elif len(ii_v3ab) != 1:
        raise RuntimeError(f"{len(ii_v3ab)} V3ab-outer/V3-dorsal intersections")
    # We can make the V3ab boundary from this.                                                                                                                   
    v3ab_outer = np.hstack([v3ab_outer[:, :ii_v3ab[0]+1], pts])
    v3ab_bound = np.hstack([v3ab_outer, pts, v3d[:, ii_v3d[0]+1:]])
    # IPS0 has to intersect with V3ab in 2 places; easiest way here is to find
    # the closest points to the start and end. The ips0_outer contour is from
    # the ext_contours, so we clip it first.
    ips0_outer = ips0_outer[:, 1:-1]
    v3ab_segs = (v3ab_outer[:, :-1], v3ab_outer[:, 1:])
    (start_ips0, ii0_v3ab) = seg_nearest(v3ab_segs, ips0_outer[:,0], argmin=1)
    (end_ips0, ii1_v3ab) = seg_nearest(v3ab_segs, ips0_outer[:,-1], argmin=1)
    ii_v3ab = [int(ii0_v3ab.item()), int(ii1_v3ab.item())]
    if ii_v3ab[0] > ii_v3ab[1]:
        ips0_outer = np.fliplr(ips0_outer)
        (start_ips0, end_ips0) = (end_ips0, start_ips0)
        ii_v3ab = [ii_v3ab[1], ii_v3ab[0]]
    ips0_outer = np.hstack(
        [start_ips0, ips0_outer[:,1:-1], end_ips0])
    # If there are intersections between ips0_outer and v3ab_outer now, we want
    # to use those.
    (ii_ips0, ii_v3abx, pts) = find_crossings(ips0_outer, v3ab_outer)
    if len(ii_ips0) == 1:
        # We keep whichever segment is longer.
        if ii_ips0[0] > ips0_outer.shape[1] - ii_ips0[1] - 1:
            # Keep the segment from the intersection to the end
            ips0_outer = np.hstack([pts, ips0_outer[:, ii_ips0[0]+1:]])
            ii_v3ab[0] = ii_v3ab[0]
        else:
            # Keep the segment from the beginning to the intersection.
            ips0_outer = np.hstack([ips0_outer[:, :ii_ips0[0]+1], pts])
            ii_v3ab[1] = ii_v3ab[1]
    elif len(ii_ips0) == 2:
        # We keep the middle section between the two intersections.
        ips0_outer = np.hstack(
            [pts[:,[0]],
             ips0_outer[:, ii_ips0[0]+1:ii_ips0[1]+1],
             pts[:, [1]]])
        ii_v3ab = ii_v3abx
    elif len(ii_ips0) != 0:
        raise RuntimeError(f"{len(ii_ips0)} IPS0-outer/V3ab-outer intersections")
    ips0_bound = np.hstack(
        [ips0_outer, np.fliplr(v3ab_outer[:, ii_v3ab[0]+1:ii_v3ab[1]+1])])
    # LO1 should intersect the outer of these boundaries.
    outer = np.hstack(
        [v3ab_outer[:, :ii_v3ab[0]+1],
         ips0_outer,
         v3ab_outer[:, ii_v3ab[1]+1:]])
    (ii_lo1, ii_out, pts) = find_crossings(lo1_outer, outer)
    if len(ii_lo1) < 1:
        ii_lo1 = [lo1_outer.shape[1] - 2]
        outer_seg = (outer[:,:-1], outer[:,1:])
        # We use -2 to avoid the extended contour.
        (pts, ii_out) = seg_nearest(outer_seg, lo1_outer[:,-2], argmin=1)
        ii_out = [int(ii_out.item())]
    # We just want the first of these; we ignore the rest.                                                                                                       
    (ii_lo1, ii_out) = (ii_lo1[0], ii_out[0])
    lo1_outer = np.hstack([lo1_outer[:, :ii_lo1+1], pts[:,[0]]])
    lo1_bound = np.hstack(
        [v3d[:, :ii_v3d[0]+1],
         np.fliplr(outer[:, ii_out+1:]),
         np.fliplr(lo1_outer)])
    # Last, we just need to divide V3ab up into V3a and V3b.                                                                                                     
    (ii_in, ii_ab, pts) = find_crossings(v3ab_inner, v3ab_bound)
    if len(ii_in) != 2:
        raise RuntimeError(f"{len(ii_in)} V3ab-outer/V3ab-inner intersections")
    # Check to see if the V3ab inner contours is backward.
    if iscloser(np.mean(ips0_outer, 1), pts[:,0], pts[:,1]):
        v3ab_inner = np.fliplr(v3ab_inner)
        (ii_in, ii_ab, pts) = find_crossings(v3ab_inner, v3ab_bound)
    v3ab_inner = np.hstack(
        [pts[:,[0]], v3ab_inner[:, ii_in[0]+1:ii_in[1]+1], pts[:,[1]]])
    if ii_ab[0] < ii_ab[1]:
        v3a_bound = np.hstack(
            [v3ab_inner[:, [0]],
             v3ab_bound[:, ii_ab[0]+1:ii_ab[1]+1],
             np.fliplr(v3ab_inner[:, 1:])])
        v3b_bound = np.hstack(
            [v3ab_inner,
             v3ab_bound[:, ii_ab[1]+1:],
             v3ab_bound[:, :ii_ab[0]+1]])
    else:
        v3a_bound = np.hstack(
            [v3ab_inner[:, [0]],
             v3ab_bound[:, ii_ab[0]+1:],
             v3ab_bound[:, :ii_ab[1]+1],
             np.fliplr(v3ab_inner[:, 1:])])
        v3b_bound = np.hstack(
            [v3ab_inner,
             v3ab_bound[:, ii_ab[1]+1:ii_ab[0]+1]])
    # Put these all together and return.                                                                                                                         
    contours = {}
    contours['V3ab_outer'] = v3ab_outer
    contours['V3ab_inner'] = v3ab_inner
    contours['IPS0_outer'] = ips0_outer
    contours['LO1_outer'] = lo1_outer
    bounds = {
        'V3a': v3a_bound,
        'V3b': v3b_bound,
        'IPS0': ips0_bound,
        'LO1': lo1_bound}
    if chirality == 'lh':
        # Make the boundaries counter-clockwise.                                                                                                                 
        bounds = {k: np.fliplr(v) for (k,v) in bounds.items()}
    bounds = {k: fix_polygon(b) for (k,b) in bounds.items()}
    return (contours, bounds)
def proc(chirality, v3ab_outer, v3ab_inner, ips0_outer, lo1_outer, v3_dorsal):
    """Process a set of dorsal contours into ventral area boundary traces.
    """
    contours = dict(
        V3ab_outer=v3ab_outer.T,
        V3ab_inner=v3ab_inner.T,
        IPS0_outer=ips0_outer.T,
        LO1_outer=lo1_outer.T,
        V3_dorsal=np.fliplr(v3_dorsal.T))
    (preproc_contours, ext_contours) = calc_extended_contours(
        chirality,
        contours)
    (contours, bounds) = calc_normalized_contours(
        chirality,
        preproc_contours,
        ext_contours)
    return bounds
