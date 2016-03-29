#
import numpy
from astropy import wcs
from astropy.io import fits
import astropy.units as u

def main(RA, Dec, RA_bound, Dec_bound, pix_res=0.5*u.arcsec):
    '''for a set of points (RA,Dec), tell whether or not they are
        within the footprint of a region defined by polygon boundaries RA_bound, Dec_bound
        pix_res: pixel size of world coordinate system used as intermediary

        for a similar task, see
        http://pyregion.readthedocs.org/en/latest/users/overview.html#use-regions-for-spatial-filtering
    '''
    
    # TODO: some unit-checking here

    for coord in [RA, RA_bound]:
        assert (coord.max()<=360.0 and coord.min()>=0.0), "Problem with RA inputs: not in range 0 to 360"
    for coord in [Dec, Dec_bound]:
        assert (coord.max()<=+90.0 and coord.min()>=-90.0), "Problem with Dec inputs: not in range -90 to 90"
        
    # create a WCS based on polygon boundaries
    # first find how big the WCS has to be in degrees
    RA_range = RA_bound.max()-RA_bound.min()
    Dec_range = Dec_bound.max()-Dec_bound.min()

    # and how big it would be in pixels
    pix_res_deg = pix_res.to(u.deg).value
    RA_npix = int(RA_range/pix_res_deg)
    Dec_npix = int(Dec_range/pix_res_deg)

    # then find the centre
    pix_x_cen = 0.5*RA_npix
    pix_y_cen = 0.5*Dec_npix
    
    RA_cen = RA_bound.min() + 0.5*RA_range
    Dec_cen = Dec_bound.min() + 0.5*Dec_range

    # modified from http://docs.astropy.org/en/stable/wcs/#building-a-wcs-structure-programmatically
    # Create a new WCS object.  
    w = wcs.WCS(naxis=2)

    w.wcs.crpix = [pix_x_cen, pix_y_cen]
    w.wcs.cdelt = [0-pix_res_deg, pix_res_deg]
    w.wcs.crval = [RA_cen, Dec_cen]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    # assume rotation angle gets set to zero by default?
    
    # figure out the (x,y) positions of boundaries using WCS
    bound_xy = w.wcs_world2pix(RA_bound, Dec_bound, 1)

    # input format point_in-poly wants    
    poly_xy = numpy.column_stack((bound_xy[0],bound_xy[1])) 

    # sort the (x,y) positions according to position angle, so we have a closed polygon (may not be necessary) 
    # inspired by http://stackoverflow.com/questions/10846431/ordering-shuffled-points-that-can-be-joined-to-form-a-polygon-in-python
    pangle = numpy.arctan2(bound_xy[1]-pix_y_cen,bound_xy[0]-pix_x_cen)
    ang_sort = numpy.argsort(pangle)
        
    # transform the catalog RA, Dec to (x,y) positions using WCS
    cat_xy = w.wcs_world2pix(RA, Dec, 1)
    
    # see if catalog points are within boundaries, return Boolean indices
    return(point_in_poly(cat_xy[0], cat_xy[1], poly_xy[ang_sort]))


# from https://github.com/dstndstn/astrometry.net/blob/master/util/miscutils.py#L443
def point_in_poly(x, y, poly):
    '''
    Performs a point-in-polygon test for numpy arrays of *x* and *y*
    values, and a polygon described as 2-d numpy array (with shape (N,2))
    poly: N x 2 array
    Returns a numpy array of bools.
    '''
    x = numpy.atleast_1d(x)
    y = numpy.atleast_1d(y)
    inside = numpy.zeros(x.shape, bool)
    # This does a winding test -- count how many times a horizontal ray
    # from (-inf,y) to (x,y) crosses the boundary.
    for i in range(len(poly)):
        j = (i-1 + len(poly)) % len(poly)
        xi,xj = poly[i,0], poly[j,0]
        yi,yj = poly[i,1], poly[j,1]

        if yi == yj:
            continue

        I = numpy.logical_and(
            numpy.logical_or(numpy.logical_and(yi <= y, y < yj),
                          numpy.logical_and(yj <= y, y < yi)),
            x < (xi + ((xj - xi) * (y - yi) / (yj - yi))))
        inside[I] = numpy.logical_not(inside[I])
    return inside
