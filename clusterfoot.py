#
import numpy
from astropy import wcs
from astropy.table import Table
import astropy.units as u
from scipy.interpolate import interp1d
from skmonaco import mcquad
from scipy.integrate import quad

def main(datfile):
    '''read datafile where each line contains cluster name, centre, SB profile, footprint FOV
            and for each cluster, figure out what fraction of luminosity is within FOV
    '''
    
    # TODO: some error-checking here
    mydata = Table.read(datfile, format='ascii.commented_header')
    
    # loop over all the rows in the table
    for row in range(0,len(mydata)):
        ID, RA, dec = mydata['ID'][row], mydata['RA'][row], mydata['Dec'][row]
        sbprof_r, sbprof_ir = np.loadtxt(mydata['sb'][row], unpack=True, usecols=(0,1))
        footprint = np.loadtxt(mydata['foot'][row]) # TBD - what is the format?

        #transform footprint to xy coordinates with cluster centre at x=0, y=0
        foot_xy = coord_xform(RA, dec, footprint) 

        # define the function to be integrated: R*I(R)
        integrand_fn_r = interp1d(sbprof_r, sbprof_r*sbprof_ir, kind = 'cubic') 
        # compute the integral from R=0 to R=inf
        total_lum = quad(integrand_fn_r, 0, np.inf)

        # define another function, integrand_fn_r over the footprint but zero outside
        bd_integrd_fn_xy = make_bounded_integrand(integrand_fn_r, foot_xy)        
        # determine the limits of integration
        x0, y0 = foot_xy[0].min, foot_xy[1].min
        x1, y1 = foot_xy[0].max, foot_xy[1].max
        # integrate the function
        footprint_lum = mcquad(bd_integrd_fn_xy,npoints=100000, xl=[x0,y0], xu=[x1,y1])

        # compute fractional luminosity
        fract_lum = footprint_lum/total_lum
        
        print(ID, fract_lum)
        # done with loop over cluster

    return

def coord_xform(RA, Dec, footprint, pix_res=0.1*u.arcsec): # UNFINISHED - sort out footprint.ra vs footprint[0], centre shift, etc
    '''given RA, Dec, and footprint polygon in (RA, Dec) pairs
       return coordinates of footprint in Cartesian coordinates centered on RA, Dec
    '''

    # create a WCS based on polygon boundaries
    # first find how big the WCS has to be in degrees
    RA_range = footprint.ra.max()-footprint.ra.min()
    Dec_range = footprint.dec.max()-footprint.dec.min()

    # and how big it would be in pixels
    pix_res_deg = pix_res.to(u.deg).value
    RA_npix = int(RA_range/pix_res_deg)
    Dec_npix = int(Dec_range/pix_res_deg)

    # then find the centre
    pix_x_cen = 0.5*RA_npix
    pix_y_cen = 0.5*Dec_npix    
    RA_cen = footprint.ra.min() + 0.5*RA_range
    Dec_cen = footprint.dec.min() + 0.5*Dec_range

    # modified from http://docs.astropy.org/en/stable/wcs/#building-a-wcs-structure-programmatically
    # Create a new WCS object.  I assume that the rotation angle gets set to zero by default.
    w = wcs.WCS(naxis=2)
    w.wcs.crpix = [pix_x_cen, pix_y_cen]
    w.wcs.cdelt = [0-pix_res_deg, pix_res_deg]
    w.wcs.crval = [RA_cen, Dec_cen]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
   
    # transform the footprint RA, Dec to (x,y) positions using WCS
    foot_xy = w.wcs_world2pix(footprint.ra, footprint.dec, 1)

    # sort the (x,y) positions according to position angle, so we have a closed polygon (may not be necessary) 
    # inspired by http://stackoverflow.com/questions/10846431/ordering-shuffled-points-that-can-be-joined-to-form-a-polygon-in-python
    pangle = numpy.arctan2(foot_xy.y-pix_y_cen,foot_xy.x-pix_x_cen)
    ang_sort = numpy.argsort(pangle)

    # shift footprint so that it's centered at (0,0) rather than image centre
    #TBD
    
    # return footprint
    return(foot_xy[ang_sort])

def make_bounded_integrand(i_r, footprint):
    '''return a function(x,y) which computes i(sqrt(x**2+y**2)) if (x,y) inside footprint
       and returns 0 if (x,y) outside footprint
    '''
    def bound_int(x, y):
        if point_in_poly(x, y, footprint)[0]:
            r = np.sqrt(x**2+y**2)
            return(i_r(r))
        else:
            return(0)
     return(bound_int)


# from https://github.com/dstndstn/astrometry.net/blob/master/util/miscutils.py#L443
# not entirely well-tested
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

