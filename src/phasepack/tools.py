import numpy as np
from scipy.fftpack import fftshift, ifftshift

# Try and use the faster Fourier transform functions from the pyfftw module if
# available
try:
    from pyfftw.interfaces.scipy_fftpack import fft2, ifft2
# Otherwise use the normal scipy fftpack ones instead (~2-3x slower!)
except ImportError:
    import warnings
    warnings.warn("""
Module 'pyfftw' (FFTW Python bindings) could not be imported. To install it, try
running 'pip install pyfftw' from the terminal. Falling back on the slower
'fftpack' module for 2D Fourier transforms.""")
    from scipy.fftpack import fft2, ifft2


def lowpassfilter(size, cutoff, n):
    """
    Constructs a low-pass Butterworth filter:

        f = 1 / (1 + (w/cutoff)^2n)

    usage:  f = lowpassfilter(sze, cutoff, n)

    where:  size    is a tuple specifying the size of filter to construct
            [rows cols].
        cutoff  is the cutoff frequency of the filter 0 - 0.5
        n   is the order of the filter, the higher n is the sharper
            the transition is. (n must be an integer >= 1). Note
            that n is doubled so that it is always an even integer.

    The frequency origin of the returned filter is at the corners.
    """

    if cutoff < 0. or cutoff > 0.5:
        raise Exception('cutoff must be between 0 and 0.5')
    elif n % 1:
        raise Exception('n must be an integer >= 1')
    if len(size) == 1:
        rows = cols = size
    else:
        rows, cols = size

    if (cols % 2):
        xvals = np.arange(-(cols - 1) / 2.,
                          ((cols - 1) / 2.) + 1) / float(cols - 1)
    else:
        xvals = np.arange(-cols / 2., cols / 2.) / float(cols)

    if (rows % 2):
        yvals = np.arange(-(rows - 1) / 2.,
                          ((rows - 1) / 2.) + 1) / float(rows - 1)
    else:
        yvals = np.arange(-rows / 2., rows / 2.) / float(rows)

    x, y = np.meshgrid(xvals, yvals, sparse=True)
    radius = np.sqrt(x * x + y * y)

    return ifftshift(1. / (1. + (radius / cutoff) ** (2. * n)))


def filtergrids(rows, cols):

    # Set up X and Y spatial frequency matrices, fx and fy, with ranges
    # normalised to +/- 0.5 The following code adjusts things appropriately for
    # odd and even values of rows and columns so that the 0 frequency point is
    # placed appropriately.
    if cols%2 == 1:
        fxrange = np.arange(-(cols-1)/2, (cols+1)/2)/cols
    else:
        fxrange = np.arange(-cols/2,(cols/2))/cols

    if rows%2 == 1:
        fyrange = np.arange(-(rows-1)/2,(rows+1)/2)/rows
    else:
        fyrange = np.arange(-rows/2,(rows/2))/rows

    fx, fy = np.meshgrid(fxrange, fyrange)

    # Quadrant shift so that filters are constructed with 0 frequency at
    # the corners
    fx = ifftshift(fx)
    fy = ifftshift(fy)

    # Construct spatial frequency values in terms of normalised radius from
    # centre.
    f = np.sqrt(fx**2 + fy**2)
    return f, fx, fy


def monogenicfilters(rows, cols):

    f, fx, fy = filtergrids(rows, cols)
    f[0,0] = 1  # Set DC value to 1 to avoid divide by zero

    H1 = 1j*fx/f
    H2 = 1j*fy/f

    H1[0,0] = 0  # Restore 0 DC value
    H2[0,0] = 0
    f[0,0] = 0
    return H1, H2, f

def highpassmonogenic(img, maxwavelength, n=2):

    IMG = fft2(img)

    # Generate monogenic and filter grids
    H1, H2, freq = monogenicfilters(img.shape[0], img.shape[1])

    H =  1.0 - 1.0 / (1.0 + (freq * maxwavelength)**(2*n))
    f = np.real(ifft2(H*IMG))
    h1f = np.real(ifft2(H*H1*IMG))
    h2f = np.real(ifft2(H*H2*IMG))

    phase = np.arctan(f/np.sqrt(h1f**2 + h2f**2 + np.finfo(float).eps))
    orient = np.arctan2(h2f, h1f)
    E = np.sqrt(f**2 + h1f**2 + h2f**2)

    return phase, orient, E


def histtruncate(img, lHistCut, uHistCut):

    if lHistCut < 0 or lHistCut > 100 or uHistCut < 0 or uHistCut > 100:
        raise("Histogram truncation values must be between 0 and 100")

    if np.ndim(img) > 2:
        raise("histtruncate only defined for grey scale images")

    newimg = img.copy()
    sortv = np.sort(newimg.ravel())   # Generate a sorted array of pixel values.

    # Any NaN values will end up at the end of the sorted list. We
    # need to ignore these.
#    N = sum(.!isnan.(sortv))  # Number of non NaN values. v0.6
    # N = sum(broadcast(!,isnan.(sortv)))  # compatibity for v0.5 and v0.6
    N = np.sum(np.invert(np.isnan(sortv)))

    # Compute indicies corresponding to specified upper and lower fractions
    # of the histogram.
    lind = np.floor(1 + N*lHistCut/100) #floor(Int, 1 + N*lHistCut/100)
    hind = np.ceil(N - N*uHistCut/100) # ceil(Int, N - N*uHistCut/100)

    low_val  = sortv[int(lind)]
    high_val = sortv[int(hind)]

    # Adjust image
    newimg[newimg < low_val] = low_val
    newimg[newimg > high_val] = high_val

    return newimg


def ppdrc(img, wavelength, clip=0.01, n=2):

    ph, _, E = highpassmonogenic(img, wavelength, n)

    # Construct each dynamic range reduced image
    # dimg = Vector{Array{Float64,2}}(undef, nscale)

    dimg = histtruncate(np.sin(ph)*np.log(1+E), clip, clip)

    return dimg

    # if nscale == 1   # Single image, highpassmonogenic() will have returned single
    #                  # images, hence this separate case
    #     dimg[1] = histtruncate(sin.(ph).*log1p.(E), clip, clip)

    # else             # ph and E will be arrays of 2D arrays
    #     range = zeros(nscale,1)
    #     for k = 1:nscale
    #         dimg[k] = histtruncate(sin.(ph[k]).*log1p.(E[k]), clip, clip)
    #         range[k] = maximum(abs.(dimg[k]))
    #     end

    #     maxrange = maximum(range)
    #     # Set the first two pixels of each image to +range and -range so that
    #     # when the sequence of images are displayed together, say using linimix(),
    #     # there are no unexpected overall brightness changes
    #     for k = 1:nscale
    #         dimg[k][1] =  maxrange
    #         dimg[k][2] = -maxrange
    #     end
    # end




def rayleighmode(data, nbins=50):
    """
    Computes mode of a vector/matrix of data that is assumed to come from a
    Rayleigh distribution.

    usage:  rmode = rayleighmode(data, nbins)

    where:  data    data assumed to come from a Rayleigh distribution
            nbins   optional number of bins to use when forming histogram
                    of the data to determine the mode.

    Mode is computed by forming a histogram of the data over 50 bins and then
    finding the maximum value in the histogram. Mean and standard deviation
    can then be calculated from the mode as they are related by fixed
    constants.

        mean = mode * sqrt(pi/2)
        std dev = mode * sqrt((4-pi)/2)

    See:
        <http://mathworld.wolfram.com/RayleighDistribution.html>
        <http://en.wikipedia.org/wiki/Rayleigh_distribution>
    """
    n, edges = np.histogram(data, nbins)
    ind = np.argmax(n)
    return (edges[ind] + edges[ind + 1]) / 2.


def perfft2(im, compute_P=True, compute_spatial=False):
    """
    Moisan's Periodic plus Smooth Image Decomposition. The image is
    decomposed into two parts:

        im = s + p

    where 's' is the 'smooth' component with mean 0, and 'p' is the 'periodic'
    component which has no sharp discontinuities when one moves cyclically
    across the image boundaries.

    useage: S, [P, s, p] = perfft2(im)

    where:  im      is the image
            S       is the FFT of the smooth component
            P       is the FFT of the periodic component, returned if
                    compute_P (default)
            s & p   are the smooth and periodic components in the spatial
                    domain, returned if compute_spatial

    By default this function returns `P` and `S`, the FFTs of the periodic and
    smooth components respectively. If `compute_spatial=True`, the spatial
    domain components 'p' and 's' are also computed.

    This code is adapted from Lionel Moisan's Scilab function 'perdecomp.sci'
    "Periodic plus Smooth Image Decomposition" 07/2012 available at:

        <http://www.mi.parisdescartes.fr/~moisan/p+s>
    """

    if im.dtype not in ['float32', 'float64']:
        im = np.float64(im)

    rows, cols = im.shape

    # Compute the boundary image which is equal to the image discontinuity
    # values across the boundaries at the edges and is 0 elsewhere
    s = np.zeros_like(im)
    s[0, :] = im[0, :] - im[-1, :]
    s[-1, :] = -s[0, :]
    s[:, 0] = s[:, 0] + im[:, 0] - im[:, -1]
    s[:, -1] = s[:, -1] - im[:, 0] + im[:, -1]

    # Generate grid upon which to compute the filter for the boundary image
    # in the frequency domain.  Note that cos is cyclic hence the grid
    # values can range from 0 .. 2*pi rather than 0 .. pi and then pi .. 0
    x, y = (2 * np.pi * np.arange(0, v) / float(v) for v in (cols, rows))
    cx, cy = np.meshgrid(x, y)

    denom = (2. * (2. - np.cos(cx) - np.cos(cy)))
    denom[0, 0] = 1.     # avoid / 0

    S = fft2(s) / denom
    S[0, 0] = 0      # enforce zero mean

    if compute_P or compute_spatial:

        P = fft2(im) - S

        if compute_spatial:
            s = ifft2(S).real
            p = im - s

            return S, P, s, p
        else:
            return S, P
    else:
        return S
