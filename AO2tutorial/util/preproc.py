import numpy as np
from ccdproc import subtract_bias, flat_correct, gain_correct
from astropy.nddata import CCDData

from astropy.io import fits
import astropy.units as u
from astropy.stats import sigma_clip
from astropy.convolution import convolve, Box1DKernel
from numpy.polynomial.legendre import legfit, legval


__all__ = ['suboverscan',
           'make_master_bias', 'make_master_flat',
           'preproc']

#%%
def suboverscan(data, xbin, chip=2, ybox=5):
    ''' Subtracts overscan region from the input image
    TODO: Resolve -- currently only chip 2 is available & xbin=1 is not tested.
    TODO: Change so that it uses ccdproc after the release of 1.3

    Note
    ----
    Using overscan is not much beneficial than bias subtraction, since there 
    does exist some 2D pattern. See "Bias pattern" in the link:
        https://www.naoj.org/Observing/Instruments/FOCAS/ccdinfo.html
    
    Following is from 
        https://www.naoj.org/Observing/Instruments/FOCAS/ccdinfo.html
    updated on 2010-09-28.
    
    |                      | Chip2           |                    |                     |                     | Chip1           |                    |                     |                         |
    | -------------------- | --------------- | ------------------ | ------------------- | ------------------- | --------------- | ------------------ | ------------------- | ----------------------- |
    |                      | ch1             | ch2                | ch3                 | ch4                 | ch1             | ch2                | ch3                 | ch4                     |
    | gain (e/ADU)         | 2.105           | 1.968              | 1.999               | 1.918               | 2.081           | 2.047              | 2.111               | 2.087                   |
    | readout noise (e)    | 4.3(*1)         | 3.7                | 3.4                 | 3.6                 | 4.2(*1)         | 3.8                | 3.6                 | 4.0                     |
    | active area(*2)      | [9:520,49:4224] | [553:1064,49:4224] | [1081:1592,49:4224] | [1625:2136,49:4224] | [9:520,49:4224] | [553:1064,49:4224] | [1081:1592,49:4224] | [1626:2137,49:4224](*3) |
    | over-scan region(*2) | [521:536,*]     | [537:552,*]        | [1593:1608,*]       | [1609:1624,*]       | [521:536,*]     | [537:552,*]        | [1593:1608,*]       | [1610:1625,*](*3)       |

    (*1) Modification of grounding cables has reduced the readout noise of ch1.

    (*2) These values are for images without binning.
    
    (*3) There is an extra column at x=1609 of Chip1 which causes a shift in the
        position of ch4.
       if chip == 2:
            gain = [2.105, 1.968, 1.999, 1.918]
            ronoise = [4.3, 3.7, 3.4, 3.6]

    Parameters
    ----------
    ybox: int
        Binning factor to smooth the image along the Y-axis.

    Return
    ------
    ccdout: astropy.nddata.CCDData
        The overscan subtracted image (overscan subtracted but not trimmed).
    '''
#    data = fits.getdata(fname)
#    hdr = fits.getheader(fname)
#    ylen = int(hdr['naxis2'])
#    xbin = int(hdr['bin-fct1'])
#    chip = int(hdr['det-id'])
#    print('{:s} -- chip {:d}'.format(fname, chip))
#    if fname[:-5] != hdr['frameid']:
#        print('\tThe header name (FRAMEID) and file name not coherent.')
    
    ylen = np.shape(data)[0]
    
    if chip == 1:
        raise ValueError('Chip 1 is not yet implemented...')
    
    # In FITS, L[a:b] means L[a], L[a+1], ..., L[b], so total (b-a+1) elements.
    # Also the indexing starts from 1.
    # Finally, FITS uses XY order, while the np.ndarray is using row-column
    # order, i.e., YX order, as most programming languages do.
    # Thus, to port to Python, L[a:b] should become L[b:a-1].
    if xbin == 1:
        if chip == 2:
            print('x Binning 1: I have not checked it yet')
            overscan = [data[:, 520:536],   # in FITS, 521:536
                        data[:, 536:552],   # in FITS, 537:552
                        data[:, 1592:1608], # in FITS, 1593:1608
                        data[:, 1608:1624]] # in FITS, 1609:1624
#        else:
        
    elif xbin == 2:
        if chip == 2:
            overscan = [data[:, 260:276],   # in FITS, 261:276
                        data[:, 276:292],   # in FITS, 277:292
                        data[:, 812:828],   # in FITS, 813:828
                        data[:, 828:844]] # in FITS, 829:844
            # length in x-direction, including overscan region
            ch_xlen = [276, 276, 276, 276]
#        else:
        
    overscan_ch = []
    for ch in range(4):
        overscan_pattern = np.average(overscan[ch], axis=1) 
        # The original code ``ovsub.cl`` uses ``IMAGES.IMFILTER.BOXCAR`` task for
        # smoothing the overscan region. Same is implemented in astropy.convolution.
        # convolve with option ``boundary='extend'``.
        overscan_smoothed = convolve(overscan_pattern, 
                                     Box1DKernel(ybox), 
                                     boundary='extend')
        overscan_map = np.repeat(overscan_smoothed, ch_xlen[ch])
        overscan_map = overscan_map.reshape(ylen, ch_xlen[ch])
        overscan_ch.append(overscan_map)
    
    overscan_map = np.hstack(overscan_ch)
    overscan_subtracted = data - overscan_map
#    ccdout = CCDData(overscan_subtracted, header = hdr, unit='adu')
    
#    if outputdir != '':
#        ccdout.write(outputdir, overwrite=True)
    
    return overscan_subtracted
    
def initialize_2Dcombiner(table, colname_nx='NAXIS1', colname_ny='NAXIS2',
                          dtype=np.float32):
    empty_arr = np.empty((table[colname_ny][0],
                          table[colname_nx][0],
                          len(table)),
                         dtype=dtype)
    return empty_arr


def clip_median_combine(data, sigma=3, iters=5, min_value=0, axis=-1):
    ''' Median combine after sigma clipping and minimum value correction.
    '''
    data = np.atleast_1d(data)
    data[data < min_value] = min_value
    data_clipped = sigma_clip(data, sigma=sigma, iters=iters, axis=axis)
    data_combined = np.ma.median(data_clipped, axis=axis)
    
    return data_combined


def print_info(Nccd, min_value, sigma, iters):
    info_str = ('Median combine {:d} images: minimum pixel value set to {:.1f}, '
                + '{:.1f}-sigma {:d}-iteration')
    print('All files loaded.')
    print(info_str.format(Nccd, min_value, sigma, iters))    
    return

def check_exptime(table, colname_file, colname_nx, colname_ny, colname_exptime):
    exptimes = table[colname_exptime].data.data

    if len(np.unique(exptimes)) != 1:
        print('There are more than one exposure times:')
        print('\texptimes = ', end=' ')
        print(np.unique(exptimes), end=' ')
        print('seconds')
        table[colname_file, colname_nx, colname_ny, 
              colname_exptime].pprint(max_width=150)

    return exptimes

def make_master_bias(table, sigma=3, iters=5, min_value=0, colname_file='file',
                     colname_nx='NAXIS1', colname_ny='NAXIS2',
                     dtype=np.float32, output=''):
    ''' Make master bias from the given bias table.

    bias_table: astropy.table.Table
        This should contain the filename and naxis data.
    '''
    Nccd = len(table)
    bias_orig = initialize_2Dcombiner(table=table,
                                      colname_nx=colname_nx,
                                      colname_ny=colname_ny,
                                      dtype=dtype)

    for i in range(Nccd):
        bias_orig[:, :, i] = fits.getdata(table[colname_file][i])

    print_info(Nccd=Nccd, min_value=min_value, sigma=sigma, iters=iters)
    mbias = clip_median_combine(data=bias_orig, sigma=sigma, iters=iters,
                                min_value=min_value, axis=-1)
    mbias = mbias.astype(dtype)
    
    if output != '':
        print('Saving to {:s}'.format(output))
        hdu = fits.PrimaryHDU(data=mbias.data)
        hdu.writeto(output, overwrite=True)
            
    return mbias.data


def make_master_dark(table, mbias, sigma=3, iters=5, min_value=0, 
                     colname_file='file', colname_exptime='EXPTIME',
                     colname_nx='NAXIS1', colname_ny='NAXIS2', 
                     dtype=np.float32, output=''):
    '''Make dark frame from the given dark table
    '''
    
    if not isinstance(mbias, CCDData):
        mbias = CCDData(data=mbias, unit='adu')
    else:
        mbias = mbias.copy()
    
    Nccd = len(table)

    exptimes = check_exptime(table=table, colname_file=colname_file, 
                             colname_nx=colname_nx, colname_ny=colname_ny, 
                             colname_exptime=colname_exptime)

    dark_orig = initialize_2Dcombiner(table=table,
                                      colname_nx=colname_nx,
                                      colname_ny=colname_ny,
                                      dtype=dtype)

    for i in range(Nccd):
        dark_orig[:, :, i] = ( (fits.getdata(table[colname_file][i]) - mbias)
                              / exptimes[i])
    print_info(Nccd=Nccd, min_value=min_value, sigma=sigma, iters=iters)
    mdark = clip_median_combine(dark_orig, sigma=sigma, iters=iters, 
                                min_value=min_value, axis=-1)
    mdark = mdark.astype(dtype)
    
    if output != '':
        print('Saving to {:s}'.format(output))
        hdu = fits.PrimaryHDU(data=mdark.data)
        hdu.writeto(output, overwrite=True)

    return mdark.data


def make_master_flat(table, mbias, sigma=3, iters=5, min_value=0,
                     colname_file='file', colname_exptime='EXPTIME',
                     colname_nx='NAXIS1', colname_ny='NAXIS2',
                     dtype=np.float32, output=''):
    ''' Make master flat from the given flat table and master bias.
    
    Dark subtraction is not implemented yet.
    '''
    
    if not isinstance(mbias, CCDData):
        mbias = CCDData(data=mbias, unit='adu')
    else:
        mbias = mbias.copy()
    
    Nccd = len(table)

    exptimes = check_exptime(table=table, colname_file=colname_file, 
                             colname_nx=colname_nx, colname_ny=colname_ny, 
                             colname_exptime=colname_exptime)
    
    flat_orig = initialize_2Dcombiner(table=table,
                                      colname_nx=colname_nx,
                                      colname_ny=colname_ny,
                                      dtype=dtype)

    for i in range(Nccd):
        flat_orig[:, :, i] = ( fits.getdata(table[colname_file][i]) 
                              / exptimes[i])
    
    print_info(Nccd=Nccd, min_value=min_value, sigma=sigma, iters=iters)
    mflat = clip_median_combine(flat_orig, sigma=sigma, iters=iters, 
                                min_value=min_value, axis=-1)

    print('and bias subtraction ...')
    mflat -= mbias    
    mflat = mflat.astype(dtype)
    
    if output != '':
        print('Saving to {:s}'.format(output))
        hdu = fits.PrimaryHDU(data=mflat.data)
        hdu.writeto(output, overwrite=True)

    return mflat.data


def response_correct(data, normdata1d, dispaxis=0, output='', threshold=0.,
                     low_reject=3., high_reject=3.,
                     iters=3, function='legendre', order=3):
    ''' Response correction for a 2-D spectrum
    Parameters
    ----------
    data : numpy.ndarray
        The data to be corrected. Usually a (combined) flat field image.
    normdata1d: numpy.ndarray
        1-D numpy array which contains the suitable normalization image. 
    dispaxis : {0, 1}
        The dispersion axis. 0 and 1 mean column and line, respectively.
    threshold : float
        The final 2-D response map pixels smaller than this value will be 
        replaced by 1.0.
    
    Usage
    -----
    nsmooth = 7
    normdata1d = np.sum(mflat[700:900, :] , axis=0)
    normdata1d = convolve(normdata1d, Box1DKernel(nsmooth), boundary='extend')
    response = preproc.response_correct(data = mflat.data, 
                                        normdata1d=normdata1d, 
                                        dispaxis=1,
                                        order=10)
    
    '''
    
    nlambda = len(normdata1d)
    nrepeat = data.shape[dispaxis - 1]
    
    if data.shape[dispaxis] != nlambda:
        wstr = "data shape ({:d}, {:d}) with dispaxis {:d} \
        does not match with normdata1d ({:d})"
        wstr = wstr.format(data.shape[0], data.shape[1], 
                           dispaxis, normdata1d.shape[0])
        raise Warning(wstr)
    
    x = np.arange(0, nlambda)
    
    if function == 'legendre':
        fitted = legval(x, legfit(x, normdata1d, deg=order))
        # TODO: The iteration here should be the iteration over the
        # fitting, not the sigma clip itself.
        residual = normdata1d - fitted
        clip = sigma_clip(residual, iters=iters,
                          sigma_lower=low_reject, sigma_upper=high_reject)
    else:
        raise Warning("{:s} is not implemented yet".format(function))
        
    mask = clip.mask
    weight = (~mask).astype(float)  # masked pixel has weight = 0.
    coeff = legfit(x, normdata1d, deg=order, w=weight)
    
    if function == 'legendre':
        response = legval(x, coeff)
    
    response /= np.average(response)
    response[response < threshold] = 1.
    response_map = np.repeat(response, nrepeat)
    
    if dispaxis == 0:
        response_map = response_map.reshape(nlambda, nrepeat)
    elif dispaxis == 1:
        response_map = response_map.reshape(nrepeat, nlambda)
    
    response2d = data/response_map
    
    return response2d


def preproc(fnames, mbias, mflat, min_value=0, crrej=False):
    
    if not isinstance(mbias, CCDData):
        master_bias = CCDData(data=mbias, unit='adu')
    else:
        master_bias = mbias.copy()

    if not isinstance(mflat, CCDData):
        master_flat = CCDData(data=mflat, unit='adu')
    else:
        master_flat = mflat.copy()

    processed_ccds = []

    for fname in fnames:
        print('Preprocessing started for {:s}'.format(fname))
        obj_p = CCDData(fits.getdata(fname), 
                        meta=fits.getheader(fname),
                        unit='adu')
        gain = obj_p.header['gain']
        xbin = obj_p.header['bin-fct1']
        chip = obj_p.header['det-id']
        # TODO: change ccd.data to just ccd (after ccdproc ver>1.3 release)
        # TODO: gain value differs from ch to ch.
        
        if crrej:
            ronoise = 4.0
            import astroscrappy
            # TODO: implement spec crrej
            m, obj_p.data = astroscrappy.detect_cosmics(obj_p.data,
                                                        satlevel=np.inf,
                                                        sepmed=False,
                                                        cleantype='medmask',
                                                        fsmode='median',
                                                        gain=gain,
                                                        readnoise=ronoise)


#       TODO: Use ccdproc when ccdproc 1.3 is released
#        obj_p = subtract_bias(obj_p, master_bias)
#        obj_p = flat_correct(obj_p, master_flat, min_value=min_value)
#        obj_p = gain_correct(obj_p, gain=gain, gain_unit=u.electron/u.adu)
        
        obj_p = (obj_p.data - master_bias)
        obj_p = suboverscan(data=obj_p, xbin=xbin, chip=chip, ybox=5)
        obj_p = obj_p / master_flat * np.mean(master_flat)
        obj_p = obj_p * gain
        obj_p.astype(np.float32)
        processed_ccds.append(obj_p)

        print('\tDone')

    return processed_ccds