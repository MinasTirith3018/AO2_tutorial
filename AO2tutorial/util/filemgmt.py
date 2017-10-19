import os
import numpy as np

__all__ = ['separate_chips', 
           'check_chip1', 'group_summary']


def separate_chips(directory_prefix='', fname_prefix='FCSA', verbose=True):
    ''' Separates files into chips 1 and 2.
    
    FOCAS has convention that the filename ends with odd number is from
    chip 1 and the even number from chip 2. Based on this convention, this
    function automatically separates images into two chips (make 
    subdirectories).
    
    Parameters
    ----------
    directory_prefix: str, optional
        The directory where the images are stored. Either absolute or relative.
        Set to default ('') if you are currently at the directory where the
        images are stored.
        
    fname_prefix: str, optional
        Only the files that start with this string will be affected.
    
    verbose: bool, optional
        Whether to print out the name of the files. This is not necessarily
        in alphabetical order, since the file movement can be done by
        multithreading.
   
    '''
    if directory_prefix == '':
        directory_prefix = os.getcwd()
        
    if not os.path.isabs(directory_prefix): # if relative directory,
        directory_prefix = os.path.join(os.getcwd(), directory_prefix)
    
    if not os.path.isdir(directory_prefix): # if data directory does not exist,
        raise NameError('The directory {:s} does not exist'.format(directory_prefix))

    os.makedirs(os.path.join(directory_prefix, 'chip1'), exist_ok=True)
    os.makedirs(os.path.join(directory_prefix, 'chip2'), exist_ok=True)
    
    for file in os.listdir(directory_prefix):
        oldpath = os.path.join(directory_prefix, file)
        
        if file.startswith(fname_prefix):
            print(oldpath)
            if file.endswith(("1.fits", "3.fits", "5.fits", "7.fits", "9.fits")):
                    newpath = os.path.join(directory_prefix, 'chip1', file)
                    os.rename(oldpath, newpath)
        
            elif file.endswith(("0.fits", "2.fits", "4.fits", "6.fits", "8.fits")):
                newpath = os.path.join(directory_prefix, 'chip2', file)
                os.rename(oldpath, newpath)

def group_summary(table, save=False, output='summary.csv'):
    ''' Make a summary csv file.
    
    Parameters
    ----------
    table: atropy.table.Table
    '''
    grouped = table.group_by(['det-id', 'obs-mod', 'data-typ',
                              'bin-fct1', 'bin-fct2',
                              'exptime'])
    if save:
        grouped.write(output, format='ascii.csv', overwrite=True)

    return grouped


def check_chip1(fits_tab):
    ''' Check whether there is chip1 data in chip2 directory.
    
    This is made because this tutorial uses chip 2 data only.
    
    Parameters
    ----------
    fits_tab: astropy.table.Table
    '''
    
    chip1 = fits_tab['det-id'] == 1
    if not np.count_nonzero(chip1) == 0:
        raise ValueError('There exists some file(s) from chip 1!!\n')