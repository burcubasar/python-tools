import os, glob, errno
import nibabel as nib
import numpy as np
import imageio



def load_nifti(fname, scale=True):
    nib_data = nib.load(fname)
    data = nib_data.get_fdata()
    if scale: data = (data - np.min(data)) / (np.max(data)-np.min(data))
    header = nib_data.header
    affine = nib_data.affine
    return data, header, affine



def load_image(filename):
    image = np.array(imageio.imread(filename))
    return image



def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise



