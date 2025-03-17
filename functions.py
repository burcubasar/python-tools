import numpy as np
from scipy.ndimage import zoom
from scipy.interpolate import interpn
from scipy.spatial.transform import Rotation



def make_grid(image_sz=(128,128), origin_offset=(0,0), homogeneous=True):
    ''' construct 2d or 3d grid '''
    if len(image_sz) == 2: 
        nrows, ncols = image_sz
        nslices = 1
        z_offset = 0
    elif len(image_sz) == 3:
        nrows, ncols, nslices = image_sz
    else:
        assert len(image_sz) in [2,3], 'Input must be 2D or 3D'

    x_offset, y_offset, z_offset = origin_offset
    # endpont=False ia ssame as np.linsapce(0,sz-1,sz) 
    grid = np.meshgrid(np.linspace(0, ncols, ncols, endpoint=False) + x_offset,
                             np.linspace(0, nrows, nrows, endpoint=False) + y_offset,
                             np.linspace(0, nslices, nslices, endpoint=False) + z_offset)
    grid = np.array(grid).reshape((3, -1))
    if homogeneous: grid = np.vstack((grid, np.ones((1, grid.shape[1]))))
    return grid



def rotate_volume(volume, rotation_matrix, method='linear'):
    ''' rotate 3d volume at its center '''
    nrows, ncols, nslices = volume.shape 
    vxi, vyi, vzi = np.linspace(-ncols/2, ncols/2, ncols), np.linspace(-nrows/2, nrows/2, nrows), np.linspace(-nslices/2, nslices/2, nslices) # x is the width, i.e. the number of columns
    xyz = np.array(np.meshgrid(vxi, vyi, vzi)).reshape(3,-1)
    xyz = np.vstack((xyz, np.ones((1, xyz.shape[1]))))
    rotated_xyz = rotation_matrix @ xyz
    rotated_volume = interpn((vxi, vyi, vzi), volume.transpose([1,0,2]), rotated_xyz[:3, :].T, bounds_error=False,
                                method=method).reshape(volume.shape)
    return rotated_volume



def standardize(data):
    data = data - np.nanmean(data) 
    data = data / np.nanstd(data)
    return data



def rescale(data, lower=0, upper=1):
    ''' rescale data to range (lower, upper). Defaults to (0,1) '''
    data = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))
    if lower != 0 and upper != 1:
        data = data * (upper - lower) + lower
    return data


def resize(data, sz):
    ''' resize 2d or 3d data '''
    if data.ndim == 2:
        x,y = data.shape
        data = zoom(data, (sz[0]/x, sz[1]/y), order=0, cval=np.nanmin(data))
    elif data.ndim == 3:
        x,y,z = data.shape
        data = zoom(data, (sz[0]/x, sz[1]/y, sz[2]/z ), order=0, cval=np.nanmin(data))
    data = np.array(data, dtype='float32')
    return data




def rotmat_from_eul(params, seq='xyz', degrees=True, rot_first=True):
    if rot_first:
        mat = np.eye(4)
        mat[:3,:3] = Rotation.from_euler(seq, [params[3:]], degrees=degrees).as_matrix()
        mat[:3, 3] = np.asarray([params[0], params[1], params[2]])
        # print(mat)
    else:
        mat, t = np.eye(4),np.eye(4)
        mat[:3,:3] = Rotation.from_euler(seq, [params[3:]], degrees=degrees).as_matrix()
        t[:3, 3] = np.asarray([params[0], params[1], params[2]])
        mat = mat @ t
    return mat



def get_normal(v1, v2, v3, normalize=True):
    v13, v23 = v1 - v3, v2 - v3
    normal_vector = np.cross(v13, v23) 
    if normalize: normal_vector /= np.linalg.norm(normal_vector)
    return normal_vector




def get_angle(u, v, degrees=True):
    angle = np.arccos(np.dot(u, v)/(np.linalg.norm(u)*np.linalg.norm(v)))
    if degrees: angle = np.rad2deg(angle)
    return angle

