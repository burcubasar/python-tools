from functions import *
import numpy as np
from PIL import Image

def write_gif_from_array(buffer,img,fname, flag=False):
    img = rescale(img,0,1)
    new_frame = Image.fromarray((np.uint8(np.squeeze(img)*255.)))
    buffer.append(new_frame)
    if flag:
        buffer[0].save(fname + '.gif', format='GIF',
                    append_images=buffer[1:],
                    save_all=True,
                    duration=200, loop=0)
        

