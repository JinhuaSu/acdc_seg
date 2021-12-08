#%%
import numpy
import numpy as np
import math
from PIL import Image, ImageDraw

# from acdc_seg.model import get_circle_mask
def get_raw_radia_mask(w,h,circle_n):
    img = Image.new('L', (w, h), 0)
    cx = w // 2
    cy = h // 2
    per_range = 2* 3.14 / circle_n
    R = max(w,h)
    for i in range(1,circle_n):
        theta_start = i * per_range
        theta_end = (i+1) * per_range
        x_start, y_start =cx+ R * math.cos(theta_start), cy+R *math.sin(theta_start) 
        x_end, y_end =cx+ R * math.cos(theta_end), cy+R *math.sin(theta_end) 
        polygon = [(cx,cy),(x_start,y_start),(x_end,y_end)] 
        ImageDraw.Draw(img).polygon(polygon, outline=i, fill=i)
    mask = numpy.array(img)
    return mask
    # morograph or poligon to mask
polygon = [(3,3),(3,71),(7,3)] 
width = 20
height = 20

img = Image.new('L', (width, height), 0)
ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
mask = numpy.array(img)
# print(mask)
print(get_raw_radia_mask(20,20,6))
# %%
