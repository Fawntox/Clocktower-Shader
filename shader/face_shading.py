import numpy as np
import matplotlib.colors as mcolors
from PIL import Image




splotches = np.load("shader\\output images\splotches.npy")
print(splotches.min(), splotches.max())

h = (splotches * 0.001) % 1
s = np.ones_like(h)*0.5
v = np.ones_like(h)*0.5
v[splotches == 0] = 0
hsv_img = np.stack([h,s,v], axis=-1)

rgb_img = mcolors.hsv_to_rgb(hsv_img)
output = (rgb_img * 255).astype(np.uint8)
Image.fromarray(output).save("shader\\output images\\splotchified.png")


