from PIL import Image
import numpy as np
import cv2
import helper_function as hf
import matplotlib.colors as mcolors


image = Image.open("shader\input images\input0.jpg")


colour_image = np.array(image)
Image.fromarray(cv2.convertScaleAbs(colour_image)).save("shader\\output images\\output5.png")


print("greying...")
gray_image = np.array(image.convert("L"))
print("blurring...")


nimage = cv2.GaussianBlur(colour_image, (9,9), 3)
savee = nimage.astype(np.uint8)
Image.fromarray(savee).save("shader\\output images\\blurred.png")

print("sobel edging...")

crazy_sobel,direction_array = hf.image_to_good_soble_per_colour_with_direction(nimage)
normalized_sobel = cv2.normalize(crazy_sobel, None, 0, 255, cv2.NORM_MINMAX)
color_like2 = cv2.cvtColor(normalized_sobel.astype(np.uint8), cv2.COLOR_GRAY2RGB)
Image.fromarray(color_like2).save("shader\\output images\\soble.png")


print("double thresholding...")
# dbt = hf.double_thresholding(crazy_sobel,14,40) 
dbt =  hf.adaptive_double_threshold(crazy_sobel,301,4,8,10,100)


savee = cv2.cvtColor(dbt.astype(np.uint8), cv2.COLOR_GRAY2RGB)
Image.fromarray(savee).save("shader\\output images\\dbt.png")

print("filling (extending lines)....")

filled,proxy_direction = hf.extend_lines(dbt,direction_array,15,5)


savee = cv2.cvtColor(filled.astype(np.uint8), cv2.COLOR_GRAY2RGB)
Image.fromarray(savee).save("shader\\output images\\filled.png")

print("filling more....")

full = hf.fill_in_the_blank(filled)

savee = cv2.cvtColor(full.astype(np.uint8), cv2.COLOR_GRAY2RGB)
Image.fromarray(savee).save("shader\\output images\\fuller.png")

print("finding splotches...")

splotches =  hf.splotch_map(full)

h = (splotches * 0.001) % 1
s = np.ones_like(h)*0.5
v = np.ones_like(h)*0.5
v[splotches == 0] = 0
hsv_img = np.stack([h,s,v], axis=-1)

rgb_img = mcolors.hsv_to_rgb(hsv_img)
output = (rgb_img * 255).astype(np.uint8)
clipped_splotches = np.uint8(np.clip(splotches, 0, 255)) 
Image.fromarray(clipped_splotches).save("shader\\output images\\splotches.png")
Image.fromarray(output).save("shader\\output images\\splotchified.png")


print("mixing splotches...")

H,W = splotches.shape
tally2 = {}

for i in range(H):
    for j in range(W):
        key = splotches[i,j]
        if (key> 0):
            if key in tally2:
                t1 = tally2[key][0] + 1
                t2 = tally2[key][1] + colour_image[i,j]
                tally2[key] = (t1,t2)
            else:
                tally2[key] = (1,np.zeros(3) + colour_image[i,j] )


averaged =  np.zeros_like(colour_image)
for i in range(H):
    for j in range(W):
        if (splotches[i,j] > 0):
            info = tally2[splotches[i,j]]
            if (info[0] > 5):
                averaged[i,j] = info[1] / info[0]

Image.fromarray(averaged).save("shader\\output images\\uniform.png")

print("done")

image.close()
