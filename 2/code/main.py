# CS194-26 (CS294-26): Project 1 starter Python code

# these are just some suggested libraries
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from skimage.transform import rescale

# name of the input file
imname = './italy.jpg'

# read in the image as grayscale (the glass plate scan is stacked grayscale)
im = cv.imread(imname, cv.IMREAD_GRAYSCALE)

# convert to float in [0,1] (might want to do this later on to save memory)
im = im.astype(np.float32) / 255.0
    
# compute the height of each part (just 1/3 of total)
height = int(np.floor(im.shape[0] / 3.0))

# separate color channels
b = im[:height]
g = im[height: 2*height]
r = im[2*height: 3*height]

# align the images
# functions that might be useful for aligning the images include:
# np.roll, np.sum, sk.transform.rescale (for multiscale)
def ncc(a, b):
    a_sub = a - np.mean(a)
    b_sub = b - np.mean(b)
    a_norm = a_sub / np.linalg.norm(a_sub)
    b_norm = b_sub / np.linalg.norm(b_sub)
    return np.sum(a_norm * b_norm)

def align(x,y, center = (0,0), window = 15):
    print(x.shape)
    is_base_case = max(x.shape) < 400
    if is_base_case:
        cx, cy = center
    else:
        x_down = rescale(x, 0.5)
        y_down = rescale(y, 0.5)
        new_center = align(x_down, y_down)
        cy, cx = new_center[1][0] * 2, new_center[1][1] * 2
        window = 10

    distance = 0
    best_i, best_j = cy, cx
    for i in np.arange(cy - window, cy + window):
        for j in np.arange(cx - window, cx + window):
            x_new = np.roll(x, i, axis = 0)
            x_new = np.roll(x_new, j, axis = 1)
            h_x,w_x = x.shape
            h_y, w_y = y.shape
            x_center = x_new[int(h_x * 0.1) : int(h_x * 0.9), int(w_x * 0.1): int(w_x * 0.9)]
            b_center = y[int(h_y * 0.1) : int(h_y * 0.9), int(w_y * 0.1) :int(w_y* 0.9)]
            
            score = ncc(x_center, b_center)
            better = score > distance
            if better:
                distance = score
                result = x_new
                best_i, best_j = i, j
    print(best_i,best_j)
    return result, (best_i, best_j)


# print(x.shape)
# print(x_new.shape)
            
# print(b_center.shape)
# print(x_center.shape)
ag, ag_offset = align(g, b)
ar, ar_offset = align(r, b)


# create a color image
im_out = np.dstack([ar, ag, b])

# display the image using matplotlib (expects RGB)
plt.figure(figsize=(8, 8))
plt.imshow(im_out)
plt.title('Colorized')
plt.axis('off')
plt.show() 


# prepare for OpenCV saving/display (expects BGR uint8)
out_uint8 = np.clip(im_out * 255.0, 0, 255).astype(np.uint8)
out_bgr = cv.cvtColor(out_uint8, cv.COLOR_RGB2BGR)

# save the image
fname = './out_fname.jpg'
cv.imwrite(fname, out_bgr)