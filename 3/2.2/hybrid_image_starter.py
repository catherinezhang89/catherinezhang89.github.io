import matplotlib.pyplot as plt
from align_image_code import align_images
import numpy as np
from scipy.signal import convolve2d 
import matplotlib.pyplot as plt
import os
import skimage.io as skio
import cv2
from skimage.transform import rescale
from skimage.color import rgb2gray

# First load images

# high sf
im1 = plt.imread('./nutmeg.jpg') / 255.

# low sf
im2 = plt.imread('./DerekPicture.jpg') / 255.

# Next align images (this code is provided, but may be improved)
im1_aligned, im2_aligned = align_images(im1, im2)

y0 = 360  # cut off black top
im1_aligned = im1_aligned[y0:]
im2_aligned = im2_aligned[y0:]

## You will provide the code below. Sigma1 and sigma2 are arbitrary 
## cutoff values for the high and low frequencies

def gaussian_blur(img, sigma):
    k_size = int(2 * np.ceil(3 * sigma) + 1)
    g_kernel = cv2.getGaussianKernel(k_size, sigma)
    gaussian = g_kernel @ g_kernel.T

    channels = []
    for c in range(3):
        channels.append(convolve2d(img[:,:,c], gaussian, mode = "same", boundary= "symm"))
    blurred = np.stack(channels, axis = 2)
    return blurred

def unsharp_mask_filter(img, alpha, sigma):
    k_size = int(2 * np.ceil(3 * sigma) + 1)
    g_kernel = cv2.getGaussianKernel(k_size, sigma)
    gaussian = g_kernel @ g_kernel.T
    
    identity = np.zeros((k_size,k_size))
    identity[k_size // 2,k_size // 2] = 1
    unsharp_filter = identity + alpha * identity - alpha * gaussian

    channels = []
    for c in range(3):
        channels.append(convolve2d(img[:,:,c], unsharp_filter, mode = "same", boundary= "symm"))
    sharpened = np.stack(channels, axis = 2)
    
    return sharpened

def hybrid_image(im1, im2, s1, s2):
    high_freq = im1 - gaussian_blur(im1, s1)
    low_freq = gaussian_blur(im2, s2)
    return high_freq, low_freq, high_freq + low_freq
sigma1 = 4
sigma2 = 8
high_freq, low_freq, hybrid = hybrid_image(im1_aligned, im2_aligned, sigma1, sigma2)


hybrid_clipped = np.clip(hybrid, 0, 1)

fig, axes = plt.subplots(1, 3, figsize=(15, 6))
panels = [("nutmeg (high freq source)", im1_aligned),
          ("derek (low freq source)", im2_aligned),
          ("dermeg", hybrid_clipped)]
for ax, (name, im) in zip(axes, panels):
    ax.imshow(np.clip(im, 0, 1))
    ax.set_title(name)
    ax.axis("off")
plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
fig.savefig("outputs/3_hybrid_pair1.png", dpi=100, bbox_inches="tight")
plt.show()

# skio.imsave("outputs/3_hybrid_pair1.jpg", (hybrid_clipped * 255).astype(np.uint8))


# for s1, s2 in [(4, 8), (4, 10), (4, 15)]:
#     h = np.clip(hybrid_image(im1_aligned, im2_aligned, s1, s2), 0, 1)
#     plt.figure(); plt.imshow(h); plt.title(f"s1={s1}, s2={s2}")
# plt.show()


# # --- log magnitude of the Fourier transform ---
# pictures = [im1_aligned, im2_aligned, high_freq, low_freq, hybrid]
# names = ["oski (input)", "dog (input)", "high-passed oski", "low-passed dog", "hybrid"]
# fig, axes = plt.subplots(1, 5, figsize=(20, 4))
# for ax, img, name in zip(axes, pictures, names):
#     gray = rgb2gray(img)
#     spectrum = np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray))) + 1e-8)
#     ax.imshow(spectrum, cmap="gray")
#     ax.set_title(name)
#     ax.axis("off")
# plt.tight_layout()
# plt.show()