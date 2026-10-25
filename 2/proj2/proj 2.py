import numpy as np
from scipy.signal import convolve2d 
import matplotlib.pyplot as plt
import os
import skimage.io as skio
import cv2
from skimage.transform import rescale


os.makedirs("outputs", exist_ok=True)

# --- part 1.1 ---

def conv2d_4loops(img, kernel):
    kh, kw = kernel.shape
    H, W = img.shape
    flipped_ker = np.flip(kernel)

    pad_top    = kh // 2
    pad_bottom = kh - 1 - pad_top
    pad_left   = kw // 2
    pad_right  = kw - 1 - pad_left
    padded_img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)))
    out = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            for u in range(kh):
                for v in range(kw):
                    out[i][j] += padded_img[i + u][j + v] * flipped_ker[u][v]
    return out


def conv2d_2loops(img, kernel):
    kh, kw = kernel.shape
    H, W = img.shape
    flipped_ker = np.flip(kernel)
    
        
    pad_top    = kh // 2
    pad_bottom = kh - 1 - pad_top
    pad_left   = kw // 2
    pad_right  = kw - 1 - pad_left
    padded_img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)))
    out = np.zeros((H, W))

    for i in range(H):
        for j in range(W):
            window = padded_img[i : i + kh, j : j + kw]
            out[i, j] = np.sum(window * flipped_ker)
    return out


# --- filters ---
box = np.ones((9, 9)) / 81
Dx = np.array([[1, 0, -1]])
Dy = np.array([[1], [0], [-1]])


me = skio.imread("me.JPG", True)
me = rescale(me, 0.1, anti_aliasing=True)

box_out = conv2d_2loops(me, box)
dx_out = conv2d_2loops(me, Dx)
dy_out = conv2d_2loops(me, Dy)


results1_1 = {
    "original": me,
    "box 9x9": box_out,
    "Dx": dx_out,
    "Dy": dy_out,
}


# --- part 1.2 ---
cameraman = skio.imread("cameraman.png", True)
cameraman_dxout = convolve2d(cameraman, Dx, mode = "same")
cameraman_dyout = convolve2d(cameraman, Dy, mode = "same")
grad_mag = np.sqrt((cameraman_dxout) ** 2 + (cameraman_dyout) ** 2)
t = 0.24
binarized = grad_mag > 0.24

results1_2 = {
    "cameraman": cameraman,
    "Dx": cameraman_dxout,
    "Dy": cameraman_dyout,
    "gradient_mag": grad_mag,
    "threshold" : binarized
}


# fig, axes = plt.subplots(1, 5, figsize=(20, 4))
# for ax, (name, im) in zip(axes, results1_2.items()):
#     ax.imshow(im, cmap="gray")
#     ax.set_title(name)
#     ax.axis("off")
# plt.tight_layout()
# fig.savefig("outputs/1_2_figure.png", dpi=100, bbox_inches="tight")   # optional; make sure "outputs" exists
# plt.show()

# thresholds = [0.2, 0.23, 0.24, 0.25, 0.26]
# fig, axes = plt.subplots(1, 5, figsize=(16, 4))
# for ax, t in zip(axes, thresholds):
#     ax.imshow(grad_mag > t, cmap="gray")
#     ax.set_title(f"t = {t}")
#     ax.axis("off")
# plt.show()


# --- part 1.3 ---
sigma = 2
k_size = int(2 * np.ceil(3 * sigma) + 1)
gaussian = cv2.getGaussianKernel(k_size, sigma) 
g = gaussian @ gaussian.T

blur = convolve2d(cameraman, g, mode = "same")
blur_dxout = convolve2d(blur, Dx, mode = "same")
blur_dyout = convolve2d(blur, Dy, mode = "same")
blur_grad_mag = np.sqrt((blur_dxout) ** 2 + (blur_dyout) ** 2)

blur_edges = blur_grad_mag > 0.08


# fig, axes = plt.subplots(1, 6, figsize=(12, 4))
# axes[0].imshow(g, cmap="gray");           axes[0].set_title("gaussian")

# axes[1].imshow(blur_dxout, cmap="gray");  axes[1].set_title("blur dx ")
# axes[2].imshow(blur_dyout, cmap="gray");  axes[2].set_title("blur dy ")
# axes[3].imshow(blur_grad_mag, cmap="gray");  axes[3].set_title("blur grad mag")
# axes[4].imshow(blur_edges, cmap="gray");     axes[4].set_title("blur edges")

# for ax in axes:
#     ax.axis("off")
# plt.tight_layout()
# plt.show()


# --- derivative of gaussian ---
dog_x =convolve2d(g, Dx, mode = "same")
dog_y = convolve2d(g, Dy, mode = "same")

# fig, axes = plt.subplots(1, 3, figsize=(12, 4))
# axes[0].imshow(dog_x, cmap="gray");           axes[0].set_title("DoG dx")
# axes[1].imshow(dog_y, cmap="gray");  axes[1].set_title("DoG dy ")

# for ax in axes:
#     ax.axis("off")
# plt.tight_layout()
# plt.show()

dog_dxout = convolve2d(cameraman, dog_x, mode = "same")
dog_dyout = convolve2d(cameraman, dog_y, mode = "same")
dog_grad_mag = np.sqrt((dog_dxout) ** 2 + (dog_dyout) ** 2)

dog_edges = dog_grad_mag > 0.08

results1_3 = {
    "cameraman": cameraman,
    "blurred": blur,
    "blurred_dx": blur_dxout,
    "blurred_dy": blur_dyout,
    "blurred gradient magnitude": blur_grad_mag,
    "blurred edges" : blur_edges,
    "DoG_x" : dog_x,
    "DoG_y" : dog_y,
    "DoG_dx": dog_dxout,
    "DoG_dy": dog_dyout,
    "DoG gradient magnitude": dog_grad_mag,
    "DoG edges" : dog_edges,
}

# fig, axes = plt.subplots(1, 6, figsize=(12, 4))
# axes[0].imshow(cameraman, cmap="gray");  axes[0].set_title("original")
# axes[1].imshow(dog_dxout, cmap="gray");  axes[1].set_title("DoG dx ")
# axes[2].imshow(dog_dyout, cmap="gray");  axes[2].set_title("DoG dy ")
# axes[3].imshow(dog_grad_mag, cmap="gray");  axes[3].set_title("DoG grad mag")
# axes[4].imshow(dog_edges, cmap="gray");     axes[4].set_title("DoG edges")

# for ax in axes:
#     ax.axis("off")
# plt.tight_layout()
# plt.show()

# --- part 2 ---
# --- 2.1 ---

taj = skio.imread("taj.jpg")

sigma = 2
k_size = int(2 * np.ceil(3 * sigma) + 1)
gaussian = cv2.getGaussianKernel(k_size, sigma) 
g = gaussian @ gaussian.T

taj_r_blurred = convolve2d(taj[:, :, 0], g, mode="same", boundary="symm")
taj_g_blurred = convolve2d(taj[:, :, 1], g, mode="same", boundary="symm")
taj_b_blurred = convolve2d(taj[:, :, 2], g, mode="same", boundary="symm")
taj_blurred = np.stack((taj_r_blurred, taj_g_blurred, taj_b_blurred), axis=2)


taj_high_freq = taj - taj_blurred
def taj_sharper(alpha):
    return np.clip(taj + taj_high_freq * alpha, 0, 255).astype(np.uint8)
taj_sharp_half = taj_sharper(0.5)
taj_sharp_1 = taj_sharper(1)
taj_sharp_2 = taj_sharper(2)
taj_sharp_5 = taj_sharper(5)
hf_vis = np.clip(taj_high_freq + 128, 0, 255).astype(np.uint8) 


# --- show org, blur, sharp, and high freq ---
# fig, axes = plt.subplots(1, 4, figsize=(20, 5))
# for ax, (name, im) in zip(axes, [("original", taj),
#                                  ("blurred", np.clip(taj_blurred, 0, 255).astype(np.uint8)),
#                                  ("high freq", hf_vis),
#                                  ("sharpened", sharp_u8)]):
#     ax.imshow(im)          # no cmap: these are color images
#     ax.set_title(name)
#     ax.axis("off")
# plt.tight_layout()
# fig.savefig("outputs/2_1_taj_figure.png", dpi=100, bbox_inches="tight")
# plt.show()

# --- sharpen alpha value changes ---
# fig, axes = plt.subplots(1, 4, figsize=(20, 5))
# for ax, (name, im) in zip(axes, [("sharpened, a = 0.5", taj_sharp_half),
#                                  ("a = 1", taj_sharp_1),
#                                  ("a = 2", taj_sharp_2),
#                                  ("a = 5", taj_sharp_5)]):
#     ax.imshow(im)          # no cmap: these are color images
#     ax.set_title(name)
#     ax.axis("off")
# plt.tight_layout()
# fig.savefig("outputs/2_1_taj_figure.png", dpi=100, bbox_inches="tight")
# plt.show()



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

b = np.clip(unsharp_mask_filter(taj, 0.5, 2), 0, 255).astype(np.uint8) 

# blur then sharpen
orig = skio.imread("Casa Batllo.JPG").astype(float) 
blurred = gaussian_blur(orig, sigma=2)
resharp = unsharp_mask_filter(blurred, alpha=1.0, sigma=2)

# show = lambda x: np.clip(x, 0, 255).astype(np.uint8)
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# for ax, (name, im) in zip(axes, [("original", orig), ("blurred", blurred), ("resharpened", resharp)]):
#     ax.imshow(show(im)); ax.set_title(name); ax.axis("off")
# plt.tight_layout()
# fig.savefig("outputs/2_1_eval.png", dpi=100, bbox_inches="tight")
# plt.show()

for a in [0.5, 1, 1.5, 2, 3, 4]:
    r = np.clip(unsharp_mask_filter(blurred, a, 2), 0, 255)
    print(a, np.abs(orig - r).mean())