import numpy as np
from scipy.signal import convolve2d 
import matplotlib.pyplot as plt
import os
import skimage.io as skio
import cv2
from skimage.transform import resize
from skimage.color import rgb2gray

os.makedirs("outputs", exist_ok=True)

def gaussian_blur(img, sigma):
    k_size = int(2 * np.ceil(3 * sigma) + 1)
    g_kernel = cv2.getGaussianKernel(k_size, sigma)
    gaussian = g_kernel @ g_kernel.T

    channels = []
    for c in range(3):
        channels.append(convolve2d(img[:,:,c], gaussian, mode = "same", boundary= "symm"))
    blurred = np.stack(channels, axis = 2)
    return blurred


def gaussian_stack(img, levels, sigma):
    save = [img]
    for i in range(1, levels):
        new = gaussian_blur(img, sigma * 2**(i-1))
        save.append(new)
    return save


def laplacian_stack(img, levels, sigma):
    g = gaussian_stack(img, levels, sigma)
    lap = []
    for i in range(levels - 1):
        lap.append(g[i] - g[i + 1])     
    lap.append(g[-1])             
    return lap

# apple  = skio.imread("apple.jpeg")[:, :, :3] / 255
# orange = skio.imread("orange.jpeg")[:, :, :3] / 255

# levels, sigma = 5, 2  

# g_apple  = gaussian_stack(apple,  levels, sigma)
# l_apple  = laplacian_stack(apple,  levels, sigma)
# g_orange = gaussian_stack(orange, levels, sigma)
# l_orange = laplacian_stack(orange, levels, sigma)

def show_lap(L, is_last):
    # signed band-pass levels: shift so zero is mid-gray; the last level is a normal image
    return np.clip(L if is_last else L + 0.5, 0, 1)

def plot_stacks(gstack, lstack, name):
    fig, axes = plt.subplots(2, levels, figsize=(3 * levels, 6))
    for i in range(levels):
        axes[0, i].imshow(np.clip(gstack[i], 0, 1))
        axes[0, i].set_title(f"{name} Gaussian L{i}")
        axes[1, i].imshow(show_lap(lstack[i], i == levels - 1))
        axes[1, i].set_title(f"{name} Laplacian L{i}")
    for ax in axes.ravel():
        ax.axis("off")
    plt.tight_layout()
    return fig

# # os.makedirs("outputs", exist_ok=True)
# fig = plot_stacks(g_apple, l_apple, "apple")
# fig.savefig("outputs/3_stacks_apple.png", dpi=100, bbox_inches="tight")
# fig = plot_stacks(g_orange, l_orange, "orange")
# fig.savefig("outputs/3_stacks_orange.png", dpi=100, bbox_inches="tight")
# plt.show()


# --- part 2.4 ---
def blend(imA, imB, mask, levels=5, sigma=2):
    lapA = laplacian_stack(imA, levels, sigma)
    lapB = laplacian_stack(imB, levels, sigma)
    maskS = gaussian_stack(mask, levels, sigma)   

    parts_A, parts_B = [], []
    for i in range(levels):
        parts_A.append(maskS[i] * lapA[i])
        parts_B.append((1 - maskS[i]) * lapB[i])

    blended = sum(parts_A) + sum(parts_B)
    return np.clip(blended, 0, 1), parts_A, parts_B, maskS


# mask = np.zeros_like(apple)
# mask[:, :apple.shape[1] // 2] = 1
# oraple, pA, pB, maskS = blend(apple, orange, mask, levels, sigma)

# def show_signed(x, is_last):
#     # band-pass levels are signed, so shift so zero is mid-gray; last level is a normal image
#     return np.clip(x if is_last else x + 0.5, 0, 1)

# col_titles = ["apple (weighted)", "orange (weighted)", "mask", "result (sum)"]

# fig, axes = plt.subplots(levels, 4, figsize=(8, 10), constrained_layout=True)
# for i in range(levels):
#     last = (i == levels - 1)
#     cells = [show_signed(pA[i], last),
#              show_signed(pB[i], last),
#              np.clip(maskS[i], 0, 1),
#              show_signed(pA[i] + pB[i], last)]
#     for j, im in enumerate(cells):
#         ax = axes[i, j]
#         ax.imshow(im)
#         ax.set_xticks([]); ax.set_yticks([])
#         if i == 0:
#             ax.set_title(col_titles[j], fontsize=9)
#         if j == 0:
#             ax.set_ylabel(f"level {i}", fontsize=9)

# fig.savefig("outputs/2_4_levels_grid.png", dpi=100)
# plt.show()

madrid = skio.imread("3/proj2/madrid.jpg")
tahoe = skio.imread("3/proj2/tahoe.jpg")

froakie = skio.imread("3/proj2/froakie.jpg")
greninja = skio.imread("3/proj2/gren.jpeg")

from skimage.color import rgb2hsv
from scipy.ndimage import label, binary_fill_holes, binary_opening, binary_erosion

levels, sigma = 5, 2

os.makedirs("outputs", exist_ok=True)
levels, sigma = 5, 2

def circle_mask(H, W, r_frac=0.35, cy_frac=0.5, cx_frac=0.5):
    yy, xx = np.ogrid[:H, :W]
    r = r_frac * min(H, W)
    m = (yy - cy_frac * H) ** 2 + (xx - cx_frac * W) ** 2 <= r ** 2
    return np.stack([m.astype(float)] * 3, axis=2)


def flower_mask(H, W, r_frac=0.4, petals=6, depth=0.35):
    yy, xx = np.mgrid[:H, :W]
    dy, dx = yy - H / 2, xx - W / 2
    rho, theta = np.hypot(dx, dy), np.arctan2(dy, dx)
    R = r_frac * min(H, W) * (1 + depth * np.cos(petals * theta))   # radius varies with angle
    m = rho <= R
    return np.stack([m.astype(float)] * 3, axis=2)

def load_pair(inner_path, outer_path, long_side=500):
    outer = skio.imread(outer_path)[:, :, :3] / 255.
    s = long_side / max(outer.shape[:2])
    outer = resize(outer, (int(outer.shape[0] * s), int(outer.shape[1] * s)), anti_aliasing=True)
    inner = skio.imread(inner_path)[:, :, :3] / 255.
    inner = resize(inner, outer.shape[:2], anti_aliasing=True)
    return inner, outer
def plot_levels(pA, pB, maskS, result, name):
    titles = ["inside mask (weighted)", "outside mask (weighted)", "mask", "sum of level"]
    fig, axes = plt.subplots(levels, 4, figsize=(8, 10), constrained_layout=True)
    for i in range(levels):
        last = (i == levels - 1)
        sh = (lambda x: np.clip(x, 0, 1)) if last else (lambda x: np.clip(x + 0.5, 0, 1))
        cells = [sh(pA[i]), sh(pB[i]), np.clip(maskS[i], 0, 1), sh(pA[i] + pB[i])]
        for j, im in enumerate(cells):
            ax = axes[i, j]
            ax.imshow(im)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0: ax.set_title(titles[j], fontsize=9)
            if j == 0: ax.set_ylabel(f"level {i}", fontsize=9)
    fig.savefig(f"outputs/2_4_{name}_levels.png", dpi=100)
    plt.close(fig)

def run_blend(inner_path, outer_path, mask_fn, name):
    inner, outer = load_pair(inner_path, outer_path)
    mask = mask_fn(*inner.shape[:2])
    result, pA, pB, maskS = blend(inner, outer, mask, levels, sigma)   # A = inner (where mask = 1)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (t, im) in zip(axes, [("inside mask", inner), ("outside mask", outer), ("mask", mask), ("blend", result)]):
        ax.imshow(np.clip(im, 0, 1)); ax.set_title(t); ax.axis("off")
    plt.tight_layout()
    fig.savefig(f"outputs/2_4_{name}_overview.png", dpi=100, bbox_inches="tight")

    plot_levels(pA, pB, maskS, result, name)
    skio.imsave(f"outputs/2_4_{name}_result.jpg", (result * 255).astype(np.uint8))
    return result

run_blend("3/proj2/madrid.jpg", "3/proj2/tahoe.jpg", flower_mask, "madrid_in_tahoe_flower")
# run_blend("froakie.jpg", "gren.jpeg",lambda H, W: circle_mask(H, W, r_frac=0.35, cy_frac=0.32), "froakie_in_greninja_circle")

plt.show()