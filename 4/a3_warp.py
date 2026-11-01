"""
CS180 Project 4A - A.3: inverse warping (nearest neighbor + bilinear) and rectification.

Usage (rectify one rectangular object in a photo):
    python a3_warp.py path/to/IMG_0875.jpg --aspect 1.33 --out out/rect_painting
        -> window opens, click 4 corners of the object in order TL, TR, BR, BL
    --aspect = real width / height of the object (1.0 for a square tile, 0.773 for letter paper upright)
    --pts x1,y1,x2,y2,x3,y3,x4,y4   skip clicking and use these corners instead
    --max_dim 1000                  downscale photo so longest side <= this (speed)

Outputs (in --out folder): comparison.png, crop_compare.png, nn.png, bilinear.png, corners.png
Conventions: points are (x, y) = (col, row); pixel centers sit at integer coords;
im[row, col] = im[y, x].
"""
import argparse
import os
import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass


# ---------------------------------------------------------------- I/O
def load(path, max_dim=1000):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    s = max_dim / max(im.size) if max_dim else 1
    if s < 1:
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    return np.asarray(im, dtype=np.float64) / 255.0


# ---------------------------------------------------------------- A.2
def build_system(im1_pts, im2_pts):
    rows, rhs = [], []
    for (x, y), (xp, yp) in zip(im1_pts, im2_pts):
        rows.append([x, y, 1, 0, 0, 0, -x * xp, -y * xp]); rhs.append(xp)
        rows.append([0, 0, 0, x, y, 1, -x * yp, -y * yp]); rhs.append(yp)
    return np.array(rows, float), np.array(rhs, float)


def computeH(im1_pts, im2_pts):
    """H such that im2_pt ~ H @ [im1_pt, 1]."""
    im1_pts, im2_pts = np.asarray(im1_pts, float), np.asarray(im2_pts, float)
    assert len(im1_pts) == len(im2_pts) >= 4, f"{len(im1_pts)} vs {len(im2_pts)} points"
    A, b = build_system(im1_pts, im2_pts)
    h, *_ = np.linalg.lstsq(A, b, rcond=None)
    return np.append(h, 1).reshape(3, 3)


def apply_H(H, pts):
    pts = np.asarray(pts, float)
    Q = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    return Q[:, :2] / Q[:, 2:3]


# ---------------------------------------------------------------- A.3
def output_bbox(im, H):
    """Bounding box (xmin, ymin, xmax, ymax) of the image's 4 corners pushed through H."""
    h, w = im.shape[:2]
    corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], float)
    wc = apply_H(H, corners)
    xmin, ymin = np.floor(wc.min(0)).astype(int)
    xmax, ymax = np.ceil(wc.max(0)).astype(int)
    return xmin, ymin, xmax, ymax


def _inverse_map(im, H, bbox=None):
    """For every output pixel, where it comes from in the source (inverse warping)."""
    xmin, ymin, xmax, ymax = bbox if bbox is not None else output_bbox(im, H)
    xs, ys = np.meshgrid(np.arange(xmin, xmax + 1), np.arange(ymin, ymax + 1))
    src = apply_H(np.linalg.inv(H), np.stack([xs.ravel(), ys.ravel()], axis=1))
    sx = src[:, 0].reshape(xs.shape)
    sy = src[:, 1].reshape(xs.shape)
    h, w = im.shape[:2]
    valid = (sx >= 0) & (sx <= w - 1) & (sy >= 0) & (sy <= h - 1)  # alpha mask
    return sx, sy, valid, (xmin, ymin)


def warpImageNearestNeighbor(im, H, bbox=None):
    """Returns (warped image, alpha mask, (xmin, ymin) offset of output pixel [0,0])."""
    sx, sy, valid, offset = _inverse_map(im, H, bbox)
    out = np.zeros(sx.shape + im.shape[2:])
    xi = np.round(sx[valid]).astype(int)
    yi = np.round(sy[valid]).astype(int)
    out[valid] = im[yi, xi]
    return out, valid.astype(float), offset


def warpImageBilinear(im, H, bbox=None):
    """Returns (warped image, alpha mask, (xmin, ymin) offset of output pixel [0,0])."""
    sx, sy, valid, offset = _inverse_map(im, H, bbox)
    h, w = im.shape[:2]
    out = np.zeros(sx.shape + im.shape[2:])
    x, y = sx[valid], sy[valid]
    x0 = np.clip(np.floor(x).astype(int), 0, w - 2)  # keep x0+1 in bounds
    y0 = np.clip(np.floor(y).astype(int), 0, h - 2)
    dx, dy = x - x0, y - y0
    if im.ndim == 3:
        dx, dy = dx[:, None], dy[:, None]
    out[valid] = ((1 - dx) * (1 - dy) * im[y0, x0] +
                  dx * (1 - dy) * im[y0, x0 + 1] +
                  (1 - dx) * dy * im[y0 + 1, x0] +
                  dx * dy * im[y0 + 1, x0 + 1])
    return out, valid.astype(float), offset


# ---------------------------------------------------------------- rectification
def click_corners(im):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(im)
    ax.set_title("Click 4 corners: TL, TR, BR, BL")
    pts = []
    for i in range(4):
        (x, y), = plt.ginput(1, timeout=0)
        pts.append([x, y])
        ax.plot(x, y, "r+", ms=14)
        ax.text(x + 5, y - 5, "TL TR BR BL".split()[i], color="yellow", fontsize=12)
        fig.canvas.draw()
    plt.close(fig)
    return np.array(pts)


def rectify(im, src_pts, aspect, margin=0.5, max_out=3000):
    """Map the clicked quad to an axis-aligned rectangle of the given aspect (w/h).
    The rectangle is sized to roughly match the object's size in the photo.
    Output is limited to the rectangle plus `margin` (fraction of its size) on each side,
    so far-away parts of the scene don't blow up the canvas."""
    src_pts = np.asarray(src_pts, float)
    side_h = 0.5 * (np.linalg.norm(src_pts[3] - src_pts[0]) + np.linalg.norm(src_pts[2] - src_pts[1]))
    Hr = side_h
    Wr = aspect * Hr
    dst_pts = np.array([[0, 0], [Wr, 0], [Wr, Hr], [0, Hr]])
    H = computeH(src_pts, dst_pts)

    xmin, ymin, xmax, ymax = output_bbox(im, H)
    mx, my = margin * Wr, margin * Hr
    bbox = (int(max(xmin, -mx)), int(max(ymin, -my)),
            int(min(xmax, Wr + mx)), int(min(ymax, Hr + my)))
    assert (bbox[2] - bbox[0]) < max_out and (bbox[3] - bbox[1]) < max_out, f"output too big: {bbox}"
    return H, dst_pts, bbox


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--aspect", type=float, required=True, help="real width/height of the object")
    ap.add_argument("--out", default="out/rect")
    ap.add_argument("--pts", default=None, help="x1,y1,...,x4,y4 (TL,TR,BR,BL); skip clicking")
    ap.add_argument("--max_dim", type=int, default=1000)
    ap.add_argument("--margin", type=float, default=0.5)
    args = ap.parse_args()

    im = load(args.image, args.max_dim)
    if args.pts:
        src = np.array([float(v) for v in args.pts.split(",")]).reshape(4, 2)
    else:
        src = click_corners(im)
    print("corners (TL,TR,BR,BL):", np.round(src, 1).tolist())

    H, dst, bbox = rectify(im, src, args.aspect, args.margin)
    np.set_printoptions(suppress=True, precision=5)
    print("H:\n", H)

    t = time.perf_counter(); nn, alpha, off = warpImageNearestNeighbor(im, H, bbox); t_nn = time.perf_counter() - t
    t = time.perf_counter(); bil, _, _ = warpImageBilinear(im, H, bbox); t_bil = time.perf_counter() - t
    print(f"output {nn.shape[1]}x{nn.shape[0]}   NN: {t_nn:.3f}s   bilinear: {t_bil:.3f}s   "
          f"(bilinear {t_bil / t_nn:.1f}x slower)")

    os.makedirs(args.out, exist_ok=True)
    rgba = lambda img: np.dstack([np.clip(img, 0, 1), alpha])  # transparent where no data
    plt.imsave(f"{args.out}/nn.png", rgba(nn))
    plt.imsave(f"{args.out}/bilinear.png", rgba(bil))

    # rectangle outline in output coords
    rx = dst[[0, 1, 2, 3, 0], 0] - off[0]
    ry = dst[[0, 1, 2, 3, 0], 1] - off[1]

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(im); ax[0].plot(*src[[0, 1, 2, 3, 0]].T, "r-", lw=2); ax[0].set_title("original + clicked corners")
    ax[1].imshow(rgba(nn)); ax[1].plot(rx, ry, "r--", lw=1); ax[1].set_title(f"nearest neighbor ({t_nn:.2f}s)")
    ax[2].imshow(rgba(bil)); ax[2].plot(rx, ry, "r--", lw=1); ax[2].set_title(f"bilinear ({t_bil:.2f}s)")
    for a in ax: a.axis("off")
    fig.tight_layout(); fig.savefig(f"{args.out}/comparison.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # zoomed crop: a small window near the rectangle's top-left corner, upscaled so pixels are visible
    cx, cy = int(-off[0] + 0.25 * (dst[1, 0])), int(-off[1] + 0.25 * dst[2, 1])
    r = max(20, int(0.08 * dst[1, 0]))
    sl = (slice(max(cy - r, 0), cy + r), slice(max(cx - r, 0), cx + r))
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(np.clip(nn[sl], 0, 1), interpolation="nearest"); ax[0].set_title("nearest neighbor (zoom)")
    ax[1].imshow(np.clip(bil[sl], 0, 1), interpolation="nearest"); ax[1].set_title("bilinear (zoom)")
    for a in ax: a.axis("off")
    fig.tight_layout(); fig.savefig(f"{args.out}/crop_compare.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"saved -> {args.out}/")


if __name__ == "__main__":
    main()
