"""
CS180 Project 4A: Image Warping and Mosaicing  (A.1 - A.4 in one file)

Run from the project folder (photos in pics/):
    python3 "Project 4A.py"                 # A.2 printout + all three A.4 mosaics
    python3 "Project 4A.py" a2              # A.2 only: system of equations, H, reprojection error
    python3 "Project 4A.py" mosaic          # A.4 only -> out/mosaic_<name>/
    python3 "Project 4A.py" rect pics/IMG_0907.jpg --aspect 0.67 --out out/rect2
                                            # A.3 rectification: click TL, TR, BR, BL corners

Conventions: points are (x, y) = (col, row); pixel centers sit at integer coords; im[row, col] = im[y, x].
Needs numpy, matplotlib, pillow, scipy (scipy only for gaussian_filter in the 2-band blend).
"""
import argparse
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from scipy.ndimage import gaussian_filter  # blurring only; all warping is implemented below

try:  # optional iPhone HEIC support
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass


# =====================================================================
# A.1  Pictures + correspondences (clicked on the 600x800 photos in pics/)
# =====================================================================
PIC_DIR = "pics"

painting = {"im1_name":"IMG_0875","im2_name":"IMG_0874",
 "im1Points":[[525,501],[468,457],[429,422],[406,396],[388,361],[409,528],[437,524],[471,517],[329,378],[536,407],[380,287]],
 "im2Points":[[239,510],[192,469],[154,439],[130,411],[111,373],[125,546],[157,540],[189,531],[42,390],[252,423],[104,297]]}

tree = {"im1_name":"IMG_0894","im2_name":"IMG_0895",
        "im1Points":[[98,204],[131,221],[162,236],[69,475],[291,582],[341,501],[449,481],[314,82],[296,68],[296,100],[280,88]],
        "im2Points":[[171,199],[203,214],[232,226],[156,462],[373,565],[419,486],[536,465],[382,59],[362,45],[363,78],[347,65]]}

gateway = {"im1_name":"IMG_0904","im2_name":"IMG_0905",
           "im1Points":[[426,578],[423,632],[253,613],[371,650],[397,709],[130,330],[399,331],[494,261],[304,285],[424,476],[248,437]],
           "im2Points":[[355,570],[351,623],[183,613],[301,643],[326,700],[50,322],[329,330],[419,264],[234,279],[354,471],[177,436]]}

SETS = {"painting": painting, "tree": tree, "gateway": gateway}


# =====================================================================
# A.2  Recover homographies
# =====================================================================
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


def print_a2():
    """Deliverable printout: system of equations, recovered H, reprojection error."""
    np.set_printoptions(suppress=True, precision=6, linewidth=150)
    for name, d in SETS.items():
        p1, p2 = np.array(d["im1Points"], float), np.array(d["im2Points"], float)
        A, b = build_system(p1, p2)
        H = computeH(p1, p2)
        err = np.linalg.norm(apply_H(H, p1) - p2, axis=1)
        print(f"=== {name} ===  A: {A.shape}, b: {b.shape}")
        print("A (first 4 rows):\n", A[:4]); print("b (first 4):", b[:4])
        print("H:\n", H)
        print("reprojection error (px):", err.round(1), " mean:", err.mean().round(2), "\n")


# =====================================================================
# A.3  Warp the images (inverse warping) + rectification
# =====================================================================
def load(path, max_dim=1000):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    s = max_dim / max(im.size) if max_dim else 1
    if s < 1:
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    return np.asarray(im, dtype=np.float64) / 255.0


# ---------------------------------------------------------------- A.2

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

def run_rectify(argv=None):
    ap = argparse.ArgumentParser(prog="Project 4A.py rect")
    ap.add_argument("image")
    ap.add_argument("--aspect", type=float, required=True, help="real width/height of the object")
    ap.add_argument("--out", default="out/rect")
    ap.add_argument("--pts", default=None, help="x1,y1,...,x4,y4 (TL,TR,BR,BL); skip clicking")
    ap.add_argument("--max_dim", type=int, default=1000)
    ap.add_argument("--margin", type=float, default=0.5)
    args = ap.parse_args(argv)

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


# =====================================================================
# A.4  Blend the images into a mosaic
# =====================================================================
def load_native(path):
    """Load at the file's own size (the points were clicked at this size)."""
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return np.asarray(im, dtype=np.float64) / 255.0


def edge_weight(h, w):
    """Distance to nearest image border (bwdist of the border), normalized to max 1."""
    y, x = np.mgrid[0:h, 0:w]
    d = np.minimum.reduce([x, w - 1 - x, y, h - 1 - y]).astype(float) + 1
    return d / d.max()


def warp_onto_canvas(im, H, canvas_box):
    """Warp image + its weight map with H, then place them on the shared canvas."""
    h, w = im.shape[:2]
    rgbw = np.dstack([im, edge_weight(h, w)])          # weight rides along as a 4th channel
    warped, alpha, (ox, oy) = warpImageBilinear(rgbw, H)
    cx0, cy0, cx1, cy1 = canvas_box
    canvas = np.zeros((cy1 - cy0 + 1, cx1 - cx0 + 1, 4))
    ys, xs = oy - cy0, ox - cx0
    canvas[ys:ys + warped.shape[0], xs:xs + warped.shape[1]] = warped * alpha[..., None]
    return canvas[..., :3], canvas[..., 3]              # image, weight


def blend_feather(ims, ws):
    num = sum(w[..., None] * im for im, w in zip(ims, ws))
    den = sum(ws)[..., None]
    return np.where(den > 0, num / np.maximum(den, 1e-8), 0)


def masked_blur(im, mask, sigma):
    """Blur only using pixels inside the mask (so the black outside doesn't bleed in)."""
    num = np.dstack([gaussian_filter(im[..., c] * mask, sigma) for c in range(3)])
    den = gaussian_filter(mask, sigma)[..., None]
    return np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0)


def blend_two_band(ims, ws, sigma=4):
    masks = [(w > 0).astype(float) for w in ws]
    lows = [masked_blur(im, m, sigma) for im, m in zip(ims, masks)]
    highs = [(im - lo) * m[..., None] for im, lo, m in zip(ims, lows, masks)]
    # low band: smooth weighted average (feathering)
    low = blend_feather(lows, ws)
    # high band: take it from the image with the largest weight at each pixel
    best = np.argmax(np.stack(ws), axis=0)
    high = np.zeros_like(ims[0])
    for i, hi in enumerate(highs):
        high[best == i] = hi[best == i]
    covered = (sum(ws) > 0)[..., None]
    return np.clip(low + high, 0, 1) * covered


def make_mosaic(name, d, out_root="out"):
    im1 = load_native(os.path.join(PIC_DIR, d["im1_name"] + ".jpg"))
    im2 = load_native(os.path.join(PIC_DIR, d["im2_name"] + ".jpg"))
    H = computeH(d["im1Points"], d["im2Points"])        # im1 -> im2 (reference)
    I = np.eye(3)

    b1 = output_bbox(im1, H)
    b2 = output_bbox(im2, I)
    canvas_box = (min(b1[0], b2[0]), min(b1[1], b2[1]), max(b1[2], b2[2]), max(b1[3], b2[3]))

    w1_im, w1 = warp_onto_canvas(im1, H, canvas_box)
    w2_im, w2 = warp_onto_canvas(im2, I, canvas_box)
    ims, ws = [w1_im, w2_im], [w1, w2]

    naive = np.where((w2 > 0)[..., None], w2_im, w1_im)  # overwrite: shows the hard seam
    feather = blend_feather(ims, ws)
    twoband = blend_two_band(ims, ws)

    out = os.path.join(out_root, f"mosaic_{name}")
    os.makedirs(out, exist_ok=True)
    for tag, img in [("naive", naive), ("feather", feather), ("twoband", twoband)]:
        plt.imsave(f"{out}/{tag}.jpg", np.clip(img, 0, 1))

    fig, ax = plt.subplots(1, 2, figsize=(10, 6.5))
    for a, im, p, t in [(ax[0], im1, d["im1Points"], "image 1"), (ax[1], im2, d["im2Points"], "image 2 (reference)")]:
        p = np.array(p); a.imshow(im); a.scatter(p[:, 0], p[:, 1], c="r", marker="+", s=80)
        for i, (x, y) in enumerate(p): a.text(x + 6, y - 6, str(i), color="yellow", fontsize=9)
        a.set_title(t); a.axis("off")
    fig.tight_layout(); fig.savefig(f"{out}/sources.jpg", dpi=120, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(18, 7))
    for a, img, t in [(ax[0], naive, "overwrite (no blending)"), (ax[1], feather, "feathered (weighted avg)"), (ax[2], twoband, "2-band blend")]:
        a.imshow(np.clip(img, 0, 1)); a.set_title(t); a.axis("off")
    fig.tight_layout(); fig.savefig(f"{out}/compare.jpg", dpi=120, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(w1, cmap="gray"); ax[0].set_title("weight: warped im1")
    ax[1].imshow(w2, cmap="gray"); ax[1].set_title("weight: im2")
    for a in ax: a.axis("off")
    fig.tight_layout(); fig.savefig(f"{out}/weights.jpg", dpi=100, bbox_inches="tight"); plt.close(fig)

    np.set_printoptions(suppress=True, precision=5)
    print(f"[{name}] canvas {canvas_box[2]-canvas_box[0]+1}x{canvas_box[3]-canvas_box[1]+1}  H=\n{H}\n -> {out}/")


def run_mosaics():
    for name, d in SETS.items():
        make_mosaic(name, d)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "a2":
        print_a2()
    elif cmd == "mosaic":
        run_mosaics()
    elif cmd == "rect":
        run_rectify(sys.argv[2:])
    elif cmd == "all":
        print_a2()
        run_mosaics()
    else:
        print(__doc__)
