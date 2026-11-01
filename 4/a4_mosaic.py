"""
CS180 Project 4A - A.4: blend images into a mosaic.

Procedure
1. Keep im2 as the reference (unwarped). H = computeH(im1_pts, im2_pts) maps im1 -> im2's frame.
2. Canvas = union of the warped im1 bounding box and im2's box (one-shot).
3. Each image gets a weight ("alpha") map in its OWN frame: distance to the nearest image edge
   (= bwdist of the border), so 0 at the edges, max in the center. The weight map is warped
   with exactly the same H as the image, so weights and pixels stay aligned.
4. Blend:
   - feather:  out = sum(w_i * im_i) / sum(w_i)                       (weighted averaging)
   - 2-band:   low frequencies (Gaussian blur) are feathered as above;
               high frequencies (im - blur) come from whichever image has the larger weight
               -> smooth color transition, no ghosting of fine detail.

Usage:  python a4_mosaic.py          (runs the three sets below, writes to out/mosaic_<name>/)
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from scipy.ndimage import gaussian_filter 

from a3_warp import computeH, apply_H, warpImageBilinear, output_bbox

SETS = {
    "painting": {"im1_name": "IMG_0875", "im2_name": "IMG_0874",
                 "im1Points": [[525,501],[468,457],[429,422],[406,396],[388,361],[409,528],[437,524],[471,517],[329,378],[536,407],[380,287]],
                 "im2Points": [[239,510],[192,469],[154,439],[130,411],[111,373],[125,546],[157,540],[189,531],[42,390],[252,423],[104,297]]},
    "tree":     {"im1_name": "IMG_0894", "im2_name": "IMG_0895",
                 "im1Points": [[98,204],[131,221],[162,236],[69,475],[291,582],[341,501],[449,481],[314,82],[296,68],[296,100],[280,88]],
                 "im2Points": [[171,199],[203,214],[232,226],[156,462],[373,565],[419,486],[536,465],[382,59],[362,45],[363,78],[347,65]]},
    "gateway":  {"im1_name": "IMG_0904", "im2_name": "IMG_0905",
                 "im1Points": [[426,578],[423,632],[253,613],[371,650],[397,709],[130,330],[399,331],[494,261],[304,285],[424,476],[248,437]],
                 "im2Points": [[355,570],[351,623],[183,613],[301,643],[326,700],[50,322],[329,330],[419,264],[234,279],[354,471],[177,436]]},
}
PIC_DIR = "pics"


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
    return canvas[..., :3], canvas[..., 3]


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
    for a, im, p, t in [(ax[0], im1, d["im1Points"], d["im1_name"]), (ax[1], im2, d["im2Points"], d["im2_name"] + " (reference)")]:
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


if __name__ == "__main__":
    for name, d in SETS.items():
        make_mosaic(name, d)
