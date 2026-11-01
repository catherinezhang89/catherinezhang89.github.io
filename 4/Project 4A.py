# Project 4A
import numpy as np
# A.1
painting = {"im1_name":"IMG_0875","im2_name":"IMG_0874",
 "im1Points":[[525,501],[468,457],[429,422],[406,396],[388,361],[409,528],[437,524],[471,517],[329,378],[536,407],[380,287]],
 "im2Points":[[239,510],[192,469],[154,439],[130,411],[111,373],[125,546],[157,540],[189,531],[42,390],[252,423],[104,297]]}

tree = {"im1_name":"IMG_0894","im2_name":"IMG_0895",
        "im1Points":[[98,204],[131,221],[162,236],[69,475],[291,582],[341,501],[449,481],[314,82],[296,68],[296,100],[280,88]],
        "im2Points":[[171,199],[203,214],[232,226],[156,462],[373,565],[419,486],[536,465],[382,59],[362,45],[363,78],[347,65]]}

gateway = {"im1_name":"IMG_0904","im2_name":"IMG_0905",
           "im1Points":[[426,578],[423,632],[253,613],[371,650],[397,709],[130,330],[399,331],[494,261],[304,285],[424,476],[248,437]],
           "im2Points":[[355,570],[351,623],[183,613],[301,643],[326,700],[50,322],[329,330],[419,264],[234,279],[354,471],[177,436]]}


# Q2

def build_system(im1_pts, im2_pts):
    rows, rhs = [], []
    for (x, y), (xp, yp) in zip(im1_pts, im2_pts):
        rows.append([x, y, 1, 0, 0, 0, -x*xp, -y*xp]); rhs.append(xp)
        rows.append([0, 0, 0, x, y, 1, -x*yp, -y*yp]); rhs.append(yp)
    return np.array(rows, float), np.array(rhs, float)   # A: (2n, 8), b: (2n,)

def solve_least_squares(A, b):
    return np.linalg.solve(A.T @ A, A.T @ b)   # returns h: (8,)

def computeH(im1, im2):
    A, b = build_system(im1, im2)
    h = solve_least_squares(A, b)
    return np.append(h, 1).reshape(3,3)

def apply_H(H, pts):
    P = np.hstack([pts, np.ones((len(pts), 1))])   # n×3, homogeneous
    Q = P @ H.T                                     # same as (H @ P.T).T
    return Q[:, :2] / Q[:, 2:3]                     # divide by w


# # im1 and 2 pts are n-by-2 matrices holding the (x,y) locations of n point correspondences from the two images
# # H is the recovered 3x3 homography matrix

# np.set_printoptions(suppress=True, precision=6, linewidth=150)


# for name, d in [("painting", painting), ("tree", tree), ("gateway", gateway)]:
#     p1, p2 = np.array(d["im1Points"], float), np.array(d["im2Points"], float)
#     A, b = build_system(p1, p2)
#     H = computeH(p1, p2)
#     err = np.linalg.norm(apply_H(H, p1) - p2, axis=1)
#     print(f"=== {name} ===  A: {A.shape}, b: {b.shape}")
#     print("A (first 4 rows):\n", A[:4]); print("b (first 4):", b[:4])
#     print("H:\n", H)
#     print("reprojection error (px):", err.round(1), "\n")

# A.3 Warping

# imwarped_nn = warpImageNearestNeighbor(im,H)
# imwarped_bil = warpImageBilinear(im,H)


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

 