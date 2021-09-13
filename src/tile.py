import numpy as np
from matplotlib import pyplot as plt
import cv2
from src.tool import clockwise_corners

def rotate_points(pts, l, h):
    ret = []
    for i in range(1, len(pts)):
        p = pts[i]
        ret.append([p[1], l-p[0]])
    ret.append(pts[0])
    return np.array(ret)


def four_point_transform(image, pts, normalise=False):
    ### modified code from https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
    rect = pts
    (tl, tr, br, bl) = rect
    padding = int(max([tl[0], tl[1], len(image)-br[0], len(image)-br[1]]))
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB)) + padding
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB)) + padding
    if normalise:
        maxHeight = 401
        maxWidth = 401
    dst = np.array([
        [padding, padding],
        [maxWidth - 1, padding],
        [maxWidth - 1, maxHeight - 1],
        [padding, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth + padding, maxHeight + padding))
    # return the warped image
    return warped, dst.astype(int)

def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
    x, y = pos

    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return
    channels = img.shape[2]

    alpha = np.array(alpha_mask==255)[y1o:y2o, x1o:x2o]
    alpha_inv = np.invert(np.array(alpha))

    for c in range(channels):
        img1 = img_overlay[y1o:y2o, x1o:x2o, c]
        img2 = img[y1:y2, x1:x2, c]
        img[y1:y2, x1:x2, c] = (alpha * img1 + alpha_inv * img2)
    return img

def check_between(p1, p0, p2, center_tile):
    return round(get_angle(p1, center_tile, p0)) + round(get_angle(p0, center_tile, p2)) - round(get_angle(p1, center_tile, p2))<=3

def annotate(contour, corners, center_tile=None):
    labels = []
    contour = contour.reshape((contour.shape[0], contour.shape[-1]))
    corners = corners.reshape((corners.shape[0], corners.shape[-1]))
    if center_tile is None:
        center_tile = np.mean(corners, axis=0)
    for p0 in contour:
        c = 3
        for i in range(3):
            p1 = corners[i]
            p2 = corners[i+1]
            if check_between(p1, p0, p2, center_tile):
                c = i
                break
        labels.append(c)
    return np.array(labels)

def dis_point_line(p1, p2, p0):
    return np.linalg.norm(np.cross(p2-p1, p1-p0))/np.linalg.norm(p2-p1)

def label_diff(contour, corners, labels):
    dd, v = [], []
    for label in range(4):
        l = [corners[0], corners[1], corners[2], corners[3], corners[0]]
        p1, p2 = l[label], l[label+1]
        p3 = np.mean(corners, axis=0)
        d = dis_point_line(p1, p2, p3)
        p4 = p3 + (p2-p1)/np.linalg.norm(p2-p1)
        d2 = np.mean([dis_point_line(p3, p4, x) for x in contour[labels==label]], axis=-1)
        v.append(np.var([dis_point_line(p3, p4, x) for x in contour[labels==label]], axis=-1))
        dd.append(d2-d)
    return dd, v

class Tile:
    def __init__(self, tile_img, tile_mask, tile_corners, threshold=200):
        self.orig_img = tile_img
        self.orig_mask = tile_mask
        self.orig_corners = np.array(clockwise_corners(tile_corners))
        self.img, self.corners = four_point_transform(self.orig_img, self.orig_corners)
        self.mask, _ = four_point_transform(np.float32(self.orig_mask), self.orig_corners)
        mask = self.mask.astype(np.uint8)
        contour = get_smooth_contour(mask)
        self.contour = contour.reshape((contour.shape[0], contour.shape[-1]))
        self.labels = annotate(contour, self.corners)
        self.dd, self.v = label_diff(contour, self.corners, self.labels)
        self.threshold = threshold   

    def show(self):
        img = self.img.copy()
        for c in self.corners:
            img = cv2.circle(img, tuple(c), 8, (230,0,0), thickness=3)
        f, axs = plt.subplots(1, 3, figsize=(12,18))
        axs[0].axis('off')
        axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axs[0].title.set_text('tile')
        axs[1].imshow(self.mask, cmap='gray')
        axs[1].axis('off')
        axs[1].title.set_text('tile mask')
        img = self.img.copy()
        for i in range(4):
            color = (230,230,230)
            if self.v[i]>self.threshold:
                color = (230,0,0)
            for c in self.contour[self.labels==i]:
                img = cv2.circle(img, tuple(c), 8, color, thickness=5)
        axs[2].imshow(img)
        axs[2].axis('off')
        axs[2].title.set_text('tile contour')

    def rotate(self, num=1):
        for i in range(num):
            self.img = np.rot90(self.img)
            self.mask = np.rot90(self.mask)
            self.corners = rotate_points(self.corners, len(self.img), len(self.img[0]))
            self.contour = np.array([[b[1], len(self.img)-b[0]] for b in self.contour])
    
    def scale(self, scale=1):
        w, h, _ = self.img.shape
        self.img = cv2.resize(self.img, (int(w*scale), int(h*scale)))
        self.mask = cv2.resize(self.mask, (int(w*scale), int(h*scale)))
        self.corners = np.array(self.corners*scale, dtype=int)