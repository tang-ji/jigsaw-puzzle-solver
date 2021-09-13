import numpy as np
from matplotlib import pyplot as plt
import cv2
from src.tool import clockwise_corners

def rotate_points(pts, l, h):
    a,b,c,d = pts
    return np.array([[b[1], l-b[0]], [c[1], l-c[0]], [d[1], l-d[0]], [a[1], l-a[0]]])


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

class Tile:

    def __init__(self, tile_img, tile_mask, tile_corners):
        self.orig_img = tile_img
        self.orig_mask = tile_mask
        self.orig_corners = np.array(clockwise_corners(tile_corners))
        self.img, self.corners = four_point_transform(self.orig_img, self.orig_corners)
        self.mask, _ = four_point_transform(np.float32(self.orig_mask), self.orig_corners)

    def show(self):
        img = self.img.copy()
        for c in self.corners:
            img = cv2.circle(img, tuple(c), 8, (255,0,0), thickness=3)

        plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cmap='gray')
        plt.title('tile'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(self.mask, cmap='gray')
        plt.title('tile mask'), plt.xticks([]), plt.yticks([])
        plt.show()

    def rotate(self, num=1):
        for i in range(num):
            self.img = np.rot90(self.img)
            self.mask = np.rot90(self.mask)
            self.corners = rotate_points(self.corners, len(self.img), len(self.img[0]))
    
    def scale(self, scale=1):
        w, h, _ = self.img.shape
        self.img = cv2.resize(self.img, (int(w*scale), int(h*scale)))
        self.mask = cv2.resize(self.mask, (int(w*scale), int(h*scale)))
        self.corners = np.array(self.corners*scale, dtype=int)
        