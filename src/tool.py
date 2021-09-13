import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.spatial import distance as dist

def show(imgs, h=8):
    n = len(imgs)
    if n == 1:
        plt.figure(figsize=(h, 6), dpi=80)
        plt.axis("off")
        plt.imshow(cv2.cvtColor(imgs[0], cv2.COLOR_BGR2RGB))
        return
    f, axs = plt.subplots(1, n,figsize=(h,6*n))
    for i in range(n):
        axs[i].axis('off')
        axs[i].imshow(cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB))
        
def annotate_points(img, points):
    mask_temp = img.copy()
    for c in points:
        c = np.array(c).flatten().astype(int)
        mask_temp = cv2.circle(mask_temp, tuple(c), 8, (0,255,0), thickness=3)
    show([mask_temp])
    
def white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

def clockwise_corners(pts):
    pts = np.array(pts.copy())
    xSorted = pts[np.argsort(pts[:, 0]), :]
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
    return np.array([tl, tr, br, bl])

def warpBox(image,
            box,
            target_height=None,
            target_width=None,
            return_transform=False):
    """Warp a boxed region in an image given by a set of four points into
    a rectangle with a specified width and height. Useful for taking crops
    of distorted or rotated text.
    Args:
        image: The image from which to take the box
        box: A list of four points starting in the top left
            corner and moving clockwise.
        target_height: The height of the output rectangle
        target_width: The width of the output rectangle
        return_transform: Whether to return the transformation
            matrix with the image.
    """
    box = np.float32(clockwise_corners(box))
    w, h = image.shape[1], image.shape[0]
    assert (
        (target_width is None and target_height is None)
        or (target_width is not None and target_height is not None)), \
            'Either both or neither of target width and height must be provided.'
    if target_width is None and target_height is None:
        target_width = w
        target_height = h
    M = cv2.getPerspectiveTransform(src=box, dst=np.array([[0, 0], [target_width, 0], [target_width, target_height], [0, target_height]]).astype('float32'))
    full = cv2.warpPerspective(image, M, dsize=(int(target_width), int(target_height)))
    if return_transform:
        return full, M
    return full