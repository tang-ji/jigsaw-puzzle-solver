import numpy as np
import cv2, operator
from matplotlib import pyplot as plt
import scipy.ndimage as ndimage

def show(imgs):
    n = len(imgs)
    if n == 1:
        plt.figure(figsize=(8, 6), dpi=80)
        plt.axis("off")
        plt.imshow(cv2.cvtColor(imgs[0], cv2.COLOR_BGR2RGB))
        return
    f, axs = plt.subplots(1, n,figsize=(8,6*n))
    for i in range(n):
        axs[i].axis('off')
        axs[i].imshow(cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB))
        
def annotate_points(img, points):
    mask_temp = img.copy()
    for c in points:
        c =c.astype(int)
        mask_temp = cv2.circle(mask_temp, tuple(c), 8, (0,0,0), thickness=3)
    show([mask_temp])
    
def white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

def clockwise_corners(corners):
    corners_ordered = sorted(corners, key=lambda x: x[0]+x[1])
    ret = []
    ret.append(corners_ordered[0])
    corners_ordered = sorted(corners_ordered[1:], key=lambda x: x[1])
    ret.append(corners_ordered[0])
    corners_ordered = sorted(corners_ordered[1:], key=lambda x: -x[0]-x[1])
    ret.append(corners_ordered[0])
    ret.append(corners_ordered[1])
    return ret

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


def get_tile_corners(mask):
    canny = cv2.Canny(mask, 100, 255, 1)
    corners = cv2.goodFeaturesToTrack(canny, 30, 0.1, 100)
    corners = [tuple(corner.flatten()) for corner in corners]
    tile_center = ndimage.center_of_mass(mask)
    tile_center = tuple(np.round(tile_center).astype(np.int))
    
    def distance(c1, c2):
        return (c1[0]-c2[0]) ** 2 + (c1[1]-c2[1]) ** 2
    
    lt, rt, rb, lb = tile_center, tile_center, tile_center, tile_center
    lt_d, rt_d, rb_d, lb_d = 0, 0, 0, 0
    for corner in corners:
        if corner[0] < tile_center[0] and corner[1] < tile_center[1] and distance(corner, tile_center) > lt_d:
            lt = corner
            lt_d = distance(corner, tile_center)
        if corner[0] < tile_center[0] and corner[1] > tile_center[1] and distance(corner, tile_center) > rt_d:
            rt = corner
            rt_d = distance(corner, tile_center)
        if corner[0] > tile_center[0] and corner[1] > tile_center[1] and distance(corner, tile_center) > rb_d:
            rb = corner
            rb_d = distance(corner, tile_center)
        if corner[0] > tile_center[0] and corner[1] < tile_center[1] and distance(corner, tile_center) > lb_d:
            lb = corner
            lb_d = distance(corner, tile_center)
    return np.array([lt, rt, rb, lb], dtype=int)

def get_warp_image(img, margin=9):
    orig_width = len(img[0])
    orig_height = len(img)
    target = 800
    scale = target/orig_width if orig_height > orig_width else target/orig_height

    resized = cv2.resize(img, None, fx=scale, fy=scale)
    img_balanced = white_balance(resized)
    blurred = blur(img_balanced)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    edges = cv2.threshold(s, 15, 255, cv2.THRESH_BINARY_INV)[1]

    contour_info = [(c, cv2.contourArea(c),) for c in cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]]
    mask = np.zeros(list(map(operator.add, edges.shape, (margin*2,margin*2,0))), dtype = np.uint8)
    for contour in contour_info:
        mask = cv2.fillConvexPoly(mask, contour[0]+margin, (255))
    corners = get_tile_corners(mask)
    corners = [tuple((corner-margin)/scale) for corner in corners]
    
    return warpBox(img,corners)

def blur(img, filter_size=9):
    return cv2.GaussianBlur(img, (filter_size, filter_size), 0)

def find_outline_rough(img):
    blurred = blur(img)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)

    edges = cv2.threshold(s, 20, 100, cv2.THRESH_BINARY_INV)[1]

    contour_info = [(c, cv2.contourArea(c),) for c in cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]]

    image_area = img.shape[0] * img.shape[1]  
    max_area = 0.50
    max_area = max_area * image_area

    mask = np.ones(edges.shape, dtype = np.uint8)*255
    for contour in contour_info:
        if contour[1] < max_area:
            mask = cv2.fillPoly(mask, [contour[0].reshape(contour[0].shape[0], contour[0].shape[2])], (0))
    return mask

# Color KMeans
from sklearn.cluster import KMeans

def plot_colors(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype = "uint8")
    startX = 0
    for (percent, color) in zip(hist, centroids):
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
            color.astype("uint8").tolist(), -1)
        startX = endX
    plt.figure()
    plt.axis("off")
    plt.imshow(bar)
    plt.show()
    return bar

def centroid_histogram(clt):
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins = numLabels)
    hist = hist.astype("float")
    hist /= hist.sum()
    return hist

def kmeans_color_purifier(mask, img_balanced, n_samples=20000, n_clusters=40, t=0.8, show=True):
    colors = []
    for _ in range(n_samples):
        i, j = np.random.randint(0, len(mask)), np.random.randint(0, len(mask[0]))
        if mask[i][j] > 0:
            colors.append(img_balanced[i][j])

    c_var = np.var(colors, axis=0)
    
    clt = KMeans(n_clusters=n_clusters)
    clt.fit(colors)
    if show:
        hist = centroid_histogram(clt)
        bar = plot_colors(hist, clt.cluster_centers_)
    
    img_overlap = np.zeros(mask.shape, dtype = "uint8")
    for c_mean in clt.cluster_centers_:
        m = cv2.inRange(img_balanced, c_mean-0.2*np.sqrt(c_var), c_mean+0.2*np.sqrt(c_var))
        v = np.sum((255-m == 0)*(255-mask == 0))/np.sum((255-m == 0))
        # verify the color not in the puzzle blocks
        if v > t:
            img_overlap |= m
    return img_overlap

# Pieces searching and coloring
def pieces_searching(img_overlap):
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    bin_clo = cv2.dilate(img_overlap, kernel2, iterations=1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(255-bin_clo, connectivity=4)
    # num_labels
    # stats: x、y、width、height, area
    # centroids: centers of components
    pieces = []
    output = np.zeros((img_overlap.shape[0], img_overlap.shape[1], 3), np.uint8)
    for i in range(1, num_labels):
        if stats[i][4] < img_overlap.shape[0] * img_overlap.shape[1] * 0.005:
            continue
        m = labels == i
        pieces.append(m)
        output[:, :, 0][m] = np.random.randint(0, 255)
        output[:, :, 1][m] = np.random.randint(0, 255)
        output[:, :, 2][m] = np.random.randint(0, 255)
    return pieces, output