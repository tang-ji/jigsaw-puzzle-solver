from scipy.interpolate import splprep, splev
import cv2, operator
from matplotlib import pyplot as plt
import numpy as np
import scipy.ndimage as ndimage
from src.tool import white_balance, warpBox

def get_background_corners(mask):
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
    corners = get_background_corners(mask)
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
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50), color.astype("uint8").tolist(), -1)
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

def kmeans_color_purifier(mask, img_balanced, n_samples=20000, n_clusters=40, t=0.9, show=True):
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
        if stats[i][4] < img_overlap.shape[0] * img_overlap.shape[1] * 0.003:
            continue
        m = labels == i
        pieces.append(m)
        output[:, :, 0][m] = np.random.randint(0, 255)
        output[:, :, 1][m] = np.random.randint(0, 255)
        output[:, :, 2][m] = np.random.randint(0, 255)
    return pieces, output

# def crop_piece(piece, img_balanced, margin=10):
#     piece = piece.copy().astype(np.uint8)
#     piece[piece==True]=255
#     piece[piece==False]=0
#     contour_info = [(c, cv2.contourArea(c),) for c in cv2.findContours(piece, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]]
#     image_area = piece.shape[0] * piece.shape[1]  
#     min_area = 0.0001 * image_area
#     points = np.array([])
#     for contour in contour_info:
#         if contour[1] > min_area:
#             points = contour[0].reshape(contour[0].shape[0], contour[0].shape[2])
#     minX = max(min(points[:, 1])-margin, 0)
#     maxX = min(max(points[:, 1])+margin, len(piece))
#     minY= max(min(points[:, 0])-margin, 0)
#     maxY= min(max(points[:, 0])+margin, len(piece[0]))
#     return piece[minX:maxX, minY:maxY], img_balanced[minX:maxX, minY:maxY]

# new version
def crop_piece(piece, img_balanced, margin=10):
    x_set, y_set = np.where(np.any(piece, axis=1)), np.where(np.any(piece, axis=0))
    minX = max(np.min(x_set)-margin, 0)
    maxX = min(np.max(x_set)+margin, len(piece))
    minY= max(np.min(y_set)-margin, 0)
    maxY= min(np.max(y_set)+margin, len(piece[0]))
    piece = piece[minX:maxX, minY:maxY].copy().astype(np.uint8)
    piece[piece==1]=255
    return piece, img_balanced[minX:maxX, minY:maxY].copy()

def get_smooth_contour(piece, image_area=None, n_polys=150):
    piece = piece.copy()
    piece = cv2.dilate(piece, None, iterations=3)
    piece = cv2.erode(piece, None, iterations=3)
    contour_info = [(c, cv2.contourArea(c),) for c in cv2.findContours(piece, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]]
    if image_area is None:
        image_area = piece.shape[0] * piece.shape[1]
    min_area = 0.0001 * image_area

    smoothened = []
    for contour in contour_info:
        if contour[1] > min_area:
            contour = contour[0]
            x,y = contour.T
            x = x.tolist()[0]
            y = y.tolist()[0]
            tck, u = splprep(np.array([x,y]), u=None, s=1.0, per=0)
            u_new = np.linspace(u.min(), u.max(), n_polys)
            x_new, y_new = splev(u_new, tck, der=0)
            res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new,y_new)]
            smoothened = np.asarray(res_array, dtype=np.int32)
            continue
    return smoothened

def refine_piece(piece, image_area=None, n_polys=150):
    mask = np.zeros(piece.shape, dtype = np.uint8)
    smoothened = get_smooth_contour(piece, image_area=image_area, n_polys=n_polys)
    mask = cv2.fillPoly(mask, [smoothened], (255))
    return mask

def get_vector(p1,p2):
    if type(p1).__module__ != np.__name__:
        p1 = np.array(p1)
    if type(p2).__module__ != np.__name__:
        p2 = np.array(p2)
    return p2-p1


def get_angle(p1,p2,p3):
    """calculate angle between p2_p3 and p2_p3"""
    p2p1 = get_vector(p2,p1)
    p2p3 = get_vector(p2,p3)
    cosine_angle = np.dot(p2p1, p2p3) / (np.linalg.norm(p2p1) * np.linalg.norm(p2p3))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def get_distance(pt1,pt2):
    return ((pt2[0]-pt1[0]) ** 2 + (pt2[1]-pt1[1]) ** 2) ** (1/2)

def get_90deg_corners(pt, corners, rule1, rule2, margin = 5):
    """calculate all corners that are 90 degrees from pt, where the corners c1, c2 must comply with rule1, rule2 """
    res = []
    used = []
    for c1 in corners:
        if not rule1(c1):
            continue
        for c2 in corners:
            if np.array_equal(c1,c2) or np.array_equal(c2,pt) or np.array_equal(c1,pt) or \
                    not rule2(c2) or \
                    tuple(c2) in used:
                continue
            if 90 - margin < get_angle(c1,pt,c2) < 90 + margin:
                res.append((c1,c2))
                used.append(tuple(c1))
    return res

def refine_corners(corners, tile_center, angle_diff = 20):
    corners_refined = []
    for c1 in corners:
        c1 = tuple(c1)
        f = True
        remove_l = []
        for c2 in corners_refined:
            if c2 == c1:
                f = False
                break
            if get_angle(c2, tile_center, c1) < angle_diff:
                if get_distance(c1,tile_center) > get_distance(c2,tile_center):
                    remove_l.append(tuple(c2))
                else:
                    f = False
                    break
        if f:
            corners_refined.append(tuple(c1))
            for r in remove_l:
                corners_refined.remove(r)
    corners_refined = np.array(list(corners_refined))
    return corners_refined

def get_tile_corners(mask, angle_margin=15, angle_diff_ls_thres=200):
    corners = cv2.goodFeaturesToTrack(mask, 80, 0.05, 10)
    corners = corners.reshape(corners.shape[0], corners.shape[-1])
    tile_center = np.mean(corners, axis=0)
    corners = refine_corners(corners, tile_center)
#     tile_center = ndimage.center_of_mass(mask)
#     tile_center = tuple(np.round(tile_center).astype(np.int))
    ang_opt = np.array([90, 90, 90, 90])
    ang_diff_ls = None
    side_var = None
    tile_corners = []
    ret = []

    for c1 in corners:
        if c1[1] <= tile_center[1]+mask.shape[1] and c1[0] <= tile_center[0]+mask.shape[0]:
            candidates1 = get_90deg_corners(c1, corners, lambda c: c[1] >= tile_center[1]-mask.shape[1]*0.2, lambda c: c[0] >= tile_center[0]-mask.shape[0]*0.2, angle_margin)
            for c2, c4 in candidates1:
                for c3 in corners:
                    if c3[1] >= tile_center[1]-mask.shape[1]*0.2 and c3[0] >= tile_center[0]-mask.shape[0]*0.2:
                        candidates2 = get_90deg_corners(c3, corners, lambda c: True, lambda c: True, angle_margin)  # c[0] >= tile_center[0] and c[1] <= tile_center[1])
                        for t2, t4 in candidates2:
                            if (((np.array_equal(c2, t2) and np.array_equal(c4, t4)) or
                                 (np.array_equal(c2, t4) and np.array_equal(c4, t2)))) and 90 - angle_margin < get_angle(c2,c3,c4) < 90 + angle_margin and 90 - angle_margin < get_angle(c3, c4, c1) < 90 + angle_margin:
                                new = [c1, c2, c3, c4]
                                ang_new = np.array([get_angle(new[i], new[(i + 1) % 4], new[(i + 2) % 4]) for i in range(4)])
                                ang_diff_ls_new = np.sum(np.square(ang_opt - ang_new))
                                side_var_new = np.var([get_distance(new[(i + 1) % 4], new[i]) for i in range(4)])
                                if len(tile_corners) > 0:
                                    dist_var_new = np.var([get_distance(x, tile_center) for x in new])
                                    dist_var_curr = np.var([get_distance(tile_corners[i], tile_center) for i in range(4)])
                                    if ang_diff_ls_new > ang_diff_ls or dist_var_new > dist_var_curr or side_var_new > side_var:
                                        continue
                                elif ang_diff_ls_new > angle_diff_ls_thres:
                                    continue
                                ang_diff_ls = ang_diff_ls_new
                                side_var = side_var_new
                                ret.append(np.array([c1, c2, c3, c4]))
    tile_corners = sorted(ret, key=lambda x: -cv2.contourArea(x))[0]
    return tile_corners