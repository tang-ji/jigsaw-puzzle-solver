from scipy.interpolate import splprep, splev
import cv2
import numpy as np
import scipy.ndimage as ndimage

def crop_piece(piece, img_balanced, margin=10):
    piece = piece.copy().astype(np.uint8)
    piece[piece==True]=255
    piece[piece==False]=0
    contour_info = [(c, cv2.contourArea(c),) for c in cv2.findContours(piece, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]]
    image_area = piece.shape[0] * piece.shape[1]  
    min_area = 0.0001 * image_area
    points = np.array([])
    for contour in contour_info:
        if contour[1] > min_area:
            points = contour[0].reshape(contour[0].shape[0], contour[0].shape[2])
    minX = max(min(points[:, 1])-margin, 0)
    maxX = min(max(points[:, 1])+margin, len(piece))
    minY= max(min(points[:, 0])-margin, 0)
    maxY= min(max(points[:, 0])+margin, len(piece[0]))
#     print(minY,maxY, minX,maxX)
    return piece[minX:maxX, minY:maxY], img_balanced[minX:maxX, minY:maxY]

def refine_piece(piece, image_area=None, n_polys=150):
    piece = piece.copy()
    piece = cv2.dilate(piece, None, iterations=3)
    piece = cv2.erode(piece, None, iterations=3)
    mask = np.zeros(piece.shape, dtype = np.uint8)
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
            tck, u = splprep([x,y], u=None, s=1.0, per=1)
            u_new = np.linspace(u.min(), u.max(), n_polys)
            x_new, y_new = splev(u_new, tck, der=0)
            res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new,y_new)]
            smoothened.append(np.asarray(res_array, dtype=np.int32))
            mask = cv2.fillPoly(mask, smoothened, (255))
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
        f = False
        for c2 in corners_refined:
            if get_angle(c2, tile_center, c1) < angle_diff:
                if np.sum(np.square(np.array(c1) - np.array(tile_center))) > np.sum(np.square(np.array(c2) - np.array(tile_center))):
                    corners_refined.remove(tuple(c2))
                    corners_refined.append(tuple(c1))
                f = True
                continue
        if not f:
            corners_refined.append(tuple(c1))
    corners_refined = np.array(corners_refined)
    return corners_refined

def get_tile_corners(mask, angle_margin=15, angle_diff_ls_thres=200):
    corners = cv2.goodFeaturesToTrack(mask, 80, 0.1, 20)
    corners = corners.reshape(corners.shape[0], corners.shape[-1])
#     tile_center = ndimage.center_of_mass(mask)
#     tile_center = tuple(np.round(tile_center).astype(np.int))
    tile_center = np.mean(corners, axis=0)
    ang_opt = np.array([90, 90, 90, 90])
    ang_diff_ls = None
    side_var = None
    tile_corners = []
    ret = []

    corners = refine_corners(corners, tile_center)

    for c1 in corners:
        if c1[1] <= tile_center[1]:
                candidates1 = get_90deg_corners(c1, corners, lambda c: True, lambda c: True, angle_margin)
                for c2, c4 in candidates1:
                    for c3 in corners:
                        if c3[1] >= tile_center[1]:
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