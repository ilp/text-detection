# FEDERAL RURAL UNIVERSITY OF PERNAMBUCO
# DISCIPLINE OF IMAGE PROCESSING
# PROFESSOR VALMIR MACARIO
# STUDENT IVERSON LUIS PEREIRA
# COURSE OF COMPUTER SCIENCE
import math
import numpy as np
import cv2
import os
import sys
from time import time
from scipy.spatial import KDTree
from matplotlib import pyplot as plt

#paramenters that can to influence the result
CANNY_THRESHOLD_MIN = 250
CANNY_THRESHOLD_MAX = 400
WITH_HIST_EQU = False
WITH_MORPH_DIL = True
MORPH_WINDOW = (7, 7)
INTERATIONS_MORPH = 1
MAX_RAY_LEN = 100
MAX_ANGL_DIFF = math.pi/2

def pre_processing(original_img):
    """
    improve the image to a best edge detection
    """   
    img_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    if WITH_HIST_EQU:
        return cv2.equalizeHist(img_gray)
    else:
        return img_gray     

def edge_detection(img_preprocessed):
    """
    detects edges of an image
    """
    img_edges = cv2.Canny(img_preprocessed, CANNY_THRESHOLD_MIN, CANNY_THRESHOLD_MAX)   
    if WITH_MORPH_DIL:
        kernel = np.ones(MORPH_WINDOW, np.uint8)
        dilated = cv2.dilate(img_edges, kernel, INTERATIONS_MORPH)
        return dilated
    else:
        return img_edges

def gradient_detection(img_preprocessed):
    """
    detects the horizontal and vertical gradients
    """
    sobelx64f = cv2.Sobel(img_preprocessed, cv2.CV_64F, 1, 0, ksize=5)
    sobely64f = cv2.Sobel(img_preprocessed, cv2.CV_64F, 0, 1, ksize=5)
    theta = np.arctan2(sobely64f, sobelx64f)

    return sobelx64f, sobely64f, theta

def stroke_width_transform(edges, sobelx64f, sobely64f, theta, verbose=0):
    """
    find the stroke of edges
    """
    swt = np.empty(theta.shape)
    swt[:] = np.Infinity
    rays = []

    step_x = -1 * sobelx64f
    step_y = -1 * sobely64f
    mag = np.sqrt(step_x * step_x + step_y * step_y)

    with np.errstate(divide='ignore', invalid='ignore'):
        d_all_x = step_x / mag
        d_all_y = step_y / mag

    for p_x in range(edges.shape[1]):
        for p_y in range(edges.shape[0]):

            if edges[p_y, p_x] > 0:
                d_p_x = d_all_x[p_y, p_x]
                d_p_y = d_all_y[p_y, p_x]
                if math.isnan(d_p_x) or math.isnan(d_p_y):
                    continue
                ray = [(p_x, p_y)]
                prev_x, prev_y, i = p_x, p_y, 0

                while True:
                    i += 1
                    q_x = math.floor(p_x + d_p_x * i)
                    q_y = math.floor(p_y + d_p_y * i)
                    if q_x != prev_x or q_y != prev_y:
                        try:                            
                            if edges[q_y, q_x] > 0:
                                ray.append((q_x, q_y))                               
                                if len(ray) > MAX_RAY_LEN:
                                    break                                
                                delta = max(min(d_p_x * -d_all_x[q_y, q_x] + d_p_y * -d_all_y[q_y, q_x], 1.0), -1.0)
                                if not math.isnan(delta) and math.acos(max([-1.0, min([1.0, delta])])) < MAX_ANGL_DIFF:
                                    ray_len = math.sqrt((q_x - p_x) ** 2 + (q_y - p_y) ** 2)
                                    for (rp_x, rp_y) in ray:
                                        swt[rp_y, rp_x] = min(ray_len, swt[rp_y, rp_x])
                                    rays.append(np.asarray(ray))
                                break                           
                            ray.append((q_x, q_y))                      
                        except IndexError:
                            break
                        prev_x = q_x
                        prev_y = q_y

    for ray in rays:
        median = np.median(swt[ray[:, 1], ray[:, 0]])
        for (p_x, p_y) in ray:
            swt[p_y, p_x] = min(median, swt[p_y, p_x])

    if verbose > 0:
        cv2.imwrite('output/swt.jpg', swt * 100)

    return swt

def connected_components(swt):
    """
    detects the connected components from swt result image, verifying each stroke width
    """
    # Implementation of disjoint-set
    class Label(object):
        def __init__(self, value):
            self.value = value
            self.parent = self
            self.rank = 0

        def __eq__(self, other):
            if type(other) is type(self):
                return self.value == other.value
            else:
                return False

        def __ne__(self, other):
            return not self.__eq__(other)

    def make_set(x):
        try:
            return ld[x]
        except KeyError:
            ld[x] = Label(x)
            return ld[x]

    def find(item):
        if item.parent != item:
            item.parent = find(item.parent)
        return item.parent

    def union(x, y):
        """
        :param x:
        :param y:
        :return: root node of new union tree
        """
        x_root = find(x)
        y_root = find(y)
        if x_root == y_root:
            return x_root

        if x_root.rank < y_root.rank:
            x_root.parent = y_root
            return y_root
        elif x_root.rank > y_root.rank:
            y_root.parent = x_root
            return x_root
        else:
            y_root.parent = x_root
            x_root.rank += 1
            return x_root

    ld = {}

    # apply Connected Component algorithm, comparing SWT values.
    # components with a SWT ratio less extreme than 1:3 are assumed to be
    # connected. Apply twice, once for each ray direction/orientation, to
    # allow for dark-on-light and light-on-dark texts
    trees = {}
    # Assumption: we'll never have more than 65535-1 unique components
    label_map = np.zeros(shape=swt.shape, dtype=np.uint16)
    next_label = 1
    # First Pass, raster scan-style
    swt_ratio_thresh = 3.0
    for y in range(swt.shape[0]):
        for x in range(swt.shape[1]):
            sw_point = swt[y, x]
            if 0 < sw_point < np.Infinity:
                neighbors = [(y, x-1),    # west
                             (y-1, x-1),  # northwest
                             (y-1, x),    # north
                             (y-1, x+1)]  # northeast
                connected_neighbors = None
                neighborvals = []

                for neighbor in neighbors:
                    try:
                        sw_n = swt[neighbor]
                        label_n = label_map[neighbor]
                    # out of image boundary
                    except IndexError:
                        continue
                    # labeled neighbor pixel within SWT ratio threshold
                    if label_n > 0 and sw_n / sw_point < swt_ratio_thresh and sw_point / sw_n < swt_ratio_thresh:
                        neighborvals.append(label_n)
                        if connected_neighbors:
                            connected_neighbors = union(connected_neighbors, make_set(label_n))
                        else:
                            connected_neighbors = make_set(label_n)

                if not connected_neighbors:
                    # We don't see any connections to North/West
                    trees[next_label] = (make_set(next_label))
                    label_map[y, x] = next_label
                    next_label += 1
                else:
                    # We have at least one connection to North/West
                    label_map[y, x] = min(neighborvals)
                    # For each neighbor, make note that their respective connected_neighbors are connected
                    # for label in connected_neighbors.
                    # @TODO: do I need to loop at all neighbor trees?
                    trees[connected_neighbors.value] = union(trees[connected_neighbors.value], connected_neighbors)

    # Second pass. re-base all labeling with representative label for each connected tree
    layers = {}
    for x in range(swt.shape[1]):
        for y in range(swt.shape[0]):
            if label_map[y, x] > 0:
                item = ld[label_map[y, x]]
                common_label = find(item).value
                label_map[y, x] = common_label
                try:
                    layer = layers[common_label]
                except KeyError:
                    layers[common_label] = {'x': [], 'y': []}
                    layer = layers[common_label]

                layer['x'].append(x)
                layer['y'].append(y)
    return layers

def get_letters(swt, comp):
    img_w = swt.shape[0]
    img_h = swt.shape[1]
    letters = []

    for _, c in comp.items():
        east, west, south, north = max(c['x']), min(c['x']), max(c['y']), min(c['y'])
        width, height = east - west, south - north

        if width < 8 and height < 8:
            continue
        if width < 4 or height < 4:
            continue
        if width / height > 10 or height / width > 10:
            continue

        diameter = math.sqrt(width**2 + height**2)
        median_swt = np.median(swt[(c['y'], c['x'])])
        if diameter / median_swt > 15:  # TODO: this threshold can be improved?
            continue

        if width / img_w > 0.4 or height / img_h > 0.4:
            continue

        letter = [
            median_swt,
            height,
            width,
            north,
            west
        ]
        letters.append(letter)    

    return np.asarray(letters)

def get_words(letters): # swts, heights, widths, topleft_pts, images):
    # Index-pairs of letter with similar median stroke widths and similar heights
    # We use log2 for linear distance comparison in KDTree
    # (i.e. if log2(x) - log2(y) > 1, we know that x > 2*y)
    s_ix_letr_pairs = KDTree(np.log2(letters[:, 0:1])).query_pairs(1)
    h_ix_letr_pairs = KDTree(np.log2(letters[:, 1:2])).query_pairs(1)

    # Calc the angle (direction of text) for all letter-pairs which are
    # similar and close to each other
    pairs = []
    for ix_letr1, ix_letr2 in h_ix_letr_pairs.intersection(s_ix_letr_pairs):
        diff = letters[ix_letr1, 3:5] - letters[ix_letr2, 3:5]
        # Distance between letters smaller than
        # 3 times the width of the wider letter
        dist = np.linalg.norm(diff)
        if dist < max(letters[ix_letr1, 2], letters[ix_letr2, 2]) * 3:
            angle = math.atan2(diff[0], diff[1])
            angle += math.pi if angle < 0 else 0
            pairs.append([ix_letr1, ix_letr2, angle])
    pairs = np.asarray(pairs)

    # Pairs of letter-pairs with a similar angle (direction of text)
    a_ix_pair_pairs = KDTree(pairs[:, 2:3]).query_pairs(math.pi / 12)

    chains = []
    for ix_pair_a, ix_pair_b in a_ix_pair_pairs:
        # Letter pairs [a] & [b] have a similar angle and each pair consists of
        # letter [1] & [2] which meet the similarity-requirements.
        pair_a_letr1, pair_a_letr2 = int(pairs[ix_pair_a, 0]), int(pairs[ix_pair_a, 1])
        pair_b_letr1, pair_b_letr2 = int(pairs[ix_pair_b, 0]), int(pairs[ix_pair_b, 1])

        # TODO: not correct?
        added = False
        for c in chains:
            if pair_a_letr1 in c:
                c.add(pair_a_letr2)
                added = True
            elif pair_a_letr2 in c:
                c.add(pair_a_letr1)
                added = True
        if not added:
            chains.append({pair_a_letr1, pair_a_letr2})
        added = False
        for c in chains:
            if pair_b_letr1 in c:
                c.add(pair_b_letr2)
                added = True
            elif pair_b_letr2 in c:
                c.add(pair_b_letr1)
                added = True
        if not added:
            chains.append({pair_b_letr1, pair_b_letr2})
    chains = np.asarray(chains)

    # List of sets of letters with possibly many duplicates
    # return chains
    # Single list of unique letters
    # return np.unique([int(ix) for chain in chains if len(chain) >= 3 for ix in chain])

    vecfunc = np.vectorize(len)
    chains = chains[vecfunc(chains) > 3]
    _, uniq_ix = np.unique(chains.astype(str), return_index=True)
    return chains[uniq_ix]

def draw_mask(original_img, letters, words):
    words_positions = []
    pt1 = letters[:, 3:5].astype(int)

    # south-east point [pt2] = (north + height, west + width)
    pt2_y = letters[:, 3] + letters[:, 1]
    pt2_x = letters[:, 4] + letters[:, 2]
    pt2 = np.vstack([pt2_y, pt2_x]).T.astype(int)
    
    if words is None:
        words_positions = np.column_stack([pt1, pt2])
    else:
        words_positions = np.array([[pt1[list(w)][:, 0].min(), pt1[list(w)][:, 1].min(), pt2[list(w)][:, 0].max(), pt2[list(w)][:, 1].max()] for w in words])

    final_img = original_img
    for _, [pt1y, pt1x, pt2y, pt2x] in enumerate(words_positions):
        cv2.rectangle(final_img, (pt1x, pt1y), (pt2x, pt2y), (0, 255, 0), 4)

    return final_img

def main():
    """
    run the program to detect text in images
    """
    filepath = sys.argv[1]
    original_img = cv2.imread(filepath)
    img_preprocessed = pre_processing(original_img)
    edges = edge_detection(img_preprocessed)
    sobelx, sobely, theta = gradient_detection(img_preprocessed)
    swt_result = stroke_width_transform(edges, sobelx, sobely, theta)
    components = connected_components(swt_result)
    letters = get_letters(swt_result, components)
    words = get_words(letters)
    result_img = draw_mask(original_img, letters, words)
    cv2.imwrite("output/result.jpg", result_img)

#start the program
main()
