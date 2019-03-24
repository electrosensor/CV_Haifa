# Interface for interactive selection of segment points

# Interface instruction:
# Image opens to the user once the program starts to run.
# The user then selects segments with mouse:
# •	Right click: segments a point
# •	Left click: start a line
#   o	Line is from the previous point selected to the one clicked.
#   o	All points in the line belong to the current segment
# User uses ‘space bar’ to switch between segments,
# the message board shows on which segment the user is on at a given time.
# There are 4 segments in total, each has a color: RED, GREEN, BLUE, YELLOW respectively.
# Once you finish segmenting, press ‘Esc’.

# Once manual segmentation is finished:
# The user will have four lists: seg0, seg1, seg2, seg3. Each is a list with all the points belonging to the segment.


import cv2
import dlib
import sys
import numpy as np
from matplotlib import pyplot as plt

inputImage = 'images/man.jpg'   # use JPG images
segmentedImage = 'man_seg.jpg'
segmaskImage = 'man_seg_mask.jpg'

SEGMENT_ZERO = 0
SEGMENT_ONE = 1
SEGMENT_TWO = 2
SEGMENT_THREE = 3


# mouse callback function
def mouse_click(event, x, y, flags,params):
    # if left button is pressed, draw line
    if event == cv2.EVENT_LBUTTONDOWN:
        if current_segment == SEGMENT_ZERO:
            if len(seg0) == 0:
                seg0.append((x, y))
            else:
                points = add_line_point(seg0[-1], (x, y))
                append_to_seg(seg0, points)
        if current_segment == SEGMENT_ONE:
            if len(seg1) == 0:
                seg1.append((x, y))
            else:
                points = add_line_point(seg1[-1], (x, y))
                append_to_seg(seg1, points)
        if current_segment == SEGMENT_TWO:
            if len(seg2) == 0:
                seg2.append((x, y))
            else:
                points = add_line_point(seg2[-1], (x, y))
                append_to_seg(seg2, points)
        if current_segment == SEGMENT_THREE:
            if len(seg3) == 0:
                seg3.append((x,y))
            else:
                points = add_line_point(seg3[-1], (x, y))
                append_to_seg(seg3, points)

    # right mouse click adds single point
    if event == cv2.EVENT_RBUTTONDOWN:
        if current_segment == SEGMENT_ZERO:
            seg0.append((x, y))
        if current_segment == SEGMENT_ONE:
            seg1.append((x, y))
        if current_segment == SEGMENT_TWO:
            seg2.append((x, y))
        if current_segment == SEGMENT_THREE:
            seg3.append((x, y))

    # show on seg_img with colors
    paint_segment(seg0, (0, 0, 255))
    paint_segment(seg1, (0, 255, 0))
    paint_segment(seg2, (255, 0, 0))
    paint_segment(seg3, (0, 255, 255))


# adding line points to segment
def append_to_seg(seg,points):
    for p in points:
        seg.append(p)


# given two points, this function returns all the points on line between.
# this is used when user selects lines on segments
def add_line_point(p1,p2):
    x1, y1 = p1
    x2, y2 = p2

    points = []
    issteep = abs(y2 - y1) > abs(x2 - x1)
    if issteep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    rev = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        rev = True
    deltax = x2 - x1
    deltay = abs(y2 - y1)
    error = int(deltax / 2)
    y = y1
    ystep = None
    if y1 < y2:
        ystep = 1
    else:
        ystep = -1
    for x in range(x1, x2 + 1):
        if issteep:
            points.append((y, x))
        else:
            points.append((x, y))
        error -= deltay
        if error < 0:
            y += ystep
            error += deltax
    # Reverse the list if the coordinates were reversed
    if rev:
        points.reverse()
    return points


# given a segment points and a color, paint in seg_image
def paint_segment(segment, color):
    for center in segment:
        cv2.circle(seg_img, center, 2, color, -1)


def main():
    global orig_img, seg_img, current_segment
    global seg0, seg1, seg2, seg3

    orig_img = cv2.imread(inputImage)
    seg_img = cv2.imread(inputImage)

    cv2.namedWindow("Select segments")

    # mouse event listener
    cv2.setMouseCallback("Select segments", mouse_click)

    # lists to hold pixels in each segment
    seg0 = []
    seg1 = []
    seg2 = []
    seg3 = []
    # segment you're on
    current_segment = 0

    while True:
        cv2.imshow("Select segments", seg_img)
        k = cv2.waitKey(20)

        if k == 32:  # space bar to switch between segments
            current_segment = (current_segment + 1) % 4
            print('current segment = ' + str(current_segment))
        if k == 27:  # escape
            break

    # graph cut implementation for 4 segments
    # add functions and code as you wish

    mask = np.ones(seg_img.shape[:2], np.uint8)*2
    for point in seg0:
        mask[point[1], point[0]] = 1
    for point in seg1:
        mask[point[1], point[0]] = 1
    for point in seg2:
        mask[point[1], point[0]] = 0
    for point in seg3:
        mask[point[1], point[0]] = 0

    bg_model1 = np.zeros((1, 65), np.float64)
    fg_model1 = np.zeros((1, 65), np.float64)
    mask, bg_model1, fg_model1 = cv2.grabCut(orig_img, mask, None, bg_model1, fg_model1, 5, cv2.GC_INIT_WITH_MASK)

    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    orig_img1 = orig_img * mask[:, :, np.newaxis]

    mask2 = mask[:, :]
    for point in seg0:
        mask2[point[1], point[0]] = 1
    for point in seg1:
        mask2[point[1], point[0]] = 0
    for point in seg2:
        mask2[point[1], point[0]] = 0
    for point in seg3:
        mask2[point[1], point[0]] = 0

    mask3 = mask[:, :]
    for point in seg0:
        mask3[point[1], point[0]] = 0
    for point in seg1:
        mask3[point[1], point[0]] = 1
    for point in seg2:
        mask3[point[1], point[0]] = 0
    for point in seg3:
        mask3[point[1], point[0]] = 0

    bg_model1 = np.zeros((1, 65), np.float64)
    fg_model1 = np.zeros((1, 65), np.float64)
    mask2, bg_model1, fg_model1 = cv2.grabCut(orig_img1, mask2, None, bg_model1, fg_model1, 5, cv2.GC_INIT_WITH_MASK)

    bg_model2 = np.zeros((1, 65), np.float64)
    fg_model2 = np.zeros((1, 65), np.float64)
    mask3, bg_model2, fg_model2 = cv2.grabCut(orig_img1, mask3, None, bg_model2, fg_model2, 5, cv2.GC_INIT_WITH_MASK)

    seg_img2 = orig_img * mask2[:, :, np.newaxis]
    seg_img3 = orig_img * mask3[:, :, np.newaxis]

    cv2.namedWindow("seg1")
    cv2.namedWindow("seg1a")
    cv2.namedWindow("seg1b")

    # plt.imshow(seg_img), plt.colorbar(), plt.show()
    while True:
        cv2.imshow("seg1", orig_img1)
        cv2.imshow("seg1a", seg_img2)
        cv2.imshow("seg1b", seg_img3)
        k = cv2.waitKey(20)
        if k == 27:  # escape
            break
    # destroy all windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

