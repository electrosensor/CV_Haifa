import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

inputVideoName = 'Videos\\bugs11.mp4'

def draw_flow(img, flow, step=8):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cv.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


cap = cv.VideoCapture(inputVideoName)
ret, frame1 = cap.read()
prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
# hsv = np.zeros_like(frame1)
# hsv[..., 1] = 255

for i in range(100):
    cap.read()

while ret:
    ret, frame2 = cap.read()
    next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow_samples = np.reshape(flow, (-1, 2))
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    flow_samples = np.concatenate([np.reshape(mag, (-1, 1)), np.reshape(ang, (-1, 1))], axis=1)
    # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # Set flags (Just to avoid line break in the code)
    flags = cv.KMEANS_RANDOM_CENTERS

    # Apply KMeans
    compactness, label, center = cv.kmeans(flow_samples, 2, None, criteria, 10, flags)

    # Now separate the data, Note the flatten()
    # A = flow_samples[label.ravel() == 0]
    # B = flow_samples[label.ravel() == 1]

    mask = np.reshape(label.ravel(), (frame2.shape[0], frame2.shape[1]))

    # # Plot the data
    # plt.scatter(A[:, 0], A[:, 1])
    # plt.scatter(B[:, 0], B[:, 1], c='r')
    # plt.scatter(center[:, 0], center[:, 1], s=80, c='y', marker='s')
    # plt.xlabel('Height'), plt.ylabel('Weight')
    # plt.show()

    mask = mask.astype(np.uint8)

    mask[mask == 0] = 2
    mask[mask == 1] = 3

    # Stage 1: Extraction of first segments pair
    bg_model1 = np.zeros((1, 65), np.float64)
    fg_model1 = np.zeros((1, 65), np.float64)
    mask, bg_model1, fg_model1 = cv.grabCut(frame2, mask, None, bg_model1, fg_model1, 1, cv.GC_INIT_WITH_MASK)

    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    # orig_img1 = frame2 * mask[:, :, np.newaxis]

    mask_tmpR = mask[:, :, np.newaxis] * 200
    mask_tmpG = mask[:, :, np.newaxis] * 200
    mask_tmpB = mask[:, :, np.newaxis] * 200
    out_mask = np.concatenate([mask_tmpR, mask_tmpG, mask_tmpB], axis=2)

    cv.imshow('frame2', out_mask)

    # hsv[...,0] = ang*180/np.pi/2
    # hsv[...,2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    # bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
    # cv.imshow('frame2', draw_flow(next, flow))
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv.imwrite('opticalfb.png', frame2)
        cv.imwrite('opticalhsv.png', bgr)
    prvs = next
cap.release()
cv.destroyAllWindows()

