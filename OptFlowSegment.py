import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


inputVideoName = 'Videos\\driving.mp4'
FrameNumber1 = 10
FrameNumber2 = 11

cap = cv.VideoCapture(inputVideoName)

for i in range(FrameNumber1):
    cap.read()

ret, frame1 = cap.read()
prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)

for i in range(FrameNumber2):
    cap.read()

ret, frame2 = cap.read()
next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
flow_samples = np.concatenate([np.reshape(mag, (-1, 1)), np.reshape(ang, (-1, 1))], axis=1)
# Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# Set flags (Just to avoid line break in the code)
flags = cv.KMEANS_RANDOM_CENTERS

# Apply KMeans
compactness, label, center = cv.kmeans(flow_samples, 4, None, criteria, 10, flags)

labeled_mask = np.reshape(label.ravel(), (frame1.shape[0], frame1.shape[1]))

# mask = mask.astype(np.uint8)

# graph cut implementation for 4 segments
# add functions and code as you wish

# Stage 1: Extraction of first segments pair

mask = np.ones(labeled_mask.shape[:2], np.uint8) * 2

mask[labeled_mask == 0] = 1
mask[labeled_mask == 1] = 1
mask[labeled_mask == 2] = 0
mask[labeled_mask == 3] = 0

bg_model1 = np.zeros((1, 65), np.float64)
fg_model1 = np.zeros((1, 65), np.float64)
mask1, bg_model1, fg_model1 = cv.grabCut(frame1, mask, None, bg_model1, fg_model1, 5, cv.GC_INIT_WITH_MASK)

mask1 = np.where((mask1 == 2) | (mask1 == 0), 0, 1).astype('uint8')
orig_img1 = frame1 * mask1[:, :, np.newaxis]

# Stage 1a: Extraction of first segment from first segments pair

mask1a = np.where(mask1 == 1, 2, 0).astype('uint8')
mask1a[labeled_mask == 0] = 1
mask1a[labeled_mask == 1] = 0

bg_model1a = np.zeros((1, 65), np.float64)
fg_model1a = np.zeros((1, 65), np.float64)
mask1a, bg_model1a, fg_model1a = cv.grabCut(orig_img1, mask1a, None, bg_model1a, fg_model1a, 5,
                                             cv.GC_INIT_WITH_MASK)

mask1a = np.where((mask1a == 2) | (mask1a == 0), 0, 1).astype('uint8')

# Stage 1b: Extraction of second segment from first segments pair

mask1b = mask1 - mask1a

bg_model1b = np.zeros((1, 65), np.float64)
fg_model1b = np.zeros((1, 65), np.float64)
mask1b, bg_model1b, fg_model1b = cv.grabCut(orig_img1, mask1b, None, bg_model1b, fg_model1b, 5,
                                             cv.GC_INIT_WITH_MASK)

mask1b = np.where((mask1b == 2) | (mask1b == 0), 0, 1).astype('uint8')

# Stage 2: Extraction of second segments

mask2 = np.where(mask1 == 0, 1, 0).astype('uint8')

bg_model2 = np.zeros((1, 65), np.float64)
fg_model2 = np.zeros((1, 65), np.float64)
mask2, bg_model2, fg_model2 = cv.grabCut(frame1, mask2, None, bg_model2, fg_model2, 5, cv.GC_INIT_WITH_MASK)

mask2 = np.where((mask2 == 2) | (mask2 == 0), 0, 1).astype('uint8')
orig_img2 = frame1 * mask2[:, :, np.newaxis]

# Stage 2a: Extraction of first segment from second segments pair

mask2a = np.where(mask2 == 1, 2, 0).astype('uint8')
mask2a[labeled_mask == 2] = 1
mask2a[labeled_mask == 3] = 0

bg_model2a = np.zeros((1, 65), np.float64)
fg_model2a = np.zeros((1, 65), np.float64)
mask2a, bg_model2a, fg_model2a = cv.grabCut(orig_img2, mask2a, None, bg_model2a, fg_model2a, 5,
                                             cv.GC_INIT_WITH_MASK)

mask2a = np.where((mask2a == 2) | (mask2a == 0), 0, 1).astype('uint8')

# Stage 2b: Extraction of second segment from second segments pair

mask2b = mask2 - mask2a

bg_model2b = np.zeros((1, 65), np.float64)
fg_model2b = np.zeros((1, 65), np.float64)
mask2b, bg_model2b, fg_model2b = cv.grabCut(orig_img2, mask2b, None, bg_model2b, fg_model2b, 5,
                                             cv.GC_INIT_WITH_MASK)

mask2b = np.where((mask2b == 2) | (mask2b == 0), 0, 1).astype('uint8')

# Stage 3: Coloring and composition of four binary masks (1a, 1b, 2a, 2b)

mask_tmpR = mask1a[:, :, np.newaxis] * 0
mask_tmpG = mask1a[:, :, np.newaxis] * 0
mask_tmpB = mask1a[:, :, np.newaxis] * 255
out_mask1a = np.concatenate([mask_tmpR, mask_tmpG, mask_tmpB], axis=2)

mask_tmpR = mask1b[:, :, np.newaxis] * 0
mask_tmpG = mask1b[:, :, np.newaxis] * 255
mask_tmpB = mask1b[:, :, np.newaxis] * 0
out_mask1b = np.concatenate([mask_tmpR, mask_tmpG, mask_tmpB], axis=2)

mask_tmpR = mask2a[:, :, np.newaxis] * 255
mask_tmpG = mask2a[:, :, np.newaxis] * 0
mask_tmpB = mask2a[:, :, np.newaxis] * 0
out_mask2a = np.concatenate([mask_tmpR, mask_tmpG, mask_tmpB], axis=2)

mask_tmpR = mask2b[:, :, np.newaxis] * 0
mask_tmpG = mask2b[:, :, np.newaxis] * 255
mask_tmpB = mask2b[:, :, np.newaxis] * 255
out_mask2b = np.concatenate([mask_tmpR, mask_tmpG, mask_tmpB], axis=2)

output_mask = out_mask1a + out_mask1b + out_mask2a + out_mask2b

cv.namedWindow('frame')
while True:
    cv.imshow('frame', output_mask)

    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    # elif k == ord('s'):
    #     cv.imwrite('opticalfb.png', frame)


cap.release()
cv.destroyAllWindows()

