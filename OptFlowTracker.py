import cv2 as cv
import numpy as np

# Create a VideoCapture object and read from input file
# If the input is taken from the camera, pass 0 instead of the video file name.
cap = cv.VideoCapture('Videos\\ballet.mp4')

cv.namedWindow('video')
cv.namedWindow('prev_pts')

if not cap.isOpened():
    print("cant read video")
else:
    ret, prev_frame = cap.read()
    if ret:
        prev_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)

        feature_params = dict(maxCorners=100, qualityLevel=0.2, minDistance=2, blockSize=7)
        prev_pts = cv.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

        # Parameters for lucas kanade optical flow
        lk_params = dict(winSize=(10, 10),
                         maxLevel=2,
                         criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

        # Create some random colors
        color = np.random.randint(0, 255, (100, 3))
        # Create a mask image for drawing purposes
        mask = np.zeros_like(prev_frame)

    while cap.isOpened():
        ret, next_frame = cap.read()
        if ret:
            next_gray = cv.cvtColor(next_frame, cv.COLOR_BGR2GRAY)
            # next_gray = np.float32(next_gray)

            next_pts, status, err = cv.calcOpticalFlowPyrLK(prevImg=prev_gray, nextImg=next_gray, prevPts=prev_pts, nextPts=None)

            # Select good points
            good_new = next_pts[status == 1]
            good_old = prev_pts[status == 1]

            # draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv.line(mask, (a, b), (c, d), color[i].tolist(), 2)
                frame = cv.circle(next_frame, (a, b), 5, color[i].tolist(), -1)
            img = cv.add(frame, mask)

            cv.imshow('video', next_frame)
            cv.imshow('prev_pts', img)

            if cv.waitKey(25) & 0xFF == ord('q'):
                break

            prev_gray = next_gray.copy()
            prev_pts = good_new.reshape(-1, 1, 2)

        else:
            print('An error occurred')
            break

cap.release()
cv.destroyAllWindows()


