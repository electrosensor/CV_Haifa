import cv2 as cv
import numpy as np

inputVideoName = 'Videos\\ballet.mp4'
selectPoints = False
numberOfPoints = 10


# mouse callback function
def mouse_click(event, x, y, flags, params):
    # if left button is pressed, draw line
    global current_pt
    if event == cv.EVENT_LBUTTONDOWN or event == cv.EVENT_RBUTTONDOWN:
        prev_pts[current_pt, 0] = (x, y)
        current_pt = (current_pt + 1) % numberOfPoints
    paint_pts((0, 255, 255))


# given a frame points and a color, paint in frame
def paint_pts(color):
    for i in range(numberOfPoints):
        if prev_pts[i, 0, 0] != -1 and prev_pts[i, 0, 1] != -1:
          cv.circle(prev_frame, (prev_pts[i, 0, 0], prev_pts[i, 0, 1]), 10, color, -1)


def main():
    global current_pt
    global prev_pts
    global prev_frame
    # Create a VideoCapture object and read from input file
    # If the input is taken from the camera, pass 0 instead of the video file name.
    cap = cv.VideoCapture(inputVideoName)

    cv.namedWindow('video')

    if not cap.isOpened():
        print("cant read video")
    else:
        ret, prev_frame = cap.read()
        if ret:
            prev_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)

            if selectPoints:
                # mouse event listener
                current_pt = 0
                cv.setMouseCallback("prev_pts", mouse_click)
                prev_pts = -1*np.ones([numberOfPoints, 1, 2], dtype=np.float32)
                # lists to hold pixels in each segment
                while True:
                    cv.imshow("prev_pts", prev_frame)
                    k = cv.waitKey(20)
                    print(type(prev_pts))
                    print(prev_pts.shape)
                    print(prev_pts)
                    for i in range(current_pt):
                        print(prev_pts[i])
                    if k == 27:  # escape
                        break
            else:
                feature_params = dict(maxCorners=numberOfPoints, qualityLevel=0.01, minDistance=25, blockSize=9, useHarrisDetector=False)
                prev_pts = cv.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

            # Parameters for lucas kanade optical flow
            lk_params = dict(winSize=(9, 9),
                             maxLevel=2,
                             criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

            # Create some random colors
            color = np.random.randint(0, 255, (numberOfPoints, 3))
            # Create a mask image for drawing purposes
            mask = np.zeros_like(prev_frame)

        while cap.isOpened():
            ret, next_frame = cap.read()
            if ret:
                next_gray = cv.cvtColor(next_frame, cv.COLOR_BGR2GRAY)

                next_pts, status, err = cv.calcOpticalFlowPyrLK(**lk_params, prevImg=prev_gray, nextImg=next_gray, prevPts=prev_pts, nextPts=None)

                # Select good points
                good_new = next_pts[status == 1]
                good_old = prev_pts[status == 1]

                if len(good_new) < numberOfPoints:
                    feature_params = dict(maxCorners=numberOfPoints - len(good_new), qualityLevel=0.01, minDistance=10, blockSize=3, useHarrisDetector=False)
                    new_pts = cv.goodFeaturesToTrack(next_gray, mask=None, **feature_params)
                    good_new = np.concatenate((good_new, np.reshape(new_pts, [-1, 2])), axis=0)

                # draw the tracks
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    mask = cv.line(mask, (a, b), (c, d), color[i].tolist(), 3)
                    frame = cv.circle(next_frame, (a, b), 7, color[i].tolist(), -1)
                img = cv.add(frame, mask)

                cv.imshow('video', img)

                if cv.waitKey(25) & 0xFF == ord('q'):
                    break

                prev_gray = next_gray.copy()
                prev_pts = good_new.reshape(-1, 1, 2)

            else:
                cv.imwrite('optical_flow.png', img)
                break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
