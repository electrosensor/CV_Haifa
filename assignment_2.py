import cv2
import dlib
import sys
import numpy as np


def main():

    while True:
        k = cv2.waitKey(20)
        if k == 27:  # escape
            break
    # destroy all windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

