import numpy
import cv2


def main():
    print("numpy version: " + numpy.__version__)
	print("open cv version: " + cv2.__version__)



if __name__ == '__main__':
    main()

	

def interactive_segmentation(nputImage,segmentedImage,segmaskImage):
#Input
#nputImage = '<name of input image file>'   #  use JPG images
#segmentedImage = <name of output image file – image with transparent segment overlay>
#segmaskImage = '<name of output image file – image with segmentation mask >
  print("Hello from a function")