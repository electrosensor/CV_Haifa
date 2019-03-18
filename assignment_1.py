import numpy


def main():
    print("numpy version: " + numpy.__version__)


if __name__ == '__main__':
    main()


def interactive_segmentation(input_image,segmented_image, seg_mask_image):

    # Input
    # inputImage = '<name of input image file>'  (use JPG images)
    # segmentedImage = <name of output image file – image with transparent segment overlay>
    # seg_mask_image = '<name of output image file – image with segmentation mask >

    print("Hello from a function")
