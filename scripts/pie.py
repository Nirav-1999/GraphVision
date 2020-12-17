import numpy as np
import cv2
import imutils
import os

img_dir = "/home/nirav/Desktop/test_images_pie/"
for f in os.listdir(img_dir):
    file_path = os.path.join(img_dir,f)
    image = cv2.imread(file_path)

    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(gray_img, 5)
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    """ operatedImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # modify the data type
    # setting to 32-bit floating point
    operatedImage = np.float32(operatedImage)

    # apply the cv2.cornerHarris method
    # to detect the corners with appropriate
    # values as input parameters
    dest = cv2.cornerHarris(operatedImage, 2, 5, 0.07)

    # Results are marked through the dilated corners
    dest = cv2.dilate(dest, None)

    # Reverting back to the original image,
    # with optimal threshold value
    image[dest > 0.01 * dest.max()] = [0, 0, 255] """

    # the window showing output image with corners

    # center

    circles = cv2.HoughCircles(
        img, cv2.HOUGH_GRADIENT, 1, 120, param1=100, param2=70, minRadius=5, maxRadius=0)
    if circles is not None:
        circles = np.uint16(np.around(circles))

        for i in circles[0, :]:
                #	draw	the	outer	circle
            cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 6)
            #	draw	the	center	of	the	circle
            cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)

        cv2.imshow("HoughCirlces",	image)
        cv2.imwrite(file_path.split(".")[0]+"_out."+file_path.split(".")[-1],image)

        cv2.waitKey(3000)
