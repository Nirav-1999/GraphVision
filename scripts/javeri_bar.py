import numpy as np
import cv2
import imutils
import os

directory = '/home/nirav/Desktop/test_images_bar'

for filename in os.listdir(directory):
    file_path = os.path.join(directory,filename)
    image = cv2.imread(file_path)
    
    resized = imutils.resize(image, width=300)
    # Display original image
    cv2.imshow("Initial", resized)
    cv2.waitKey(2000)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # gray it
    binary = cv2.bitwise_not(gray)  # invert
    blurred = cv2.GaussianBlur(binary, (11, 11), cv2.BORDER_DEFAULT)  # blur it
    '''#Display blurred image
    cv2.imshow("BitwiseNot and blurred", blurred)
    cv2.waitKey(3000)'''

    thresh = cv2.threshold(blurred, 130, 255, cv2.THRESH_BINARY)[1]  # thresholding
    '''#Display thresholded image
    cv2.imshow("Thresh", thresh)
    cv2.waitKey(2000)'''

    cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

        (x, y, w, h) = cv2.boundingRect(approx)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display thresholded image
    cv2.imshow("Final", image)
    cv2.imwrite(file_path.split(".")[0]+"_out."+file_path.split(".")[-1],image)

    cv2.waitKey(3000)