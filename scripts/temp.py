import pytesseract
import cv2
import os

img_path = "/home/nirav/Desktop/Workspace/ML/DJ_Strike/test_images_bar/30_out.png"
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imshow("image",img_rgb)
img_data = pytesseract.image_to_data(img_rgb,output_type='data.frame')
img_data = img_data[img_data['text'] != 'Nan']
print(img_data)
key = cv2.waitKey()
if key==27:
    cv2.destroyAllWindows()