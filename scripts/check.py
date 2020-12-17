import os
from PIL import Image

img_dir = r"/home/nirav/Desktop/Workspace/ML/DJ_Strike/data/pie charts/"
for filename in os.listdir(img_dir):
    try :
        with Image.open(img_dir + "/" + filename) as im:
             print('ok')
    except :
        print(img_dir + "/" + filename)
        os.remove(img_dir + "/" + filename)