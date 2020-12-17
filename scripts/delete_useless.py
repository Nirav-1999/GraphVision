import os

in_dir = "/home/nirav/Desktop/Workspace/ML/DJ_Strike/data/test_set/"

count = 0
for directory in os.listdir(in_dir):
    print(directory)
    for img in os.listdir(os.path.join(in_dir,directory)):
        if img.endswith(".gif"):
            os.remove(os.path.join(in_dir,directory,img) )
            count+=1
            print(count)