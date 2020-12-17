import numpy as np
import cv2
import os

img_dir = "/home/nirav/Desktop/Workspace/ML/DJ_Strike/test_images_bar"
for f in os.listdir(img_dir):
    bars = []
    file_path = os.path.join(img_dir,f)
    img = cv2.imread(file_path)
    # cv2.imshow('image',img)
    _,xmin,_ = img.shape
    ymax = 0
    blurred = cv2.GaussianBlur(img,(5,5),0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    # edges = cv2.Canny(gray, 100, 200)
    # cv2.imshow('some',edges)
    # cv2.imshow('image',gray)

    # ret,thresh = cv2.threshold(gray,0,255,1)
    # cv2.imshow('thresh',thresh)
    edges = cv2.Canny(gray, 100, 200)
    cv2.imshow('some',edges)
    contours,h = cv2.findContours(edges,1,2)
    # cv2.drawContours(img,contours,-1,(0,255,0),3)
    # cv2.drawContours(edges,contours,-1,(255,255,255),3)
    # cv2.imshow('all_contours_edges',edges)
    out_dir = "/home/nirav/Desktop/Workspace/ML/DJ_Strike/test_images_bar"
    cv2.imshow('all_contours',img)
    i=0
    print("Lower limit -------->  ",img.shape[0]*img.shape[1]*1.5/100)
    print("Upper limit -------->  ",img.shape[0]*img.shape[1]*90/100)
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
        print(len(approx),' -> ',cv2.contourArea(approx))
        if len(approx)<9:
            if cv2.contourArea(approx)>img.shape[0]*img.shape[1]*1.5/100 and cv2.contourArea(approx)<img.shape[0]*img.shape[1]*50/100:
                print ("square")
#                 print(cnt)
                cv2.drawContours(img,[approx],0,(0,0,255),3)
                print("++++++++++++++++++++++++")
                print(approx)
                bars.append(approx)
                for ele in approx:
                    if ele[0][0] < xmin:
                        xmin = ele[0][0]
                    if ele[0][1] > ymax:
                        ymax = ele[0][1]
                        
                # cv2.imshow('img_rect'+str(i),img)
                i+=1
                print("+++++++++++++++++> ",cv2.contourArea(approx))
        # elif len(approx) == 9:
        #     print ("half-circle")
        #     cv2.drawContours(img,[cnt],0,(255,255,0),-1)
        # elif len(approx) > 15:
        #     print ("circle")
        #     cv2.drawContours(img,[cnt],0,(0,255,255),-1)

    cv2.imshow('img',img)
#     cv2.imwrite(file_path.split(".")[0]+"_out."+file_path.split(".")[-1],img)
    key = cv2.waitKey()
    if key==27:
        cv2.destroyAllWindows() 
    print("Hell yeah!!---->",i)
    print(img.shape)
    print(xmin,ymax)

img1 = "/home/nirav/Desktop/Workspace/ML/DJ_Strike/scripts/some.png"
img2 = "/home/nirav/Desktop/Workspace/ML/DJ_Strike/data/training_set/bar_graphs/185.FIG-BAR-GRAPH-STACKED-BAR-COLORS-1.png"
