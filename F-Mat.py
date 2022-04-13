import numpy as np
from skimage import data
import matplotlib.pyplot as plt
import cv2 as cv


def Mouse_Event(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        updatedImg = cv.circle(img1, (x,y), radius=5, color=(255,0,0), thickness=-1)

        cv.imshow('Left Image',updatedImg)
        xind, yind = x, y
        pointx = np.asarray([xind, yind, 1])
        line_params = np.dot(F,pointx)
        a = line_params[0]
        b = line_params[1]
        c = line_params[2]
        px = range(0, 740)
        py = ((-a*px)-c) / b    
        plt.plot(px,py,color = 'red')
        # plt.show()

        


if __name__ == '__main__':

    img1, img2, groundtruth_disp = data.stereo_motorcycle()

    img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)


    sift = cv.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    pts1 = []
    pts2 = []

    # ratio test as per Lowe's paper
    for i, (m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            good.append(m)
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_LMEDS)
    print (F)
    plt.ion()
    fig1 = plt.figure('Right Image')
    plt.imshow(img2)
    cv.imshow('Left Image',img1)
    cv.setMouseCallback('Left Image', Mouse_Event)
    cv.waitKey(0)
    cv.destroyAllWindows() 


