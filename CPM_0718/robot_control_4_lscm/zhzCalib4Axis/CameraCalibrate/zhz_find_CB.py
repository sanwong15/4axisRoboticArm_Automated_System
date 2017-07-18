#!/usr/bin/env python2
# -*- coding:utf-8 -*-
__author__ = 'Li Hao'
__version__ = '1.0'
__date__ = '2017.02.24'
__copyright__ = "Copyright 2017, PI"

import os
import sys
import numpy as np

__current_path = os.path.dirname(__file__) or '.'
sys.path.insert(0, os.path.abspath(os.path.join(__current_path, os.path.pardir)))

import cv2

from CalibrationBoard import HalconBoardInfoCollector


def test():
    # zx, begin
    # zx, oi, begin
    #detector = HalconBoardInfoCollector(patternSize_hw=(3, 3), resolution_mm=99)
    # zx, oi, end
    detector = HalconBoardInfoCollector(patternSize_hw=(7, 7), resolution_mm=99)
    # zx, end
    img = cv2.imread('Image-1.png')
    found, img_pts_nx2 = detector.findImgPts(img, subPix=True, filter='Bilateral', thresh=(0, 255, 15), waitTime_s=-1)

    img_pts_nx2.tofile('impPts')
    print 'imgpos ： ', img_pts_nx2

    detector.drawImagePts(img, img_pts_nx2, found)
    # i = 0
    
    # for p in img_pts_nx2:
    #    cv2.circle(img, tuple(p), 10+1*i, color=[0, 0, 255, 0])
    #    i+=1
    
    #cv2.imshow('CalImg.jpg', img)
    # zx, begin
    #cv2.imwrite('./Image-1.jpg', img)
    # zx, end
    
    # zx, begin
    # zx, When you need to run the whole program, you need to comment the following two lines.
    # cv2.imshow('Image-1', img)
    # cv2.waitKey(0)
    # zx, end
    
    return img_pts_nx2




if __name__ == '__main__':
    os.chdir(__current_path)
    imgPts = test()
    import yaml

    # imgPts = np.load('impPts', )

    # zx, begin
    # zx, original implementation, begin
    #with open('RobotPoints.yaml', 'r') as fid:
    # zx, oi, end
    with open('RobotPoints_zx_7_7.yaml', 'r') as fid:
    # zx, end
        yamlData = yaml.load(fid)
    print yamlData
    pos=[]
    # zx, begin
    # zx, oi, begin
    #for index in range(5, 14):
    # zx, oi, end
    # zx, 3 and 51 are the index of data in the file of RobotPoints_zx_7_7.yaml
    for index in range(3, 52):
    # zx, end
        pos.append(yamlData[index]['Pos'])
    pos = np.array(pos, dtype=np.float32)[:, 0:2]


    # zx, begin
    # zx, oi, begin
    #oneCol = np.ones((9,1), dtype=pos.dtype)
    # zx, oi, end
    oneCol = np.ones((49,1), dtype=pos.dtype)
    # zx, end
    print oneCol
    print imgPts
    # imgPts = np.hstack((imgPts, oneCol))
    # pos = np.hstack((pos, oneCol))
    # imgPts = imgPts.T.reshape(-1, 2)
    # pos = pos.T.reshape(-1, 2)

    imgPts = np.float32(imgPts)
    pos = np.float32(pos)

    print 'imgPts : ', imgPts
    print 'pos: ', pos


    print imgPts.shape
    print pos.shape
    H = cv2.findHomography(imgPts, pos)[0]
    print 'H: ', H

    # pts = np.array([741.65, 469.68, 1], dtype=np.float32)
    # zx, begin
    # zx, oi, begin
    #pts = np.r_[imgPts.T, np.ones((1, 9))]
    # zx, oi, end
    pts = np.r_[imgPts.T, np.ones((1, 49))]
    # zx, end
    # pts = np.r_[pos.T, np.ones((1, 9))]
    # pts = pts.reshape((3,1))

    print pts, pts.shape

    dstPt = np.dot(H, pts)
    # dstImgPts = np.dot(H, pts)

    dstPt = dstPt / dstPt[2]
    # dstImgPts = dstImgPts / dstImgPts[2]
    print 'dstPt: ', dstPt.T
    # print 'dstImgPts: ', dstImgPts.T
