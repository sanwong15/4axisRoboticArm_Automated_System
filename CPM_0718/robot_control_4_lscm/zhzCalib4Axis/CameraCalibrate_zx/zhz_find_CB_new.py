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

    detector = HalconBoardInfoCollector(patternSize_hw=(7, 7), resolution_mm=99)

    #img = cv2.imread('Image-1.png')
    img = cv2.imread('2000-01-01-081100.jpg') 
    found, img_pts_nx2 = detector.findImgPts(img, subPix=True, filter='Bilateral', thresh=(0, 255, 15), waitTime_s=-1)

    detector.drawImagePts(img, img_pts_nx2, found)
   
    # zx, begin
    # first step: uncomment the following code line to get the image processed by opencv.
    # second step: comment the following code line to compute the mapping between image coordinate and robotics arm coordinate.
    #cv2.imwrite('2000-01-01-081100-processed.jpg', img)
    # zx, end 
    return img_pts_nx2

if __name__ == '__main__':
    os.chdir(__current_path)
    imgPts = test()
    import yaml

    with open('RobotPoints_zx_lab.yaml', 'r') as fid:
        yamlData = yaml.load(fid)

    pos = []
    for index in range(0, 49):
        pos.append(yamlData[index]['Pos'])
    pos = np.array(pos, dtype=np.float32)[:, 0:2]

    imgPts = np.require(imgPts, requirements='C', dtype=np.float32)
    pos = np.require(pos, requirements='C', dtype=np.float32)

    print 'imgPts:\n', imgPts
    print 'pos:\n', pos

    H = cv2.findHomography(imgPts, pos)[0]
    print 'H:\n', H

    #print 'zx, imgPts.shape = {}, imgPts.T.shape = {}'.format(imgPts.shape, imgPts.T.shape)
    # [541, 511] 
    # 1000.jpg
    #imgPts_test = np.array([[473.2088623, 215.57687378], [445, 238], [511, 541]]) # [[x1, y1], [x2, y2]]
    # 1001.jpg
    #imgPts_test = np.array([[345.75000, 379.00000], [469.50000, 369.50000], [673.25000, 248.25000], [783.00000, 122.75000], [889.00000, 196.75000]])
    # 1002.jpg    
    imgPts_test = np.array([[329.25000, 377.75000], 
                            [361.25000, 221.25000], 
                            [492.50000, 197.75000], 
                            [525.75000, 640.75000], 
                            [598.25000, 550.75000],
                            [636.50000, 320.50000],
                            [678.75000, 457.00000],
                            [813.50000, 323.00000],
                            [808.00000, 204.25000],
                            [898.50000, 300.00000]
                            ])
    #print 'zx, imgPts_test.shape = {}, imgPts_test.T.shape = {}'.format(imgPts_test.shape, imgPts_test.T.shape)
    pts = np.r_[imgPts.T, np.ones((1, 49))]
    print 'zx, pts:\n', pts
    print 'zx, pts.shape = {}'.format(pts.shape)
    print 'zx, imgPts_test.shape = {}'.format(imgPts_test.shape)
    pts_test = np.r_[imgPts_test.T, np.ones((1, imgPts_test.shape[0]))]
    print 'zx, pts_test.shape = {}'.format(pts_test.shape)
    print 'zx, pts_test = {}'.format(pts_test)
    dstPt = np.dot(H, pts)
    #print 'zx, H.shape = {}, pts.shape = {}, dstPt.shape = {}'.format(H.shape, pts.shape, dstPt.shape)
    dstPt_test = np.dot(H, pts_test)

    print 'zx, dstPt_test = {}'.format(dstPt_test)
    #print 'zx, dstPt_test[2] = {}'.format(dstPt_test[2])
    dstPt_test = dstPt_test / dstPt_test[2]
    dstPt = dstPt / dstPt[2]
    print 'dstPt_test:\n', dstPt_test.T
    print 'dstPt:\n', dstPt.T
    print 'error:\n', np.linalg.norm(dstPt.T[:, :2] - pos, axis=1)
