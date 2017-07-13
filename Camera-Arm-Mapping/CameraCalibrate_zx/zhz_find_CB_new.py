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
    img = cv2.imread('Test-image-1.png') 
    found, img_pts_nx2 = detector.findImgPts(img, subPix=True, filter='Bilateral', thresh=(0, 255, 15), waitTime_s=-1)

    DrawImg = detector.drawImagePts(img, img_pts_nx2, found)
    
    # Trial
    # Write image
    cv2.imwrite("Test-image-1-Mapped.jpg",img)
    
    return img_pts_nx2

if __name__ == '__main__':
    os.chdir(__current_path)
    imgPts = test()
    import yaml

    # was using RobotPoints_zx_7_7.yaml
    # Then i tried RobotPoints_7x7_Mapping_mod.yaml which contain 52 points
    with open('RobotPoints_7x7_Mapping.yaml', 'r') as fid:
        yamlData = yaml.load(fid)

    pos = []
    # Range was original from 3 to 52 because WATCH point was included in the old files
    for index in range(0, 49):
        pos.append(yamlData[index]['Pos'])
    pos = np.array(pos, dtype=np.float32)[:, 0:2]

    imgPts = np.require(imgPts, requirements='C', dtype=np.float32)
    pos = np.require(pos, requirements='C', dtype=np.float32)

    print 'imgPts:\n', imgPts
    print 'pos:\n', pos

    H = cv2.findHomography(imgPts, pos)[0]
    print 'H:\n', H

    pts = np.r_[imgPts.T, np.ones((1, 49))]
    dstPt = np.dot(H, pts)

    dstPt = dstPt / dstPt[2]
    print 'dstPt:\n', dstPt.T
    print 'error:\n', numpy.linalg.norm(dstPt.T[:, :2] - pos, 1)
