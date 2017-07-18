#!/usr/bin/python2.7
# -*- coding:utf-8 -*-
__author__ = 'pi'
__version__ = '1.0'
__date__ = '04/08/2016'


import os
import sys
CurrentPath = os.path.dirname(__file__) or '.'
sys.path.insert(0, os.path.abspath(os.path.join(CurrentPath, os.path.pardir, os.path.pardir)))

import numpy as np
import numpy.testing as testNumpy
from unittest import TestCase

import ImageProcessTool as IPT


class TestConvertRoi(TestCase):
    def testZero_xyxy(self):
        Roi_xyxy = [10, 20, 10, 20]
        GTRoi_xywh = [10, 20, 0, 0]
        Roi_xywh = IPT.cvtRoi(Roi_xyxy, IPT.ROI_CVT_XYXY2XYWH)
        testNumpy.assert_array_equal(Roi_xywh, GTRoi_xywh)

    def testZero_xywh(self):
        Roi_xywh = [10, 20, 0, 0]
        GTRoi_xyxy = [10, 20, 10, 20]
        Roi_xyxy = IPT.cvtRoi(Roi_xywh, IPT.ROI_CVT_XYWH2XYXY)
        testNumpy.assert_array_equal(Roi_xyxy, GTRoi_xyxy)

    def testMinus_xyxy(self):
        Roi_xyxy = [-10, -20, -30, -40]
        GTRoi_xywh = [-10, -20, -20, -20]
        Roi_xywh = IPT.cvtRoi(Roi_xyxy, IPT.ROI_CVT_XYXY2XYWH)
        testNumpy.assert_array_equal(Roi_xywh, GTRoi_xywh)

    def testMinus_xywh(self):
        Roi_xywh = [-10, -20, -20, -20]
        GTRoi_xyxy = [-10, -20, -30, -40]
        Roi_xyxy = IPT.cvtRoi(Roi_xywh, IPT.ROI_CVT_XYWH2XYXY)
        testNumpy.assert_array_equal(Roi_xyxy, GTRoi_xyxy)


class TestGetRoiImg(TestCase):
    def setUp(self):
        self.Img = np.ones((100, 100), np.uint8)

    def testZero_xyxy(self):
        Roi_xyxy = [10, 20, 10, 20]
        _, RoiImg = IPT.getRoiImg(self.Img, Roi_xyxy, IPT.ROI_TYPE_XYXY)
        assert RoiImg is not None
        testNumpy.assert_array_equal(RoiImg, np.array([]).reshape(0, 0))

    def testZero_xywh(self):
        Roi_xywh = [10, 20, 0, 0]
        _, RoiImg = IPT.getRoiImg(self.Img, Roi_xywh, IPT.ROI_TYPE_XYWH)
        assert RoiImg is not None
        testNumpy.assert_array_equal(RoiImg, np.array([]).reshape(0, 0))

    def testMinusValid_xyxy(self):
        Roi_xyxy = [-10, -20, 30, 40]
        _, RoiImg = IPT.getRoiImg(self.Img, Roi_xyxy, IPT.ROI_TYPE_XYXY)
        assert RoiImg.shape == (40, 30)

    def testMinusValid_xywh(self):
        Roi_xywh = [-10, -20, 30, 40]
        _, RoiImg = IPT.getRoiImg(self.Img, Roi_xywh, IPT.ROI_TYPE_XYWH)
        print RoiImg.shape
        assert RoiImg.shape == (20, 20)

    def testMinusInValid_xyxy(self):
        Roi_xyxy = [-10, -20, -30, -40]
        with testNumpy.assert_raises(IPT.IPTError):
            IPT.getRoiImg(self.Img, Roi_xyxy, IPT.ROI_TYPE_XYXY)

    def testMinusInValid_xywh(self):
        Roi_xywh = [10, 20, -30, -40]
        with testNumpy.assert_raises(IPT.IPTError):
            IPT.getRoiImg(self.Img, Roi_xywh, IPT.ROI_TYPE_XYWH)

    def testNoneImg_xyxy(self):
        Roi_xyxy = [10, 20, 30, 40]
        with testNumpy.assert_raises(IPT.IPTError):
            IPT.getRoiImg(None, Roi_xyxy, IPT.ROI_TYPE_XYXY)

    def testNoneImg_xywh(self):
        Roi_xywh = [10, 20, 30, 40]
        with testNumpy.assert_raises(IPT.IPTError):
            IPT.getRoiImg(None, Roi_xywh, IPT.ROI_TYPE_XYWH)


class TestRoiType(TestCase):
    TypeFunc = list
    def testXYXY2XYWH(self):
        Roi_xyxy = self.TypeFunc([1, 1, 3, 3])
        Out = IPT.cvtRoi(roi=Roi_xyxy, flag=IPT.ROI_CVT_XYXY2XYWH)
        np.testing.assert_array_equal(Out, [1, 1, 2, 2])

    def testXYWH2XYXY(self):
        Roi_xywh = self.TypeFunc([1, 1, 2, 2])
        Out = IPT.cvtRoi(roi=Roi_xywh, flag=IPT.ROI_CVT_XYWH2XYXY)
        np.testing.assert_array_equal(Out, [1, 1, 3, 3])

    def testGetRoiImg_xyxy(self):
        Img = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
        Roi_xyxy = self.TypeFunc([1, 2, 5, 7])
        Offset_2x1, RoiImg = IPT.getRoiImg(Img, Roi_xyxy, roiType=IPT.ROI_TYPE_XYXY)
        GTRoiImg = Img[2:7, 1:5]
        GTOffset = np.array([1, 2]).reshape(2, 1)
        np.testing.assert_array_equal(GTRoiImg, RoiImg)
        np.testing.assert_array_equal(GTOffset, Offset_2x1)
        np.testing.assert_array_equal(RoiImg.shape, (5, 4))

    def testGetRoiImg_xywh(self):
        Img = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
        Roi_xywh = self.TypeFunc([1, 2, 4, 5])
        Offset_2x1, RoiImg = IPT.getRoiImg(Img, Roi_xywh, roiType=IPT.ROI_TYPE_XYWH)
        GTRoiImg = Img[2:7, 1:5]
        GTOffset = np.array([1, 2]).reshape(2, 1)
        np.testing.assert_array_equal(GTRoiImg, RoiImg)
        np.testing.assert_array_equal(GTOffset, Offset_2x1)
        np.testing.assert_array_equal(RoiImg.shape, (5, 4))

    # def testInRoi_xyxy(self):
    # def testInRoi_xywh(self):
    # def testInRoi_rotate(self):
    # def testDrawRoiImg_xywh(self):
    # def testDrawRoiImg_xyxy(self):
    # def testDrawRoiImg_rotate(self):


class TestRoiType_Array(TestRoiType):
    TypeFunc = np.array

class TestRoiType_tuple(TestRoiType):
    TypeFunc = tuple


if __name__ == '__main__':
    print IPT.test(doctests=True)