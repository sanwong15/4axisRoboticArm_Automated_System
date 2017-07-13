#!/usr/bin/python2.7
# -*- coding:utf-8 -*-
__author__ = 'lh'
__version__ = '1.0'
__date__ = '20/09/2015'


import os
import sys
__CurrentPath = os.path.dirname(__file__) or '.'
sys.path.insert(0, os.path.abspath(os.path.join(__CurrentPath, os.path.pardir)))

import cv2
import math
import numpy as np
import collections

import HelperFunction
import ToolBox.ImageProcessTool       as IPT
import ToolBox.NewIPT                 as NewIPT
from   ToolBox.ContourAnalyst         import ContourAnalyst
from   ToolBox.VisionGeometryLib      import VisionGeometryLib as VGL


def callBack(x):
    pass


class CalibrationBoardInfoCollector(object):
    FILTER_NONE = 'Sobel'
    FILTER_NON_LOCAL_MEANS = 'nlmean'
    FILTER_BILATERAL = 'Bilateral'
    def __init__(self, patternSize_hw, resolution_mm):
        object.__init__(self)

        self.__PatternSize = tuple(patternSize_hw)
        self.__Resolution = resolution_mm

        Height, Width = self.__PatternSize
        # self.ObjPts2D_nx2 = np.swapaxes(np.mgrid[0:Height, 0:Width].T, 0, 1).reshape(-1, 2).astype(np.float32)
        self.__ObjPts2D_nx2Ori = np.mgrid[0:Width, 0:Height].T.reshape(-1, 2)
        self.ObjPts2D_nx2 = self.__ObjPts2D_nx2Ori.copy()
        self.ObjPts2D_nx2[:, 0] = self.__ObjPts2D_nx2Ori[:, 1]
        self.ObjPts2D_nx2[:, 1] = self.__ObjPts2D_nx2Ori[:, 0]
        self.ObjPts3D_nx3 = \
            np.hstack((self.ObjPts2D_nx2, np.zeros((self.ObjPts2D_nx2.shape[0], 1), np.float32))) * self.__Resolution
        self.__AllImgPts_ListOfnx2 = []
        self.__AllObjPts_ListOfnx3 = []
        self.__ImageNumber = 0

    @property
    def AllImgPts_mxnx2(self):
        return np.float32(self.__AllImgPts_ListOfnx2)

    @property
    def AllObjPts_mxnx3(self):
        return np.float32(self.__AllObjPts_ListOfnx3)

    @property
    def ImageNumber(self):
        return self.__ImageNumber

    @property
    def ObjPts_nx3(self):
        return np.float32(self.ObjPts3D_nx3)

    @property
    def PatternSize_hw(self):
        return self.__PatternSize

    @property
    def Resolution_mm(self):
        return self.__Resolution

    # @property
    # def ObjectPoints_nx3(self):
    #     return self.ObjPts3D_nx3.copy()

    def addImgPts(self, imgPts_nx2):
        self.__AllImgPts_ListOfnx2.append(imgPts_nx2.copy())
        self.__AllObjPts_ListOfnx3.append(self.ObjPts3D_nx3.copy())
        self.__ImageNumber += 1

    def clearAllPts(self):
        self.__AllImgPts_ListOfnx2 = []
        self.__AllObjPts_ListOfnx3 = []
        self.__ImageNumber = 0

    def drawImagePts(self, img, imgPts_nx2, isFound):
        DrawImg = None
        if imgPts_nx2 is not None:
            DrawPts = imgPts_nx2.reshape(-1, 1, 2).astype(np.float32)
            DrawImg = cv2.drawChessboardCorners(img, self.PatternSize_hw[::-1], DrawPts, isFound)
        return DrawImg

    def getHomoImg_4Pts(self, srcImg, origin, xAxis, yAxis, last):
        DetectAllPts = np.array(map(np.array, [origin, xAxis, yAxis, last]))
        ImgPts_nx2 = self.ObjPts_nx3[:, :2] / self.Resolution_mm
        ImgPts_nx2 = ImgPts_nx2[:, ::-1]
        IdealOrigin = ImgPts_nx2[0, :]
        H, W = self.PatternSize_hw
        IdealLast = ImgPts_nx2[-1, :]
        IdealObjXAxis = ImgPts_nx2[-W, :]
        IdealObjYAxis = ImgPts_nx2[W-1, :]
        # print 'B', IdealOrigin
        # print 'G', IdealObjXAxis
        # print 'R', IdealObjYAxis
        # print 'C', IdealLast
        ImgSize = np.array([srcImg.shape[1], srcImg.shape[0]])
        # IdealImg = np.zeros((ImgSize[0], ImgSize[1], 3), np.uint8)
        ImgCenter = ImgSize / 2
        AllPts = np.array(map(np.array, [IdealOrigin, IdealObjXAxis, IdealObjYAxis, IdealLast]))
        Scale = [float(srcImg.shape[0]) / H, float(srcImg.shape[1]) / W]
        Distance = round(min(Scale) / 2)
        # AllPts *= Distance
        AllPts *= Distance
        # if srcImg.shape[0] < 1600:
        #     AllPts *= 50
        # else:
        #     AllPts *= 125
        Center = AllPts.mean(0)
        Offset = ImgCenter - Center
        AllPts += Offset
        HomoMatrix, _ = cv2.findHomography(DetectAllPts, AllPts.astype(np.float))
        Distance_pix = np.linalg.norm(AllPts[0] - AllPts[1]) / float(W - 1)

        HomoImg = cv2.warpPerspective(srcImg, HomoMatrix, (ImgSize[0], ImgSize[1]))
        return HomoImg, HomoMatrix, AllPts, Distance_pix

    def getHomoImg_AllPts(self, srcImg, imgPts_nx2):
        imgSize = srcImg.shape
        H, W = self.PatternSize_hw
        ImgPts_nx2 = self.ObjPts_nx3[:, :2] / self.Resolution_mm
        ImgPts_nx2 = ImgPts_nx2[:, ::-1]
        ImgSize = np.array([imgSize[1], imgSize[0]])
        ImgCenter = ImgSize / 2
        AllPts_nx2 = np.array(ImgPts_nx2)
        Scale = [float(imgSize[0]) / H, float(imgSize[1]) / W]
        Distance = round(min(Scale) / 2)
        AllPts_nx2 *= Distance
        Center = AllPts_nx2.mean(0)
        Offset = ImgCenter - Center
        AllPts_nx2 += Offset
        # Canvas = np.zeros(imgSize)
        # for i, j in zip(imgPts_nx2, AllPts_nx2):
        #     IPT.drawPoints(srcImg, i.T, (0, 0, 255))
        #     IPT.drawPoints(Canvas, j.T, (0, 0, 255))
        #     cv2.imshow('srcImg', srcImg)
        #     cv2.imshow('Canvas', Canvas)
        #     cv2.waitKey()
        HomoMatrix, _ = cv2.findHomography(imgPts_nx2.astype(np.float), AllPts_nx2.astype(np.float))
        HomoImg = cv2.warpPerspective(srcImg, HomoMatrix, (ImgSize[0], ImgSize[1]))
        return HomoImg, HomoMatrix, Distance

    def getAllHomoMatrix(self, imgSize, allImgPts_mxnx2, foundMaskList, goodPtsMask_gxnx1, goodImgMask_list):
        H, W = self.PatternSize_hw
        ImgPts_nx2 = self.ObjPts_nx3[:, :2] / self.Resolution_mm
        ImgPts_nx2 = ImgPts_nx2[:, ::-1]
        ImgSize = np.array([imgSize[1], imgSize[0]])
        ImgCenter = ImgSize / 2
        AllPts_nx2 = np.array(ImgPts_nx2)
        Scale = [float(imgSize[0]) / H, float(imgSize[1]) / W]
        Distance = round(min(Scale) / 2)
        AllPts_nx2 *= Distance
        Center = AllPts_nx2.mean(0)
        Offset = ImgCenter - Center
        AllPts_nx2 += Offset
        HomoMatrixList = []
        FoundIdx = 0
        for idx, imgPts in enumerate(allImgPts_mxnx2):
            if foundMaskList[idx]:
                GMask_nx1 = goodPtsMask_gxnx1[FoundIdx].reshape(-1)
                ImgPts = allImgPts_mxnx2[idx][GMask_nx1]
                PickupPts_nx2 = AllPts_nx2[GMask_nx1]
                HomoMatrix, _ = cv2.findHomography(ImgPts, PickupPts_nx2.astype(np.float))
                FoundIdx += 1
            else:
                HomoMatrix = np.eye(3)
            HomoMatrixList.append(HomoMatrix)
        return HomoMatrixList


class ChessboardInfoCollector(CalibrationBoardInfoCollector):
    def getBoardPicture(self, blockDisXY_pixel=(10,10), foregroundColor=(0,0,0), backgroundColor=(255,255,255),
                              outerBoardXY=(10, 10)):

        Height, Width = self.PatternSize_hw
        DrawPts_hxwx2 = (np.mgrid[0:Width+2, 0:Height+2].T).astype(np.float32)
        DrawPts_hxwx2[:, 0, :] = (DrawPts_hxwx2[:, 0, :] + DrawPts_hxwx2[:, 1, :]) / 2
        DrawPts_hxwx2[:, -1, :] = (DrawPts_hxwx2[:, -1, :] + DrawPts_hxwx2[:, -2, :]) / 2
        DrawPts_hxwx2[:, :, 0] *= blockDisXY_pixel[0]
        DrawPts_hxwx2[:, :, 1] *= blockDisXY_pixel[1]
        DrawPts_hxwx2 -= (DrawPts_hxwx2[0, 0, :] - np.array(outerBoardXY))
        ImgPts_hxwx2 = DrawPts_hxwx2[1:-1, 1:-1, :]
        ImgPts_nx2 = ImgPts_hxwx2.reshape(-1, 2)
        h, w, d = DrawPts_hxwx2.shape
        ImgShape = (DrawPts_hxwx2[-1, -1, :] + np.array(outerBoardXY)).astype(np.int)
        Img = np.zeros((ImgShape[1], ImgShape[0], 3), np.uint8)
        Img[:] = backgroundColor
        for row in xrange(h-1):
            for col in xrange(w-1):
                if (row + col) % 2 == 0:
                    cv2.rectangle(Img, pt1=tuple(DrawPts_hxwx2[row, col, :].astype(np.int)),
                                  pt2=tuple(DrawPts_hxwx2[row+1, col+1, :].astype(np.int)),
                                  color=foregroundColor, thickness=-1, lineType=cv2.cv.CV_AA, shift=0)
        return Img, ImgPts_nx2

    def findImgPts(self, img, subPix=True):
        GrayImg = HelperFunction.convertImg_BGR2Gray(img)
        IsFound, ImgPts_nx1x2 = cv2.findChessboardCorners(image=GrayImg, patternSize=self.PatternSize_hw)
        if subPix and IsFound:
            ImgPts_nx1x2 = self.__subPixel(ImgPts_nx1x2, GrayImg)
        Height, Width = self.PatternSize_hw
        ImgPts_nx2 = []
        for i in xrange(Height):
            for j in xrange(Width):
                a = (Width - j - 1) * Height
                b = i
                ImgPts_nx2.append(ImgPts_nx1x2[a + b, 0, :])
        return IsFound, np.array(ImgPts_nx2)

    def __subPixel(self, corners_nx1x2, img, windowSize=(11, 11), eps=0.1, iterMax=30):
        GrayImg = HelperFunction.convertImg_BGR2Gray(img)
        SubPix_zeroZoneSize = (-1, -1)
        SubPix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, iterMax, eps)
        CornersSub_nx1x2 = corners_nx1x2.copy()
        cv2.cornerSubPix(image=GrayImg, corners=CornersSub_nx1x2,
                         winSize=windowSize, zeroZone=SubPix_zeroZoneSize, criteria=SubPix_criteria)
        return CornersSub_nx1x2
    # @classmethod
    # def drawChessboard(cls, pattern_size, blockDisXY_pixel=(10,10), originPoint=(0,0), img=None,
    #                    foregroundColor=(0,0,0), backgroundColor=(255,255,255), outerBoard=None, outerBoardThickness=1):
    #     if img is None:
    #         ScreenResolution = cls.__getScreenResolution()
    #         img = np.zeros((ScreenResolution[1], ScreenResolution[0], 3), dtype=np.uint8)
    #         img[:] = backgroundColor
    #
    #     for i in range(pattern_size[0]+1):
    #         for j in range(pattern_size[1]+1):
    #             if (i+j) % 2:
    #                 x1 = originPoint[0] + i * blockDisXY_pixel[0]
    #                 x2 = originPoint[0] + (i + 1) * blockDisXY_pixel[1]
    #                 y1 = originPoint[1] + j * blockDisXY_pixel[1]
    #                 y2 = originPoint[1] + (j + 1) * blockDisXY_pixel[1]
    #                 img[x1: x2, y1: y2, :] = foregroundColor
    #     if outerBoard is not None:
    #         start = (int(originPoint[1]-outerBoard[0]-outerBoardThickness),
    #                  int(originPoint[0]-outerBoard[1]-outerBoardThickness))
    #         end = (int(originPoint[1]+ blockDisXY_pixel[0]*(pattern_size[1]+1) +outerBoard[0]),
    #                int(originPoint[0]+ blockDisXY_pixel[1]*(pattern_size[0]+1) +outerBoard[1]))
    #         cv2.rectangle(img, pt1=start, pt2=end, color=0, thickness=outerBoardThickness, lineType=cv2.cv.CV_AA, shift=0)
    #
    #     return img
    #


class HalconBoardInfoCollector(CalibrationBoardInfoCollector):
    def getBoardPicture(self, blockDisXY_pixel=(10,10), foregroundColor=(0,0,0), backgroundColor=(255,255,255),
                              outerBoardXY=None, outerBoardThickness=1, triangularDis=1.0):
        HalconCircleRadius = int(min(blockDisXY_pixel)*0.618*0.5)
        Height, Width = self.PatternSize_hw
        if outerBoardXY is not None:
            ImgHeight = blockDisXY_pixel[1] * (Height - 1) + 2*HalconCircleRadius + 2*outerBoardXY[1] + outerBoardThickness
            ImgWidth = blockDisXY_pixel[0] * (Width - 1) + 2*HalconCircleRadius + 2*outerBoardXY[0] + outerBoardThickness
            Offset = np.array([outerBoardXY[0], outerBoardXY[1]]) + HalconCircleRadius + outerBoardThickness/2
        else:
            ImgHeight = blockDisXY_pixel[1] * (Height - 1) + 4*HalconCircleRadius
            ImgWidth = blockDisXY_pixel[0] * (Width - 1) + 4*HalconCircleRadius
            Offset = np.array([HalconCircleRadius*2]*2)

        Img = np.zeros((ImgHeight, ImgWidth, 3), dtype=np.uint8)
        Img[:] = backgroundColor
        ImgPts_nx2 = self.ObjPts2D_nx2.copy()
        ImgPts_nx2[:, 0], ImgPts_nx2[:, 1] = self.ObjPts2D_nx2[:, 1], self.ObjPts2D_nx2[:, 0]
        ImgPts_nx2[:, 0] *= blockDisXY_pixel[0]
        ImgPts_nx2[:, 1] *= blockDisXY_pixel[1]
        ImgPts_nx2 += Offset
        ImgPtsList = np.vsplit(ImgPts_nx2, ImgPts_nx2.shape[0])
        for point in ImgPtsList:
            x, y = point.ravel()
            cv2.circle(Img, (x, y), radius=HalconCircleRadius, color=foregroundColor, thickness=-1, lineType=cv2.cv.CV_AA)
        if outerBoardXY is not None:
            UpperLeft = ImgPts_nx2[0, :] - HalconCircleRadius - np.array(outerBoardXY)
            BottomRight = ImgPts_nx2[-1, :] + HalconCircleRadius + np.array(outerBoardXY)
            start = (int(UpperLeft[0] + HalconCircleRadius * 2 * triangularDis),
                     int(UpperLeft[1]))
            end = (int(UpperLeft[0]),
                   int(UpperLeft[1] + HalconCircleRadius * 2 * triangularDis))
            Triangular = np.array([start, tuple(UpperLeft.astype(np.int)), end])
            cv2.drawContours(Img, [Triangular], 0, color=0, thickness=-1, lineType=cv2.cv.CV_AA)
            cv2.rectangle(Img, pt1=tuple(UpperLeft.astype(np.int)), pt2=tuple(BottomRight.astype(np.int)), color=0, thickness=outerBoardThickness, lineType=cv2.cv.CV_AA, shift=0)
        return Img, ImgPts_nx2

    def findImgPtsInHomoImg(self, HomoImg, HomoMatrix, Thresh, filter, roiWidth):
        FoundHomoBoard, HomoBoardContour, HomoCircleContours, HomoCirclePts_nx2 = \
            self.findCalibrationBoard(HomoImg, Thresh)
        if not FoundHomoBoard:
            return None, None
        FoundHomoSubPix, HomoCirclePts_nx2 = \
            self.__subPixel(img=HomoImg, circleContours=HomoCircleContours, roiWidth=roiWidth, type=filter, show=True)
        if not FoundHomoSubPix:
            return None, None
        HomoExtracted, HomoOriginPoint, HomoXAxis, HomoYAxis, HomoLastPoint = \
            self.extractKeyPoints(calibrationBoardContour=HomoBoardContour, circlePts_nx2=HomoCirclePts_nx2)
        if not HomoExtracted:
            return None, None
        # IPT.drawPoints(HomoImg, HomoOriginPoint.reshape(2, 1), (255, 0, 0))
        # IPT.drawPoints(HomoImg, HomoXAxis.reshape(2, 1), (0, 255, 0))
        # IPT.drawPoints(HomoImg, HomoYAxis.reshape(2, 1), (0, 0, 255))
        HomoSortedPts_nx2 = self.sortByDist(HomoCirclePts_nx2, originPt=HomoOriginPoint,
                                              xAxisPt=HomoXAxis, yAxisPt=HomoYAxis, size=self.PatternSize_hw)
        HomoSortedPts_2xn = HomoSortedPts_nx2.T.reshape(2, -1)
        OriginSortedPts_2xn = VGL.projectPts(pts=HomoSortedPts_2xn, projectMatrix=np.linalg.inv(HomoMatrix))
        OriginSortedPts_nx2 = OriginSortedPts_2xn.T.reshape(-1, 2)
        return OriginSortedPts_nx2, HomoSortedPts_nx2

    def findImgPts(self, img, subPix=True, filter='Bilateral', thresh=(0, 255, 15), waitTime_s=1):
        print 'finding...'
        GrayImg = HelperFunction.convertImg_BGR2Gray(img)
        # CalibrationBoardContour = None
        # CirclePts_nx2= None
        # sortSucceed = None
        # SortedCirclePts = None
        FoundCB = False
        FirstThresh = None
        LastThresh = None
        for Thresh in range(thresh[0], thresh[1], thresh[2]):
            FoundBoard, CalibrationBoardContour, CircleContours, CirclePts_nx2 = self.findCalibrationBoard(GrayImg, Thresh)
            if FoundBoard:
                if FirstThresh is None:
                    FirstThresh = Thresh
                LastThresh = Thresh
                FoundCB = True
                # ================ Old logic ================ #
                # FoundSubPix, CirclePtsTemp_nx2 = self.__subPixel(img=GrayImg, circleContours=CircleContours, type=filter)
                # if FoundSubPix:
                #     CirclePts_nx2 = CirclePtsTemp_nx2
                # sortSucceed, SortedCirclePts, _, _ = \
                #     self.__sortCirclePoint(img=img, calibrationBoardContour=CalibrationBoardContour,
                #                            circlePts=CirclePts_nx2, size=self.PatternSize_hw)
                # if sortSucceed:
                #     if FirstThresh is None:
                #         FirstThresh = Thresh
                #     LastThresh = Thresh
                #     FoundCB = True
                # ============== Old logic end ============== #
        if FoundCB:
            # print 'FirstThresh: ', FirstThresh
            # print 'LastThresh: ', LastThresh
            for scale in [0.5, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8, 0.1, 0.9, 0.0, 1.0]:
                Thresh = FirstThresh * scale + LastThresh * (1.0 - scale)
                FoundOriginBoard, OriginBoardContour, OriginCircleContours, OriginCirclePts_nx2 = \
                    self.findCalibrationBoard(GrayImg, Thresh)
                if FoundOriginBoard:
                    extracted, OriginPoint, OriginXObjAxis, OriginYObjAxis, LastPoint = \
                        self.extractKeyPoints(calibrationBoardContour=OriginBoardContour, circlePts_nx2=OriginCirclePts_nx2)
                    # IPT.drawPoints(img, OriginXObjAxis.reshape(2, 1), (0, 0, 255))
                    # IPT.drawPoints(img, OriginYObjAxis.reshape(2, 1), (0, 255, 0))
                    # cv2.imshow('Img', img)
                    # cv2.waitKey()

                    if not extracted:
                        continue

                    HomoImg, HomoMatrix, AllPts, Distance_pix = \
                        self.getHomoImg_4Pts(img, origin=OriginPoint, xAxis=OriginXObjAxis, yAxis=OriginYObjAxis, last=LastPoint)
                    OriginSortedPts_nx2, HomoSortedPts_nx2 = self.findImgPtsInHomoImg(HomoImg, HomoMatrix, Thresh, filter, roiWidth=Distance_pix)
                    if OriginSortedPts_nx2 is None:
                        continue
                    # FoundHomoBoard, HomoBoardContour, HomoCircleContours, HomoCirclePts_nx2 = \
                    #     self.findCalibrationBoard(HomoImg, Thresh)
                    # if not FoundHomoBoard:
                    #     continue
                    # FoundHomoSubPix, HomoCirclePts_nx2 = \
                    #     self.__subPixel(img=HomoImg, circleContours=HomoCircleContours, type=filter, show=True)
                    # if not FoundHomoSubPix:
                    #     continue
                    # HomoExtracted, HomoOriginPoint, HomoXAxis, HomoYAxis, HomoLastPoint = \
                    #     self.extractKeyPoints(calibrationBoardContour=HomoBoardContour, circlePts_nx2=HomoCirclePts_nx2)
                    # if not HomoExtracted:
                    #     continue
                    # # IPT.drawPoints(HomoImg, HomoOriginPoint.reshape(2, 1), (255, 0, 0))
                    # # IPT.drawPoints(HomoImg, HomoXAxis.reshape(2, 1), (0, 255, 0))
                    # # IPT.drawPoints(HomoImg, HomoYAxis.reshape(2, 1), (0, 0, 255))
                    # # IPT.drawPoints(HomoImg, AllPts[0].reshape(2, 1), (255, 0, 0))
                    # # IPT.drawPoints(HomoImg, AllPts[1].reshape(2, 1), (0, 255, 0))
                    # # IPT.drawPoints(HomoImg, AllPts[2].reshape(2, 1), (0, 0, 255))
                    # # cv2.imshow('Homo', HomoImg)
                    # # cv2.waitKey()
                    # HomoSortedPts_nx2 = self.sortByDist(HomoCirclePts_nx2, originPt=HomoOriginPoint,
                    #                                       xAxisPt=HomoXAxis, yAxisPt=HomoYAxis, size=self.PatternSize_hw)
                    # # HomoSortedPts_nx2 = self.sortByDist(HomoCirclePts_nx2, originPt=AllPts[0],
                    # #                                       xAxisPt=AllPts[1], yAxisPt=AllPts[2], size=self.PatternSize_hw)
                    # HomoSortedPts_2xn = HomoSortedPts_nx2.T.reshape(2, -1)
                    # OriginSortedPts_2xn = VGL.projectPts(pts=HomoSortedPts_2xn, projectMatrix=np.linalg.inv(HomoMatrix))
                    # OriginSortedPts_nx2 = OriginSortedPts_2xn.T.reshape(-1, 2)
                    NewHomoImg, NewHomoMatrix, RoiWidth = self.getHomoImg_AllPts(img, OriginSortedPts_nx2)
                    OriginSortedPts_nx2, NewHomoSortedPts_nx2 = self.findImgPtsInHomoImg(NewHomoImg, NewHomoMatrix, Thresh, filter, roiWidth=RoiWidth)
                    if OriginSortedPts_nx2 is None:
                        continue
                    if waitTime_s >= 0:
                        # cv2.drawContours(img, [CalibrationBoardContour], 0, (0,255,0))
                        # DiffImg = np.fabs(NewHomoImg.astype(np.float) - HomoImg.astype(np.float)).astype(np.uint8)
                        # cv2.namedWindow('DiffImg', cv2.WINDOW_NORMAL)
                        # cv2.imshow('DiffImg', DiffImg)
                        self.drawImagePts(img, OriginSortedPts_nx2, True)
                        cv2.namedWindow('result', cv2.WINDOW_NORMAL)
                        cv2.imshow('result', img)
                        self.drawImagePts(HomoImg, HomoSortedPts_nx2, True)
                        cv2.namedWindow('Homo', cv2.WINDOW_NORMAL)
                        cv2.imshow('Homo', HomoImg)
                        # self.drawImagePts(NewHomoImg, NewHomoSortedPts_nx2, True)
                        # cv2.namedWindow('NewHomoImg', cv2.WINDOW_NORMAL)
                        # cv2.imshow('NewHomoImg', NewHomoImg)
                        Key = chr(cv2.waitKey(int(waitTime_s*1000)) & 255)
                        if 'q' == Key:
                            raise KeyboardInterrupt, 'Pressed [q] to quit!'
                    return True, OriginSortedPts_nx2
        print 'Not found!'
        if waitTime_s >= 0:
            cv2.imshow('Homo', np.zeros(img.shape[:2][::-1], np.uint8))
            cv2.imshow('subPixel', np.zeros(img.shape[:2][::-1], np.uint8))
            cv2.imshow('result', img)
            Key = chr(cv2.waitKey(int(waitTime_s*1000)) & 255)
            if 'q' == Key:
                raise KeyboardInterrupt, 'Pressed [q] to quit!'
        return False, None

    def __LocalHist(self, grayImg):
        CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        CLAHEImg = CLAHE.apply(grayImg)
        return CLAHEImg

    def __GlobalHist(self, grayImg):
        HistImg = cv2.equalizeHist(grayImg)
        return HistImg

    def __FixThresh(self, grayImg, thresh, invFlag=False):
        if invFlag:
            Flag = cv2.THRESH_BINARY
        else:
            Flag = cv2.THRESH_BINARY_INV
        _, BinImg = cv2.threshold(src=grayImg, thresh=thresh, maxval=255, type=Flag)
        return thresh, BinImg

    def __OTSUThresh(self, grayImg, threshPercent=1.0, invFlag=False):
        if invFlag:
            Flag = cv2.THRESH_BINARY
        else:
            Flag = cv2.THRESH_BINARY_INV

        if threshPercent == 1.0:
            ThreshValue, BinImg = cv2.threshold(src=grayImg, thresh=0, maxval=255, type=Flag + cv2.THRESH_OTSU)
        else:
            OTSUThresh, _ = cv2.threshold(src=grayImg, thresh=0, maxval=255, type=cv2.THRESH_OTSU)
            ThreshValue = OTSUThresh * threshPercent
            _, BinImg = cv2.threshold(src=grayImg, thresh=ThreshValue, maxval=255, type=Flag)
        return ThreshValue, BinImg

    def __AdaptiveThresh(self, grayImg, fov=0.25, calibrationBoardSize=(7, 7), invFlag=False):
        if invFlag:
            Flag = cv2.THRESH_BINARY
        else:
            Flag = cv2.THRESH_BINARY_INV
        h, w = grayImg.shape[:2]
        Area = h * w * fov
        CircleNum = calibrationBoardSize[0] * calibrationBoardSize[1]
        BlockSize = math.sqrt(Area) / CircleNum
        if 0 == BlockSize % 2:
            BlockSize += 1
        BinImg = cv2.adaptiveThreshold(src=grayImg, maxValue=255,
                                       adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                       thresholdType=Flag, blockSize=int(BlockSize), C=1)
        return None, BinImg

    def __subPixel(self, img, circleContours, roiWidth, type='Sobel', show=False):
        if type is None:
            Type = None
        else:
            Type = type.lower()
        if show:
            ShowImg = img.copy()
        GrayImg = HelperFunction.convertImg_BGR2Gray(img)
        CirclePts = []
        for Contour in circleContours:
            Roi_xywh = cv2.boundingRect(points=Contour)
            NewRoi_xywh = np.array(Roi_xywh)
            # AddLen = int(round(NewRoi_xywh[3] / 10.0)) + 3
            # # AddLen = 3
            # NewRoi_xywh[0:2] -= AddLen
            # NewRoi_xywh[2:4] += AddLen*2

            Found, Centroid = IPT.calcCentroid(Contour)
            if not Found:
                raise ValueError('calcCentroid error!')
            NewRoi_xywh[0:2] = Centroid.ravel() - int(round((roiWidth / 2)))
            NewRoi_xywh[2:4] = int(round(roiWidth))

            RoiImg = GrayImg[NewRoi_xywh[1]:NewRoi_xywh[1]+NewRoi_xywh[3],
                              NewRoi_xywh[0]:NewRoi_xywh[0]+NewRoi_xywh[2]]
            if RoiImg.shape[0] < 3 or RoiImg.shape[1] < 3:
                return False, None
            cv2.namedWindow("RoiImg", cv2.WINDOW_NORMAL)
            cv2.imshow("RoiImg", RoiImg)

            if Type is None or 'sobel' == Type:
                FilteredImg = RoiImg
                WinName = 'NoFilter'
            elif 'bilateral' == Type:
                FilteredImg = cv2.bilateralFilter(src=RoiImg, d=9, sigmaColor=75, sigmaSpace=75)
                WinName = 'Bilateral'
            elif 'nlmean' == Type:
                FilteredImg = cv2.fastNlMeansDenoising(RoiImg, h=3, templateWindowSize=7, searchWindowSize=21)
                WinName = 'NoneLocalMeans'
            else:
                raise ValueError, 'sub pixel type error!'

            cv2.namedWindow(WinName, cv2.WINDOW_NORMAL)
            cv2.imshow(WinName, FilteredImg)

            SobelX = cv2.Sobel(src=FilteredImg, ddepth=cv2.CV_32FC1, dx=1, dy=0)
            SobelY = cv2.Sobel(src=FilteredImg, ddepth=cv2.CV_32FC1, dx=0, dy=1)
            Kernel = np.array([[  2,  1,  0],
                               [  1,  0, -1],
                               [  0, -1, -2]])
            SobelZ = cv2.filter2D(src=FilteredImg, ddepth=cv2.CV_32FC1, kernel=Kernel)
            Kernel = np.array([[  0,  1,  2],
                               [ -1,  0,  1],
                               [ -2, -1,  0]])
            SobelW = cv2.filter2D(src=FilteredImg, ddepth=cv2.CV_32FC1, kernel=Kernel)
            SobelX = np.uint8(abs(SobelX))
            SobelY = np.uint8(abs(SobelY))
            SobelZ = np.uint8(abs(SobelZ))
            SobelW = np.uint8(abs(SobelW))
            SobelXY = cv2.add(src1=SobelX, src2=SobelY)
            SobelWZ = cv2.add(src1=SobelW, src2=SobelZ)
            Sobel = cv2.add(src1=SobelXY, src2=SobelWZ)
            Thresh, BinImg = cv2.threshold(src=Sobel, thresh=0, maxval=255, type=cv2.THRESH_OTSU)
            cv2.namedWindow("SubPixel", cv2.WINDOW_NORMAL)
            cv2.imshow("SubPixel", BinImg)
            Contours, Hierarchy = cv2.findContours(image=BinImg.copy(),
                                                   mode=cv2.RETR_TREE,
                                                   method=cv2.CHAIN_APPROX_NONE)
            if not Contours:
                cv2.namedWindow('SobelBinImg', cv2.WINDOW_NORMAL)
                cv2.imshow('SobelBinImg', BinImg)
                cv2.waitKey()
            MaxIdx = IPT.findMaxAreaContours(contours=Contours, num=2)
            if 2 != len(MaxIdx):
                return False, None
            OutsideContour = Contours[MaxIdx[0]]
            InsideContour = Contours[MaxIdx[1]]
            Px1, Py1 = ContourAnalyst.getCentroid(OutsideContour).ravel() + NewRoi_xywh[:2].ravel()
            Px2, Py2 = ContourAnalyst.getCentroid(InsideContour).ravel() + NewRoi_xywh[:2].ravel()
            Px = (Px1 + Px2) / 2
            Py = (Py1 + Py2) / 2
            CirclePts.append([Px, Py])
            if show:
                IPT.drawRoi(ShowImg, roi=NewRoi_xywh, roiType=IPT.ROI_TYPE_XYWH, color=(0, 255, 0))
                IPT.drawContours(ShowImg, [OutsideContour], -1, color=(255, 0, 0), offset=tuple(NewRoi_xywh[:2]))
                IPT.drawContours(ShowImg, [InsideContour], -1, color=(0, 0, 255), offset=tuple(NewRoi_xywh[:2]))
        CirclePts_nx2 = np.array(CirclePts)
        if show:
            cv2.namedWindow('subPixel', cv2.WINDOW_NORMAL)
            cv2.imshow('subPixel', ShowImg)
        return True, CirclePts_nx2

    def findCalibrationBoard(self, img, thresh, invFlag=False, show=False):
        if 3 == img.ndim:
            Gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
        else:
            Gray = img.copy()
        Size = self.PatternSize_hw
        Gray = cv2.blur(Gray, ksize=(3, 3))
        # ----------------- hist ----------------- #
        Gray = self.__LocalHist(Gray)
        # Gray = self.__GlobalHist(Gray)

        # ----------------- thresh ----------------- #
        Thresh = thresh
        InvFlag = invFlag
        _, BinImg = self.__FixThresh(Gray, Thresh, invFlag=InvFlag)
        if show:
            cv2.namedWindow('Bin', cv2.WINDOW_NORMAL)
            cv2.imshow('Bin', BinImg)
        # ----------------- morphology ----------------- #
        # Iterations = 1
        # BinImg = cv2.morphologyEx(src=BinImg, op=cv2.MORPH_OPEN, kernel=np.ones((3,3)), iterations=Iterations)
        # BinImg = cv2.morphologyEx(src=BinImg, op=cv2.MORPH_CLOSE, kernel=np.ones((5,5)), iterations=Iterations)
        Contours, Hierarchy = cv2.findContours(image=BinImg.copy(),
                                               mode=cv2.RETR_TREE,
                                               method=cv2.CHAIN_APPROX_NONE)
        if Hierarchy is None:
            return False, None, None, None
        CandidateBoardContoursIdx = []
        for ContourIdx, SubContoursNum in collections.Counter(Hierarchy[0,:,3]).items():
            if SubContoursNum >= Size[0]*Size[1]:
                if -1 != Hierarchy[0, ContourIdx, 2]:
                    CandidateBoardContoursIdx.append(ContourIdx)
        NewCandidate = []
        CircleNum = self.PatternSize_hw[0] * self.PatternSize_hw[1]
        for candidate in CandidateBoardContoursIdx:
            if -1 == candidate:
                continue
            sons = np.where(Hierarchy[0,:,3] == candidate)[0]
            map(cv2.contourArea, Contours)
            # sonsArea = np.array([cv2.contourArea(Contours[son]) for son in sons])
            sonsArea = [cv2.contourArea(Contours[son]) for son in sons]
            sonsArea = [area for area in sonsArea if area > 0]
            if len(sonsArea) < CircleNum:
                continue
            sonsArea.sort()
            sonsArea = np.array(sonsArea[::-1][:CircleNum])
            if sonsArea.std() / sonsArea.mean() < 0.4:
                NewCandidate.append(candidate)
        CandidateBoardContoursIdx = NewCandidate
        if 1 != len(CandidateBoardContoursIdx):
            # print "CandidateBoardContoursIdx:\n", CandidateBoardContoursIdx
            return False, None, None, None
        else:
            SubContours = [Contours[i] for i in xrange(len(Hierarchy[0])) if Hierarchy[0][i][3] == CandidateBoardContoursIdx[0]]
            IPT.findMaxAreaContours(contours=SubContours, num=Size[0]*Size[1])
            MaxSubContours =  SubContours[len(SubContours) - Size[0]*Size[1] : ]
            CirclePoints = []
            for CircleContour in MaxSubContours:
                Flag, Pt_2x1 = IPT.calcCentroid(CircleContour)
                if Flag:
                    Px = Pt_2x1[0, 0]
                    Py = Pt_2x1[1, 0]
                else:
                    return False, None, None, None
                CirclePoints.append([Px, Py])
            CirclePts_nx2 = np.array(CirclePoints)
            return True, Contours[CandidateBoardContoursIdx[0]], MaxSubContours, CirclePts_nx2

    def __approxCBContour(self, calibrationBoardContour, accuracy, maxIter):
        Accuracy = accuracy
        PrePolyPts = None
        IterMax = maxIter
        while True:
            Accuracy += 1
            PolyPts = cv2.approxPolyDP(curve=calibrationBoardContour, epsilon=Accuracy, closed=True)
            if 4 == len(PolyPts):
                break
            if Accuracy > IterMax:
                return None
            PrePolyPts = PolyPts
        if PrePolyPts is None:
            print 'PrePolyPts error!'
            return None
        return PrePolyPts

    def __getOriginPoint(self, polyPts, calibrationBoardContour, circlePts_nx2):
        PolyPtsNum = len(polyPts)
        CircleToPolyMinDisPtsIdx = np.ones((PolyPtsNum, ))
        for i, PolyPt in enumerate(polyPts):
            MinDis = len(calibrationBoardContour)
            for j, CirclePt in enumerate(circlePts_nx2):
                DisPoly2Circle = np.linalg.norm(PolyPt - CirclePt)
                if DisPoly2Circle < MinDis:
                    MinDis = DisPoly2Circle
                    CircleToPolyMinDisPtsIdx[i] = j

        RepeatCornerPtsIdx = []
        UnRepeatCornerPtsIdx = []
        for CirclePtsIdx, RepeatTimes in collections.Counter(CircleToPolyMinDisPtsIdx).items():
            if RepeatTimes > 1:
                RepeatCornerPtsIdx.append(CirclePtsIdx)
            else:
                UnRepeatCornerPtsIdx.append(CirclePtsIdx)
        if 1 != len(RepeatCornerPtsIdx):
            return  None, None
        elif 3 != len(UnRepeatCornerPtsIdx):
            return None, None
        OriginPoint = np.array(circlePts_nx2[int(RepeatCornerPtsIdx[0])])
        OtherCornerPts = np.array([circlePts_nx2[int(i)] for i in UnRepeatCornerPtsIdx])
        return OriginPoint, OtherCornerPts

    def __pickupXYLastPt(self, originPt, otherPts):
        Vecs  = otherPts - originPt
        MaxAng_rad = 0
        YObjAxis = None
        XObjAxis = None
        PickX = None
        PickY = None
        for i in xrange(3):
            for j in xrange(i+1, 3):
                Ang_rad = math.acos(np.inner(Vecs[i], Vecs[j]) / (np.linalg.norm(Vecs[i]) * np.linalg.norm(Vecs[j])))
                if Ang_rad > MaxAng_rad:
                    MaxAng_rad = Ang_rad
                    if (np.cross(Vecs[i], Vecs[j])) > 0:
                        YObjAxis = otherPts[i]
                        XObjAxis = otherPts[j]
                    else:
                        XObjAxis = otherPts[i]
                        YObjAxis = otherPts[j]
                    PickX, PickY = i, j
        PtsList = otherPts.tolist()
        if PickX > PickY:
            PtsList.pop(PickX)
            PtsList.pop(PickY)
        else:
            PtsList.pop(PickY)
            PtsList.pop(PickX)
        LastPoint = PtsList.pop()
        return XObjAxis, YObjAxis, LastPoint

    def findSortedCirclePoint(self, calibrationBoardContour, circlePts):
        Extracted, OriginPoint, XAxis, YAxis, LastPoint = \
            self.extractKeyPoints(calibrationBoardContour=calibrationBoardContour, circlePts_nx2=circlePts)

        if not Extracted:
            return False, None
        SortedPts_nx2 = self.sortByDist(circlePts, originPt=OriginPoint, xAxisPt=XAxis, yAxisPt=YAxis, size=self.PatternSize_hw)
        return True, SortedPts_nx2

    def extractKeyPoints(self, calibrationBoardContour, circlePts_nx2):
        PrePolyPts = self.__approxCBContour(calibrationBoardContour, accuracy=0, maxIter=1024)
        if PrePolyPts is None:
            return False, None, None, None, None
        OriginPoint, OtherCornerPts = \
            self.__getOriginPoint(polyPts=PrePolyPts, calibrationBoardContour=calibrationBoardContour, circlePts_nx2=circlePts_nx2)

        if OriginPoint is None:
            return False, None, None, None, None
        XObjAxis, YObjAxis, LastPoint = \
            self.__pickupXYLastPt(originPt=OriginPoint, otherPts=OtherCornerPts)
        return True, OriginPoint, XObjAxis, YObjAxis, LastPoint

    def sortByDist(self, circlePts, originPt, xAxisPt, yAxisPt, size):
        TempList = []
        for Point in circlePts.copy():
            DisP2ObjY = HelperFunction.calcDistancePoint2Line(Point, originPt, yAxisPt)
            DisP2ObjX = HelperFunction.calcDistancePoint2Line(Point, originPt, xAxisPt)
            Temp = {}
            Temp['DisP2ObjY'] = DisP2ObjY
            Temp['DisP2ObjX'] = DisP2ObjX
            Temp['Value'] = Point
            TempList.append(Temp)
        TempList.sort(key=lambda s:s['DisP2ObjY'])
        TempList = np.array(TempList).reshape(size).tolist()
        for List in TempList:
            List.sort(key=lambda s:s['DisP2ObjX'])
        TempList = np.array(TempList).reshape(size[0]*size[1],).tolist()
        SortedPts = circlePts.copy()
        for i, Object in enumerate(TempList):
            SortedPts[i] = Object['Value']
        return SortedPts


# import newSub.util as util
# import newSub.detect as detect
# import newSub.fitellipse as fitellipse
# class newSubPixel(HalconBoardInfoCollector):
#     def getGradientImg(self, img):
#         sobelx64f = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
#         abs_sobelx64f = np.abs(sobelx64f)
#         sobely64f = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
#         abs_sobely64f = np.abs(sobely64f)
#         sobel = np.add(abs_sobelx64f, abs_sobely64f)
#         factor = np.max(sobel)
#         sobel = np.uint8(sobel/factor*255)  # try to scale the result otherwise it won't be saved correctly
#         return sobel
#
#     def __findPts(self, image, thread):
#         sigma = 1.51  # The default values can be found in util
#         low = 2.05
#         high = 6
#         num_quadrant = 50
#         circle_quadrant = 20
#         height, width = image.shape[:2]
#         board = np.zeros((height, width))
#         board[:] = 255
#         GrayImg = HelperFunction.convertImg_BGR2Gray(image)
#         Flag, CalibrationBoardContour, CircleContours, CirclePts_nx2 = self.findCalibrationBoard(GrayImg, thread)
#         roi = np.array(CalibrationBoardContour, np.int32)
#         cv2.fillConvexPoly(board, roi, 0)
#         board = np.array(board, dtype = np.uint8)
#         sobelimg = self.getGradientImg(image)
#         # cv2.imshow('board', board)
#         # cv2.imshow('sobelimg', sobelimg)
#         sobelimg = cv2.bilateralFilter(sobelimg, 7, 75, 75)
#         sobelimg = cv2.add(board,sobelimg)
#         mode = util.MODE_LIGHT
#
#         circles = detect.detect_circles(sobelimg, width, height, sigma, low, high, mode, num_quadrant, circle_quadrant)
#         ellipse_datax = []
#         ellipse_datay = []
#         ellipse_data = np.zeros((len(circles), 2))
#         if len(circles) != self.PatternSize_hw[0] * self.PatternSize_hw[1]:
#             return False, None
#         for circle in circles:
#             data = fitellipse.opencv_fit_ellipse(circle)
#             ellipse_datax.append(data[0])
#             ellipse_datay.append(data[1])
#         ellipse_data[:,0] = ellipse_datax
#         ellipse_data[:,1] = ellipse_datay
#         Flag, SortedCirclePts = self.findSortedCirclePoint(calibrationBoardContour=CalibrationBoardContour, circlePts=ellipse_data)
#         return Flag, SortedCirclePts
#
#
#     def findImgPts(self, img, subPix=True, filter='Bilateral', thresh=(0, 255, 15), waitTime_s=1):
#         sigma = 1.51  # The default values can be found in util
#         low = 2.05
#         high = 6
#         num_quadrant = 50
#         circle_quadrant = 20
#         patternSize = self.PatternSize_hw
#         height, width = img.shape[:2]
#         board = np.zeros((height, width))
#         board[:] = 255
#         # Flag, CalibrationBoardContour, CircleContours, CirclePts_nx2 = self.findCalibrationBoard(img=img, thresh=90)
#
#         print 'finding...'
#         GrayImg = HelperFunction.convertImg_BGR2Gray(img)
#         # CalibrationBoardContour = None
#         # CirclePts_nx2= None
#         # sortSucceed = None
#         # SortedCirclePts = None
#         FoundCB = False
#         FirstThresh = None
#         LastThresh = None
#         for Thresh in range(thresh[0], thresh[1], thresh[2]):
#             FoundBoard, CalibrationBoardContour, CircleContours, CirclePts_nx2 = self.findCalibrationBoard(GrayImg, Thresh)
#             if FoundBoard:
#                 if FirstThresh is None:
#                     FirstThresh = Thresh
#                 LastThresh = Thresh
#                 FoundCB = True
#
#         if FoundCB:
#             image = GrayImg
#             Thresh = (FirstThresh + LastThresh) / 2
#             Flag, CalibrationBoardContour, CircleContours, CirclePts_nx2 = self.findCalibrationBoard(GrayImg, Thresh)
#             roi = np.array(CalibrationBoardContour, np.int32)
#             cv2.fillConvexPoly(board, roi, 0)
#             board = np.array(board, dtype = np.uint8)
#             sobelimg = self.getGradientImg(image)
#             # cv2.imshow('board', board)
#             # cv2.imshow('sobelimg', sobelimg)
#             sobelimg = cv2.bilateralFilter(sobelimg, 7, 75, 75)
#             sobelimg = cv2.add(board, sobelimg)
#             mode = util.MODE_LIGHT
#
#             circles = detect.detect_circles(sobelimg, width, height, sigma, low, high, mode, num_quadrant, circle_quadrant)
#             if len(circles) != self.PatternSize_hw[0] * self.PatternSize_hw[1]:
#                 return False, None
#             ellipse_datax = []
#             ellipse_datay = []
#             ellipse_data = np.zeros((len(circles), 2))
#
#             for circle in circles:
#                 data = fitellipse.opencv_fit_ellipse(circle)
#                 ellipse_datax.append(data[0])
#                 ellipse_datay.append(data[1])
#             ellipse_data[:,0] = ellipse_datax
#             ellipse_data[:,1] = ellipse_datay
#             # Flag, SortedCirclePts = self.findSortedCirclePoint(calibrationBoardContour=CalibrationBoardContour, circlePts=ellipse_data)
#             # return Flag, SortedCirclePts
#             extracted, OriginPoint, OriginXObjAxis, OriginYObjAxis, LastPoint = \
#                         self.extractKeyPoints(calibrationBoardContour=CalibrationBoardContour, circlePts_nx2=ellipse_data)
#             if not extracted:
#                 return False, None
#             HomoImg, HomoMatrix, AllPts, Distance_pix = \
#                 self.getHomoImg_4Pts(image, origin=OriginPoint, xAxis=OriginXObjAxis, yAxis=OriginYObjAxis, last=LastPoint)
#             # cv2.imshow('HomoImg', HomoImg)
#             # cv2.waitKey()
#             Flag, HomoSortedPts_nx2 = self.__findPts(HomoImg, thread=Thresh)
#             if Flag:
#                 HomoSortedPts_2xn = HomoSortedPts_nx2.T.reshape(2, -1)
#                 OriginSortedPts_2xn = VGL.projectPts(pts=HomoSortedPts_2xn, projectMatrix=np.linalg.inv(HomoMatrix))
#                 OriginSortedPts_nx2 = OriginSortedPts_2xn.T.reshape(-1, 2)
#                 if waitTime_s >= 0:
#                     # cv2.drawContours(img, [CalibrationBoardContour], 0, (0,255,0))
#                     self.drawImagePts(img, OriginSortedPts_nx2, True)
#                     cv2.namedWindow('result', cv2.WINDOW_NORMAL)
#                     cv2.imshow('result', img)
#                     self.drawImagePts(HomoImg, HomoSortedPts_nx2, True)
#                     cv2.namedWindow('Homo', cv2.WINDOW_NORMAL)
#                     cv2.imshow('Homo', HomoImg)
#                     Key = chr(cv2.waitKey(int(waitTime_s*1000)) & 255)
#                     if 'q' == Key:
#                         raise KeyboardInterrupt, 'Pressed [q] to quit!'
#                 return True, OriginSortedPts_nx2
#         return False, None
#

if __name__ == '__main__':
    PatternSize_hw = (3, 3)
    # PatternSize_hw = (9, 6)
    # Resolution_mm = 22.271714922
    Resolution_mm = 6

    # ImgDir = "../../Datas/HalconBoard/Simulation/bmps"
    # ImgDir = "../../Datas/HalconBoard/PXM20160714_2/"
    ImgDir = "../../Datas/HalconBoard/EyeOnHand/webcam/"
    # ImgDir = "../../Datas/ChessBoard/OpenCVData/"
    MyCollector = HalconBoardInfoCollector(patternSize_hw=PatternSize_hw, resolution_mm=Resolution_mm)
    # Img, Pts = MyCollector.getBoardPicture(blockDisXY_pixel=(100, 100))
    scale = 10
    Img, Pts = MyCollector.getBoardPicture(blockDisXY_pixel=(71*scale, 71*scale), outerBoardXY=(53*scale, 53*scale), outerBoardThickness=242, triangularDis=1.8)
    np.savetxt('ObjPts.txt', MyCollector.ObjPts_nx3)
    print MyCollector.ObjPts_nx3
    # IsFound, FoundImgPts = MyCollector.findImgPts(Img)
    cv2.imshow('HB', Img)
    cv2.imwrite('hb.png', Img)
    print Img.shape
    exit(-1)

    from ToolBox import NewVisionGeometryLib as VGL
    A = np.eye(3)
    A_inverse = np.linalg.inv(A)
    R = VGL.getRy(0.0003)
    print R
    # r = cv2.Rodrigues(R)
    t = np.array([0, 0, 1]).reshape(3, 1)
    Rt = np.column_stack((R, np.array(t)))
    rt_hat = np.column_stack((Rt[:,0],Rt[:,1],Rt[:,3]))
    rt_hat_inverse = np.linalg.inv(rt_hat)
    # rt_front = [[0, 1, -100],
    #             [1, 0, -60],
    #             [0, 0, t[2, 0]]]
    rt_front = [[1, 0, 0], # depends on the patten size
                [0, 1, 0],
                [0, 0, 1]]
    H = np.dot(A, np.dot(rt_front, np.dot(rt_hat_inverse, A_inverse)))
    HomoImg = cv2.warpPerspective(Img, H, (Img.shape[1], Img.shape[0]))
    cv2.imshow('HomoImg', HomoImg)
    cv2.waitKey()
    # MyCollector.drawImagePts(Img, imgPts_nx2=FoundImgPts, isFound=IsFound)
    # cv2.imshow('FoundHB', Img)
    # cv2.imwrite('CB_7x7_pixel100x100.png', Img)
    # np.savetxt('ImgPts.txt', Pts)
    # exit(-1)
    # cv2.waitKey(0)
    # print 'Halcon GT delta:\n', FoundImgPts - Pts

    # PatternSize_hw = (7, 5)
    # MyCollector = ChessboardInfoCollector(patternSize_hw=PatternSize_hw, resolution_mm=Resolution_mm)
    # # Img, Pts = MyCollector.getBoardPicture(blockDisXY_pixel=(100, 100))
    # Img, Pts = MyCollector.getBoardPicture(blockDisXY_pixel=(100, 100), outerBoardXY=(30, 30))
    # IsFound, FoundPts = MyCollector.findImgPts(Img)
    # MyCollector.drawImagePts(Img, FoundPts, IsFound)
    # # IsFound, FoundImgPts = MyCollector_CB.findImgPts(Img)
    # cv2.imshow('CB', Img)
    # print Pts
    # print FoundPts
    # print FoundPts - Pts
    # cv2.waitKey()
    # # MyCollector_CB .drawImagePts(Img, imgPts_nx2=FoundImgPts, isFound=IsFound)
    # # cv2.imshow('Foundq', Img)
    # # cv2.waitKey()
    # exit(-1)
    # MyCollector = ChessboardInfoCollector(patternSize_hw=PatternSize_hw, resolution_mm=Resolution_mm)

    ImgFileList = HelperFunction.pickupImgFiles(ImgDir)
    print ImgFileList
    for imgFile in ImgFileList:
        ImgFilePath = ImgDir + imgFile
        print ImgFilePath
        Img = cv2.imread(ImgFilePath)
        IsFound, ImgPts_nx2 = MyCollector.findImgPts(Img)
        if IsFound:
            MyCollector.addImgPts(ImgPts_nx2)
            MyCollector.drawImagePts(Img, ImgPts_nx2, IsFound)
        cv2.imshow('Img', Img)
        Key = chr(cv2.waitKey() & 255)
        if Key == 'q':
            break
