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
import numpy as np
import collections

import ToolBox.FileInterfaceTool as FIT
import HelperFunction
from ToolBox.VisionGeometryLib import VisionGeometryLib as VGL
from CalibrationBoard          import HalconBoardInfoCollector, ChessboardInfoCollector, newSubPixel


CalibrateResultStructure = collections.namedtuple('CalibrateResultStructure', ['ReprojErr', 'CameraMatrix', 'DistCoeff', 'rVecs', 'tVecs'])


def CameraCalibrate(objPts_mxnx3, imgPts_mxnx2, imgSize, calDist=True):
    if calDist:
        Flags = cv2.CALIB_RATIONAL_MODEL
    else:
        Flags = cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 \
                | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6
    ReprojErr, CameraMatrix, DistCoeffs, rVecs, tVecs = \
        cv2.calibrateCamera(objectPoints=objPts_mxnx3, imagePoints=imgPts_mxnx2, imageSize=imgSize, flags=Flags)
    return ReprojErr, CameraMatrix, DistCoeffs, rVecs, tVecs

def refineCalibrate(objPts_mxnx3, imgPts_mxnx2, imgSize, rvecs, tvecs, cameraMatrix, distCoeffs, calDist=True, sort=0.8, filterRatio=0.2):
    ObjNum = objPts_mxnx3.shape[0]
    ProjectErrorList = []
    for i in xrange(ObjNum):
        ProjectPts_nx1x2, Jacobian = \
            cv2.projectPoints(objectPoints=np.float32(objPts_mxnx3[i,:,:]).reshape(-1,3),
                              rvec=rvecs[i], tvec=tvecs[i],
                              cameraMatrix=cameraMatrix, distCoeffs=distCoeffs)
        ProjectError_nx1x2 = ProjectPts_nx1x2 - imgPts_mxnx2[i].reshape(-1, 1, 2)
        ProjectError_nx1 = np.linalg.norm(x=ProjectError_nx1x2, axis=2)
        ProjectErrorList.append(ProjectError_nx1)
    ProjectError_mxnx1 = np.array(ProjectErrorList)
    ProjectError_mnx1 = ProjectError_mxnx1.reshape(-1, 1)
    ProjectErrorSortIdx_mnx1 = ProjectError_mnx1.argsort(axis=0)
    ProjectError_mnx1[ProjectErrorSortIdx_mnx1[0:int(ProjectErrorSortIdx_mnx1.size*sort)]] = True
    ProjectError_mnx1[ProjectErrorSortIdx_mnx1[int(ProjectErrorSortIdx_mnx1.size*sort):]] = False
    ProjectError_mnx1 = ProjectError_mnx1 > 0
    GoodPtsMask_mxnx1 = ProjectError_mnx1.reshape(ObjNum, -1, 1)

    GoodCirclePtsList = []
    NewObjPtsList = []
    GoodImgMask_list = []
    for i in xrange(ObjNum):
        if GoodPtsMask_mxnx1[i].mean() >= filterRatio:
            FilterCirclePts_nx2 = np.float32(imgPts_mxnx2[i][GoodPtsMask_mxnx1[i].reshape(-1, )])
            GoodCirclePtsList.append(FilterCirclePts_nx2)
            # print 'FilterCirclePts_nx2: ', FilterCirclePts_nx2.shape
            FilterObjPts_nx2 = np.float32(objPts_mxnx3[i][GoodPtsMask_mxnx1[i].reshape(-1, )])
            NewObjPtsList.append(FilterObjPts_nx2)
            GoodImgMask_list.append(True)
        else:
            GoodImgMask_list.append(False)
            print "Error: ", i

    ReprojErr, CameraMatrix, DistCoeffs, rVecs, tVecs = \
        CameraCalibrate(objPts_mxnx3=np.array(NewObjPtsList),
                        imgPts_mxnx2=np.array(GoodCirclePtsList), imgSize=imgSize, calDist=calDist)
    return ReprojErr, CameraMatrix, DistCoeffs, rVecs, tVecs, GoodPtsMask_mxnx1, GoodImgMask_list

def StereoCalibrate(objPts_mxnx3, imgPtsLeft_mxnx2, imgPtsRight_mxnx2, imageNum, imgSize, cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR, fixIntrinsic=False):
    if fixIntrinsic:
        Flags=cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_FIX_INTRINSIC
    else:
        Flags=cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_USE_INTRINSIC_GUESS
    # Retval, CameraMatrixL, DistCoeffsL, CameraMatrixR, DistCoeffsR, R, T, E, F = \
    #     cv2.stereoCalibrate(objectPoints=np.float32(objPts_mxnx3.reshape(imageNum, -1, 3)),
    #                         imagePoints1=np.float32(imgPtsLeft_mxnx2.reshape(imageNum, -1, 2)),
    #                         imagePoints2=np.float32(imgPtsRight_mxnx2.reshape(imageNum, -1, 2)),
    #                         imageSize=imgSize,
    #                         cameraMatrix1=cameraMatrixL, distCoeffs1=distCoeffL,
    #                         cameraMatrix2=cameraMatrixR, distCoeffs2=distCoeffR,
    #                         flags=Flags)
    ImgPtsLeft_mxnx2 = []
    ImgPtsRight_mxnx2 = []
    ObjPts_mxnx3 = []
    for objPts, imgPtsL, imgPtsR in zip(objPts_mxnx3, imgPtsLeft_mxnx2, imgPtsRight_mxnx2):
        if imgPtsL.shape[0] >= 4 and imgPtsR.shape[0] >= 4:
            ObjPts_mxnx3.append(objPts)
            ImgPtsLeft_mxnx2.append(imgPtsL)
            ImgPtsRight_mxnx2.append(imgPtsR)
    Retval, CameraMatrixL, DistCoeffsL, CameraMatrixR, DistCoeffsR, R, T, E, F = \
        cv2.stereoCalibrate(objectPoints=ObjPts_mxnx3,
                            imagePoints1=ImgPtsLeft_mxnx2,
                            imagePoints2=ImgPtsRight_mxnx2,
                            imageSize=imgSize,
                            cameraMatrix1=cameraMatrixL, distCoeffs1=distCoeffL,
                            cameraMatrix2=cameraMatrixR, distCoeffs2=distCoeffR,
                            flags=Flags)
    return Retval, CameraMatrixL, DistCoeffsL, CameraMatrixR, DistCoeffsR, R, T, E, F

def distortPoints(imgPts_2xn, cameraMatrix, distortCoeffs):
    Center = np.array([cameraMatrix[0, 2], cameraMatrix[1, 2]]).reshape(2, 1)
    FocalLength = np.array([cameraMatrix[0, 0], cameraMatrix[1, 1]]).reshape(2, 1)
    ImgCoordPts_3xn = np.vstack(((imgPts_2xn - Center) / FocalLength, np.zeros((1, imgPts_2xn.shape[1]))))
    DistImgPts_2xn = VGL.projectPtsToImg(ImgCoordPts_3xn, Tx2Cam=np.eye(4), cameraMatrix=cameraMatrix, distCoeffs=distortCoeffs)
    return DistImgPts_2xn

class SingleCamCalibrateAPP(object):
    def __init__(self, imgsPath, patternSize_hw, patternResolution_mm, patternType, savePath,
                 calDist=True, imgSize=None, filter=None, thresh=(0, 255, 15), waitTime=-1, save=True):
        object.__init__(self)
        if 'Halcon' == patternType:
            self.InfoCollector = HalconBoardInfoCollector(patternSize_hw=patternSize_hw, resolution_mm=patternResolution_mm)
        elif 'Chessboard' == patternType:
            self.InfoCollector = ChessboardInfoCollector(patternSize_hw=patternSize_hw, resolution_mm=patternResolution_mm)
        else:
            raise ValueError, "CalibrationBoard::BoardType error, must be 'Halcon' or 'Chessboard'"
        # self.InfoCollector = newSubPixel(patternSize_hw=patternSize_hw, resolution_mm=patternResolution_mm)

        self.__SavePath = FIT.absPath(savePath)
        self.__DetectSavePath = self.__SavePath + '/DetectResult/'
        self.__CalibrateSavePath = self.__SavePath + '/CalibrateResult/'
        FIT.createFile(self.__SavePath)
        FIT.createFile(self.__DetectSavePath)
        FIT.createFile(self.__CalibrateSavePath)

        self.__CalDist = calDist
        self.__ImgPath = imgsPath
        self.__ImgSize = imgSize
        self.__Filter = filter
        self.__Save = save
        self.__WaitTime = waitTime
        self.__Thresh = thresh
        self.__FoundMaskList = []

    @property
    def ImgSize(self):
        return self.__ImgSize

    def collectInfo(self, cameraMatrix=None, distCoeffs=None, homoMatrixList=None):
        print 'Collecting info...'
        self.__FoundMaskList = []
        if FIT.isFile(self.__ImgPath):
            # load data
            ImgPts_mnx2 = np.array(np.loadtxt(self.__ImgPath))
            if 2 != ImgPts_mnx2.shape[1] and 2 == ImgPts_mnx2.shape[0]:
                ImgPts_mnx2 = HelperFunction.format_VGL2CV(ImgPts_mnx2)
            elif 2 == ImgPts_mnx2.shape[1]:
                pass
            else:
                raise ValueError, self.__ImgPath, ' data error'
            h, w = self.InfoCollector.PatternSize_hw
            N = h * w
            M = ImgPts_mnx2.shape[0] / N
            ImgPts_ListofNx2 = np.vsplit(ImgPts_mnx2, M)
            # collect
            for idx, ImgPts_nx2 in enumerate(ImgPts_ListofNx2):
                self.InfoCollector.addImgPts(ImgPts_nx2)
                self.__FoundMaskList.append(True)
            if self.__ImgSize is None:
                self.__ImgSize = self.InfoCollector.AllImgPts_mxnx2.reshape(-1, 2).max(axis=0)
        else:
            # extract info from image

            ImgNames = HelperFunction.pickupImgFiles(self.__ImgPath)
            # ImgNames = HelperFunction.pickupImgFiles(self.__ImgPath, numMatch=r'(\d+)_aver')

            ImgFiles = [FIT.absPath(self.__ImgPath)+'/'+name for name in ImgNames]
            SrcImg = None
            for idx, imgFile in enumerate(ImgFiles):
                SrcImg = cv2.imread(imgFile)
                if distCoeffs is not None and cameraMatrix is not None:
                    SrcImg = VGL.unDistort(SrcImg, cameraMatrix, distCoeffs)

                if isinstance(self.InfoCollector, HalconBoardInfoCollector):
                    Found, ImgPts_nx2 = \
                        self.InfoCollector.findImgPts(SrcImg, filter=self.__Filter,
                                                      thresh=self.__Thresh, waitTime_s=self.__WaitTime)
                else:
                    Found, ImgPts_nx2 = self.InfoCollector.findImgPts(SrcImg)
                if Found:
                    if distCoeffs is not None and cameraMatrix is not None:
                        ImgPts_2xn = distortPoints(ImgPts_nx2.T.reshape(2, -1), cameraMatrix, distCoeffs)
                        ImgPts_nx2 = ImgPts_2xn.T.reshape(-1, 2)
                    self.InfoCollector.addImgPts(ImgPts_nx2)
                self.__FoundMaskList.append(Found)
                print 'find image: ', imgFile, ' - ', Found

                if self.__Save:
                    self.InfoCollector.drawImagePts(SrcImg, ImgPts_nx2, Found)
                    cv2.imwrite(self.__DetectSavePath + 'Show_' + ImgNames[idx], SrcImg)
            # if self.__ImgSize is None:
            self.__ImgSize = SrcImg.shape[:2][::-1]
        self.__ImgSize = tuple(self.__ImgSize)
        ObjPts_mxnx3 = self.InfoCollector.AllObjPts_mxnx3
        ImgPts_mxnx2 = self.InfoCollector.AllImgPts_mxnx2
        self.ImgPts_nx2 = ImgPts_mxnx2.reshape(-1, 2)
        self.ObjPts_nx3 = ObjPts_mxnx3.reshape(-1, 3)
        np.savetxt(self.__DetectSavePath + 'ObjPts.txt', self.ObjPts_nx3)
        np.savetxt(self.__DetectSavePath + 'ImgPts.txt', self.ImgPts_nx2)
        np.savetxt(self.__DetectSavePath + 'ImgSize.txt', self.__ImgSize)
        np.savetxt(self.__DetectSavePath + 'FoundMask.txt', self.__FoundMaskList)
        return ObjPts_mxnx3, ImgPts_mxnx2, self.__ImgSize, self.__FoundMaskList

    def calibrate(self):
        print 'Calibrating...'
        ObjPts = self.InfoCollector.AllObjPts_mxnx3
        ImgPts = self.InfoCollector.AllImgPts_mxnx2
        ReprojErr, CameraMatrix, DistCoeffs, rVecs, tVecs = \
            CameraCalibrate(objPts_mxnx3=ObjPts, imgPts_mxnx2=ImgPts, imgSize=self.__ImgSize, calDist=self.__CalDist)

        GoodNum = self.InfoCollector.ImageNumber
        print 'self.__ImgSize: ', self.__ImgSize
        print "=========before optimize============="
        print "ReprojErr:\n   ", ReprojErr
        print "CameraMatrix:\n", CameraMatrix
        print "DistCoeffs:\n  ", DistCoeffs
        print "GoodNum:       ", GoodNum
        print "======================================"
        Tocs = HelperFunction.rtVecs2Tocs(rVecs, tVecs)
        np.savetxt(FIT.joinPath(self.__CalibrateSavePath, "BeforeOpt_ReprojErr.txt"), [ReprojErr])
        np.savetxt(FIT.joinPath(self.__CalibrateSavePath, "BeforeOpt_CameraMatrix.txt"),CameraMatrix)
        np.savetxt(FIT.joinPath(self.__CalibrateSavePath, "BeforeOpt_DistCoeffs.txt"),DistCoeffs)
        np.savetxt(FIT.joinPath(self.__CalibrateSavePath, "BeforeOpt_Toc.txt"), Tocs)

        print 'Refining...'
        for sortScale in [0.8, 0.9, 0.7]:
            OptReprojErr, OptCameraMatrix, OptDistCoeffs, OptrVecs, OpttVecs, self.GoodPtsMask_mxnx1, self.GoodImgMask_list = \
                refineCalibrate(objPts_mxnx3=ObjPts, imgPts_mxnx2=ImgPts, imgSize=self.__ImgSize,
                                rvecs=rVecs, tvecs=tVecs, cameraMatrix=CameraMatrix, distCoeffs=DistCoeffs, calDist=self.__CalDist, sort=sortScale)
            OptTocs = HelperFunction.rtVecs2Tocs(OptrVecs, OpttVecs)
            print "=========after optimize============="
            print 'sort scale:    ', sortScale
            print "ReprojErr:\n   ", OptReprojErr
            print "CameraMatrix:\n", OptCameraMatrix
            print "DistCoeffs:\n  ", OptDistCoeffs
            print "GoodNum:       ", len(OptrVecs)
            print "======================================"
            if OptReprojErr < ReprojErr:
                np.savetxt(FIT.joinPath(self.__CalibrateSavePath, "ReprojErr.txt"), [OptReprojErr])
                np.savetxt(FIT.joinPath(self.__CalibrateSavePath, "CameraMatrix.txt"), OptCameraMatrix)
                np.savetxt(FIT.joinPath(self.__CalibrateSavePath, "DistCoeffs.txt"), OptDistCoeffs)
                np.savetxt(FIT.joinPath(self.__CalibrateSavePath, "Toc.txt"), OptTocs)
                self.BeforeOptResult = CalibrateResultStructure(ReprojErr, CameraMatrix, DistCoeffs, rVecs, tVecs)
                self.OptResult = CalibrateResultStructure(OptReprojErr, OptCameraMatrix, OptDistCoeffs, OptrVecs, OpttVecs)
                return self.BeforeOptResult, self.OptResult, self.GoodPtsMask_mxnx1, self.GoodImgMask_list
        raise ValueError, 'refinement error!!!'

    def run(self):
        # TempWaitTime = self.__WaitTime
        # self.__WaitTime = 1
        ObjPts_mxnx3, ImgPts_mxnx2, ImgSize, FoundMaskList = self.collectInfo()
        _, OptResult, GoodPtsMask_mxnx1, GoodImgMask_list = self.calibrate()
        self.InfoCollector.clearAllPts()
        # CameraMatrix = OptResult.CameraMatrix
        # DistCoeffs = OptResult.DistCoeff
        # self.__WaitTime = TempWaitTime
        # self.collectInfo(cameraMatrix=CameraMatrix, distCoeffs=DistCoeffs)
        # self.calibrate()
        print 'Single Camera Calibration done, Saved in:',
        print self.__DetectSavePath
        print self.__CalibrateSavePath


class StereoCamCalibrateAPP(object):
    def __init__(self, imgsPathLeft, imgsPathRight, patternSize_hw, patternResolution_mm, patternType, savePath,
                 calDist=True, fixIntrinsic=False, imgSizeLeft=None, imgSizeRight=None, filterLeft=None, filterRight=None, thresh=(0, 255, 15), waitTime=-1, save=True):
        object.__init__(self)
        self.__SavePath = FIT.absPath(savePath)
        self.__CalibrateSavePath = self.__SavePath + '/Stereo/'
        FIT.createFile(self.__SavePath)
        FIT.createFile(self.__CalibrateSavePath)

        self.__FilterLeft = filterLeft
        self.__FilterRight = filterRight
        self.__WaitTime = waitTime
        self.__Thresh = thresh
        self.SingleCamL = \
            SingleCamCalibrateAPP(imgsPathLeft, patternSize_hw, patternResolution_mm,
                                  patternType, savePath+'/Left/', calDist=calDist, imgSize=imgSizeLeft,
                                  filter=self.__FilterLeft, waitTime=self.__WaitTime, thresh=self.__Thresh, save=save)
        self.SingleCamR = \
            SingleCamCalibrateAPP(imgsPathRight, patternSize_hw, patternResolution_mm,
                                  patternType, savePath+'/Right/', calDist=calDist, imgSize=imgSizeRight,
                                  filter=self.__FilterRight, waitTime=self.__WaitTime, thresh=self.__Thresh, save=save)

        self.__CalDist = calDist
        self.__ImgPathLeft = None
        self.__ImgPathRight = None
        self.__ImgSize = None
        self.__FoundMaskList = []
        self.__FixIntrinsic = fixIntrinsic

    def __getMapIdx(self, foundMask):
        MapCal2Find = {}
        MapFind2Cal = {}
        FilterIdx = 0
        for idx, found in enumerate(foundMask):
            if found:
                MapCal2Find[FilterIdx] = idx
                MapFind2Cal[idx] = FilterIdx
                FilterIdx += 1
        return MapCal2Find, MapFind2Cal

    def __getAliveMask(self, foundMask, aliveMask, mapCal2Find):
        AliveMask = foundMask[:]
        for idx, alive in enumerate(aliveMask):
            if not alive:
                AliveMask[mapCal2Find[idx]] = False
        return AliveMask

    def collectInfo(self):
        LObjPts_mxnx3, LImgPts_mxnx2, LImgSize, LFoundMaskList = self.SingleCamL.collectInfo()
        RObjPts_mxnx3, RImgPts_mxnx2, RImgSize, RFoundMaskList = self.SingleCamR.collectInfo()
        if len(LFoundMaskList) != len(RFoundMaskList):
            raise ValueError, 'Left image number is not equal to right!!!'
        # FoundMaskList = LFoundMaskList and RFoundMaskList
        LMapCal2Find, LMapFind2Cal = self.__getMapIdx(LFoundMaskList)
        RMapCal2Find, RMapFind2Cal = self.__getMapIdx(RFoundMaskList)
        print '----------------- Calibrate Left ---------------------'
        LBeforeOpt, LOpt, LGoodPtsMask_mxnx1, LGoodImgMask_list = self.SingleCamL.calibrate()
        LAliveMask = self.__getAliveMask(foundMask=LFoundMaskList, aliveMask=LGoodImgMask_list, mapCal2Find=LMapCal2Find)

        print '----------------- Calibrate Right ---------------------'
        RBeforeOpt, ROpt, RGoodPtsMask_mxnx1, RGoodImgMask_list = self.SingleCamR.calibrate()
        RAliveMask = self.__getAliveMask(foundMask=RFoundMaskList, aliveMask=RGoodImgMask_list, mapCal2Find=RMapCal2Find)

        # GoodPtsMask = LGoodPtsMask_mxnx1 * RGoodPtsMask_mxnx1
        # pick up alive info
        AliveMask = np.array(LAliveMask) * np.array(RAliveMask)
        self.ObjPts_mxnx3 = []
        self.ImgPtsLeft_mxnx2 = []
        self.ImgPtsRight_mxnx2 = []
        for idx, alive in enumerate(AliveMask):
            if alive:
                # self.ObjPts_mxnx3.append(LObjPts_mxnx3[idx, :, :])
                # self.ImgPtsLeft_mxnx2.append(LImgPts_mxnx2[idx, :, :])
                # self.ImgPtsRight_mxnx2.append(RImgPts_mxnx2[idx, :, :])

                # self.ObjPts_mxnx3.append(LObjPts_mxnx3[idx][GoodPtsMask[idx].reshape(-1)])
                # self.ImgPtsLeft_mxnx2.append(LImgPts_mxnx2[idx][GoodPtsMask[idx].reshape(-1)])
                # self.ImgPtsRight_mxnx2.append(RImgPts_mxnx2[idx][GoodPtsMask[idx].reshape(-1)])
                LeftIdx = LMapFind2Cal[idx]
                RightIdx = RMapFind2Cal[idx]
                GoodPtsMask = (LGoodPtsMask_mxnx1[LeftIdx] * RGoodPtsMask_mxnx1[RightIdx]).reshape(-1)
                GoodPtsLeft_gx2 = LImgPts_mxnx2[LeftIdx][GoodPtsMask]
                GoodPtsRight_gx2 = RImgPts_mxnx2[RightIdx][GoodPtsMask]
                self.ObjPts_mxnx3.append(LObjPts_mxnx3[LeftIdx][GoodPtsMask])
                self.ImgPtsLeft_mxnx2.append(LImgPts_mxnx2[LeftIdx][GoodPtsMask])
                self.ImgPtsRight_mxnx2.append(RImgPts_mxnx2[RightIdx][GoodPtsMask])
        # self.ObjPts_mxnx3 = np.float32(self.ObjPts_mxnx3)
        # self.ImgPtsLeft_mxnx2 = np.float32(self.ImgPtsLeft_mxnx2)
        # self.ImgPtsRight_mxnx2 = np.float32(self.ImgPtsRight_mxnx2)

        self.LeftCalibrateResult = LOpt
        self.RightCalibrateResult = ROpt
        # self.ImgNum = self.ImgPtsLeft_mxnx2.shape[0]
        self.ImgNum = len(self.ImgPtsLeft_mxnx2)

    def calibrate(self):
        print '----------------- Calibrate Stereo ---------------------'
        print 'Calibrating...'
        self.__ImgPathLeft = self.SingleCamL.ImgSize
        self.__ImgPathRight = self.SingleCamR.ImgSize
        self.__ImgSize = tuple((np.array(self.__ImgPathLeft) + np.array(self.__ImgPathRight)) / 2)
        print 'img size: ', self.__ImgSize
        Retval, CameraMatrixL, DistCoeffsL, CameraMatrixR, DistCoeffsR, R, t, E, F = \
            StereoCalibrate(objPts_mxnx3=self.ObjPts_mxnx3,
                            imgPtsLeft_mxnx2=self.ImgPtsLeft_mxnx2, imgPtsRight_mxnx2=self.ImgPtsRight_mxnx2,
                            imageNum=self.ImgNum, imgSize=self.__ImgSize,
                            cameraMatrixL=self.LeftCalibrateResult.CameraMatrix,
                            distCoeffL=self.LeftCalibrateResult.DistCoeff,
                            cameraMatrixR=self.RightCalibrateResult.CameraMatrix,
                            distCoeffR=self.RightCalibrateResult.DistCoeff,
                            fixIntrinsic=self.__FixIntrinsic)

        print "Retval:\n    ", Retval
        print "CameraMatrixL:\n", CameraMatrixL
        print "CameraMatrixR:\n", CameraMatrixR
        print "DistCoeffsL:\n", DistCoeffsL
        print "DistCoeffsR:\n", DistCoeffsR
        print "R:\n", R
        print "t:\n", t
        print "E:\n", E
        print "F:\n", F
        Tc1c2 = VGL.Rt2T(R, t)
        np.savetxt(self.__CalibrateSavePath + 'ReProjectErr.txt', [Retval])
        np.savetxt(self.__CalibrateSavePath + 'CameraMatrixL.txt', CameraMatrixL)
        np.savetxt(self.__CalibrateSavePath + 'CameraMatrixR.txt', CameraMatrixR)
        np.savetxt(self.__CalibrateSavePath + 'DistCoeffsL.txt', DistCoeffsL)
        np.savetxt(self.__CalibrateSavePath + 'DistCoeffsR.txt', DistCoeffsR)
        np.savetxt(self.__CalibrateSavePath + 'E.txt', E)
        np.savetxt(self.__CalibrateSavePath + 'F.txt', F)
        np.savetxt(self.__CalibrateSavePath + 'R.txt', R)
        np.savetxt(self.__CalibrateSavePath + 't.txt', t)
        np.savetxt(self.__CalibrateSavePath + 'Tc1c2.txt', Tc1c2)
        print 'Stereo Camera Calibration done, Saved in:',
        print self.__CalibrateSavePath
        return Retval, CameraMatrixL, DistCoeffsL, CameraMatrixR, DistCoeffsR, R, t, E, F

    def run(self):
        self.collectInfo()
        self.calibrate()


if __name__ == '__main__':
    ConfigFilePath = 'StereoCamConfig.yaml'
    ConfigData = FIT.loadYaml(ConfigFilePath)
    ImgPathL = ConfigData['ImagesPath_Left']
    ImgPathR = ConfigData['ImagesPath_Right']
    CalibrationBoard = ConfigData['CalibrationBoard']

    MyApp = \
        StereoCamCalibrateAPP(imgsPathLeft=ImgPathL, imgsPathRight=ImgPathR,
                              patternSize_hw=CalibrationBoard['PatternSize_hw'],
                              patternResolution_mm=CalibrationBoard['Resolution_mm'],
                              patternType=CalibrationBoard['BoardType'], fixIntrinsic=ConfigData['Calibrate']['FixIntrinsic'],
                              savePath=ConfigData['OutputPath'], calDist=ConfigData['Calibrate']['Distortion'], imgSize=ConfigData['Calibrate']['ImgSize'])
    MyApp.run()

# if __name__ == '__main__':
#     Obj_mnx3 = np.loadtxt('../../Datas/ObjPts.txt').astype(np.float32)
#     Img_mnx2 = np.loadtxt('../../Datas/ImgPts.txt').astype(np.float32)
#     N = 7 * 7
#     M = Img_mnx2.shape[0] / N
#     Obj = Obj_mnx3.reshape(M, N, -1)
#     Img = Img_mnx2.reshape(M, N, -1)
#     ImgSize = tuple(np.loadtxt('../../Datas/ImgSize.txt').astype(np.int32).ravel())
#     ReprojErr, CameraMatrix, DistCoeffs, rVecs, tVecs = CameraCalibrate(objPts_mxnx3=Obj, imgPts_mxnx2=Img, imgSize=ImgSize)
#     print 'ReprojErr: ', ReprojErr
#     print 'CameraMatrix: ', CameraMatrix
#     print 'DistCoeffs: ', DistCoeffs
#     print 'rVecs: ', rVecs
#     print 'tVecs: ', tVecs
