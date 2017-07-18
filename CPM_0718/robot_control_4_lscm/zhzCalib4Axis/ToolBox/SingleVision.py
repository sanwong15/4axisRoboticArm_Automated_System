#!/usr/bin/python2.7
# -*- coding:utf-8 -*-
__author__ = 'lh'
__version__ = '1.0'
__date__ = '22/08/2015'


import cv2
import numpy as np

from VisionGeometryLib import VisionGeometryLib as VGL


class SingleVision(object):
    def __init__(self, cameraMatrix, distCoeffs, imgSize=(), resolution=None):
        """
        :param cameraMatrix: intrinsic matrix
        :param distCoeffs: distortion coeffs
        :param imgSize: image size (col x row)
        :param resolution:
        :return:
        """
        object.__init__(self)
        self.__CameraMatrix = cameraMatrix
        self.__DistCoeffs = distCoeffs
        self.__ImgSize = imgSize
        self.__Resolution = resolution

    def get3DPts(self, imgPts_2xn, z_mm, unDistortFlag=True):
        assert imgPts_2xn.ndim == 2,        "imgPts must be 2xn"
        assert imgPts_2xn.shape[0] == 2,    "imgPts must be 2xn"
        DistCoeffDic = {True : self.__DistCoeffs,
                        False: ()}
        UnDistortPts_2xn, UnDistortRay_2xn = VGL.unDistortPts(imgPts_2xn=imgPts_2xn,
                                                              cameraMatrix=self.__CameraMatrix,
                                                              distCoeffs=DistCoeffDic[unDistortFlag])
        Pts3D_3xn = VGL.Homo(UnDistortRay_2xn) * z_mm
        return Pts3D_3xn

    def projectPts2Img(self, pts_3xn, distortFlag=True):
        assert pts_3xn.ndim == 2,        "pts must be 3xn"
        assert pts_3xn.shape[0] == 3,    "pts must be 3xn"
        DistCoeffDic = {True : self.__DistCoeffs,
                        False: ()}
        ImgPts_2xn = VGL.projectPtsToImg(pts_3xn=pts_3xn, Tx2Cam=np.eye(4),
                                         cameraMatrix=self.__CameraMatrix,
                                         distCoeffs=DistCoeffDic[distortFlag])
        return ImgPts_2xn

    def unDistort(self, img):
        UnDistortImg = VGL.unDistort(img=img, cameraMatrix=self.__CameraMatrix, distCoeffs=self.__DistCoeffs)
        return UnDistortImg

    def unDistortPts(self, imgPts_2xn):
        assert imgPts_2xn.ndim == 2,        "imgPts must be 2xn"
        assert imgPts_2xn.shape[0] == 2,    "imgPts must be 2xn"
        UnDistortPts_2xn, UnDistortRay_2xn = VGL.unDistortPts(imgPts_2xn=imgPts_2xn,
                                                              cameraMatrix=self.__CameraMatrix,
                                                              distCoeffs=self.__DistCoeffs)
        return UnDistortPts_2xn, UnDistortRay_2xn


if __name__ == '__main__':
    ImgPts_2xn = np.array([[6.366322275682084637e+02, 7.899394910272621928e+02],
                           [7.416458478168997317e+02, 6.542366444992233028e+02]], dtype=np.float32)
    GroundTruth3DPts_3xn = np.array([[-13.52018097, -10.11538314],
                                     [ -5.64607480,  -7.66223813],
                                     [ 86.82860645,  87.40814098]])
    CameraMatrix = np.array([[  3.86200788e+03,   0.00000000e+00,   1.23487131e+03],
                             [  0.00000000e+00,   3.86121201e+03,   9.90865811e+02],
                             [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])
    DistCoeff = np.array([-7.34537009e-02,   3.57639028e+01,  -8.57163220e-05,  -1.95640238e-05,
                           1.85238425e+02,   8.06021246e-02,   3.53983086e+01,   1.93034174e+02])

    MySingleVision = SingleVision(cameraMatrix=CameraMatrix, distCoeffs=DistCoeff, imgSize=(2592, 1944))

    Pts3D_3xn = MySingleVision.get3DPts(ImgPts_2xn, z_mm=GroundTruth3DPts_3xn[2,1], unDistortFlag=False)
    unDistortPts3D_3xn = MySingleVision.get3DPts(ImgPts_2xn, z_mm=GroundTruth3DPts_3xn[2,1], unDistortFlag=True)
    print '================ get3DPts ================'
    print 'Pts3D_3xn:\n', Pts3D_3xn
    print 'unDistortPts3D_3xn:\n', unDistortPts3D_3xn
    print '***GroundTruthPts3D_3xn***\n', GroundTruth3DPts_3xn

    print '================ projectPts2Img ================'
    distortProjectImgPts_2xn = MySingleVision.projectPts2Img(pts_3xn=unDistortPts3D_3xn, distortFlag=True)
    projectImgPts_2xn = MySingleVision.projectPts2Img(pts_3xn=unDistortPts3D_3xn, distortFlag=False)
    print 'distortProjectImgPts_2xn:\n', distortProjectImgPts_2xn
    print 'projectImgPts_2xn:\n', projectImgPts_2xn
    print '***GroundTruthImgPts_2xn***\n', ImgPts_2xn

    print '================ unDistortPts ================'
    UnDisTortPts, UnDisTortRay = MySingleVision.unDistortPts(imgPts_2xn=distortProjectImgPts_2xn)
    print 'UnDisTortPts:\n', UnDisTortPts
    print 'UnDisTortRay:\n', UnDisTortRay
    print '***GroundTruthImgPts_2xn***\n', projectImgPts_2xn

    SrcImg = cv2.imread('./Data/cam4.png')
    unDistortImg = MySingleVision.unDistort(img=SrcImg)
    cv2.imshow('unDistortImg', cv2.resize(unDistortImg, (SrcImg.shape[1]/5, SrcImg.shape[0]/5)))
    cv2.imshow('SrcImg', cv2.resize(SrcImg, (SrcImg.shape[1]/5, SrcImg.shape[0]/5)))
    cv2.imshow('distort', cv2.resize(unDistortImg-SrcImg, (SrcImg.shape[1]/5, SrcImg.shape[0]/5)))
    cv2.waitKey()