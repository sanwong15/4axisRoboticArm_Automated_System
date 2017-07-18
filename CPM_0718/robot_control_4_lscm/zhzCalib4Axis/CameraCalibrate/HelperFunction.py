#!/usr/bin/python2.7
# -*- coding:utf-8 -*-
__author__ = 'lh'
__version__ = '1.0'
__date__ = '22/08/2015'


import os
import sys
__CurrentPath = os.path.dirname(__file__) or '.'
sys.path.insert(0, os.path.abspath(os.path.join(__CurrentPath, os.path.pardir)))

import re
import cv2
import argparse
import numpy as np

from ToolBox.VisionGeometryLib import VisionGeometryLib as VGL

def judgeBoolValue(value, judgeReason=('True', 'False')):
    if isinstance(value, bool):
        return value
    if not isinstance(value, str):
        raise ValueError, 'value must be string!'
    Value = value.lower()
    JudgeReason = list(judgeReason)
    if len(JudgeReason) != 2:
        raise ValueError, 'judgeReason error!'
    for idx, judge in enumerate(JudgeReason):
        JudgeReason[idx] = judge.lower()
    if Value == JudgeReason[0]:
        return True
    elif Value == JudgeReason[1]:
        return False
    else:
        raise ValueError, 'value not in judge reason!'

def getInputOutputPath(description=''):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("Input", type=str,
                        help="Input file path")
    parser.add_argument("Output", type=str,
                        help="Output file path")
    if 1 == len(sys.argv):
        return None, None
    args = parser.parse_args()
    InputPath = args.Input
    OutputPath = args.Output
    return InputPath, OutputPath

def getInputPath(description=''):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("Input", type=str,
                        help="Input file path")
    if 1 == len(sys.argv):
        return None
    args = parser.parse_args()
    InputPath = args.Input
    return InputPath

def format_CV2VGL(pts_nx1xd_nxd):
    if pts_nx1xd_nxd.ndim != 2:
        if 3 == pts_nx1xd_nxd.ndim and 1 == pts_nx1xd_nxd.shape[1]:
            n, _, dim = pts_nx1xd_nxd.shape
        else:
            raise ValueError, 'pts_nx1xd_nxd.ndim error'
    else:
        n, dim = pts_nx1xd_nxd.shape
    return pts_nx1xd_nxd.T.reshape(dim, -1)

def format_VGL2CV(pts_VGL_dxn, ndim=2):
    if pts_VGL_dxn.ndim != 2:
        raise ValueError, 'pts_dimxn.ndim != 2'
    dim, n = pts_VGL_dxn.shape
    if ndim == 3:
        return pts_VGL_dxn.T.reshape(-1, 1, dim)
    elif ndim == 2:
        return pts_VGL_dxn.T.reshape(-1, dim)
    else:
        raise ValueError, 'ndim must be [2] or [3]'

def convertImg_BGR2Gray(img):
    if img.ndim == 3:
        GrayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.ndim == 2:
        GrayImg = img.copy()
    else:
        raise ValueError, 'Img data error'
    return GrayImg

def calcDistancePoint2Line(srcPt, linePt1, linePt2):
    Vec1 = np.array(srcPt) - np.array(linePt1)
    Vec2 = np.array(linePt1) - np.array(linePt2)
    Area = abs(np.cross(Vec1, Vec2))
    Len = np.linalg.norm(Vec2)
    Dis = Area / Len
    return Dis

def findNumber(str, numMatch=r"(\d*)\."):
    return int(re.findall(numMatch, str)[0])

def pickupImgFiles(filePath, pickFileType=['.png', '.jpeg', '.bmp', '.jpg', '.gif'], numMatch=r"(\d*)\."):
# def pickupImgFiles(filePath, pickFileType=['.png', '.jpeg', '.bmp', '.jpg', '.gif'], numMatch=r"\d._"):
    AbsPath =  os.path.abspath(filePath)
    FileList = os.listdir(AbsPath)
    pickFileName = []
    for name in FileList:
        for type in pickFileType:
            if type in name:
                try:
                    findNumber(name, numMatch)
                    pickFileName.append(name)
                except:
                    pass

    pickFileName = sorted(pickFileName, cmp=lambda a,b: cmp(findNumber(a, numMatch), findNumber(b, numMatch)))
    return pickFileName

def transPoses2Ts(pose_nx6):
    Ttr = None
    for i in xrange(pose_nx6.shape[0]):
        Pose = pose_nx6[i, :]
        if Ttr is None:
            Ttr = VGL.Pose2T(Pose)
        else:
            Ttr = np.vstack((Ttr, VGL.Pose2T(Pose)))
    return Ttr

def transTs2Poses(T_4nx4):
    T_4nx4 = np.array(T_4nx4)
    PosesList = []
    T_nx4x4 = np.vsplit(T_4nx4, T_4nx4.shape[0]/4)
    for T in T_nx4x4:
        pose = VGL.T2Pose(T).ravel()
        PosesList.append(pose)
    PoseArray_6xn = np.hstack([pose.reshape(6, 1) for pose in PosesList])
    return PoseArray_6xn

def rtVecs2Tocs(rVecs, tVecs):
    Tocs = None
    for i in xrange(len(rVecs)):
        R = cv2.Rodrigues(rVecs[i])[0]
        t = tVecs[i]
        if Tocs is None:
            Tocs = VGL.Rt2T(R, t)
        else:
            Tocs = np.vstack((Tocs, VGL.Rt2T(R, t)))
    return Tocs

def transTs2TInvs(T4nx4):
    TListOf4x4 = np.vsplit(T4nx4, T4nx4.shape[0] / 4)
    TInvListOf4x4 = [np.linalg.inv(T) for T in TListOf4x4]
    TInv_4nx4 = np.array(TInvListOf4x4).reshape(-1, 4)
    return TInv_4nx4


if __name__ == '__main__':
    import Src.ToolBox.FileInterfaceTool as FIT
    ImgPath = '../../Datas/HalconBoard/Simulation/'
    SavedPath = '../../Datas/HalconBoard/Simulation/bmps/'
    FIT.createFile(SavedPath)
    ImgNames = pickupImgFiles(ImgPath)
    for idx, imgName in enumerate(ImgNames):
        ImgFile = ImgPath + imgName
        Img = cv2.imread(ImgFile)
        cv2.imshow('Img', Img)
        cv2.imwrite(SavedPath+str(idx)+'.bmp', Img)
        Key = chr(cv2.waitKey(10) & 255)
        if 'q' == Key:
            break
