    #!/usr/bin/python2.7
# -*- coding:utf-8 -*-
__author__ = 'lh'
__version__ = '1.0'
__date__ = '08/03/2015'


import cv2
import copy
import numpy as np
import numpy.random
import random

from VisionGeometryLib import VisionGeometryLib as VGL


def addNoisePts(point1_2xn, noise=(0, 0), normalOrMax=False):
    randomAngle_deg = numpy.random.uniform(low=0, high=360, size=point1_2xn.shape[1])
    cosValue = np.cos(randomAngle_deg)
    sinValue = np.sin(randomAngle_deg)
    # randomAngle_deg = randomAngle_deg.repeat(randomAngle_deg.shape, axis=1)
    randomAngle_deg = np.vstack((cosValue, sinValue))
    if normalOrMax:
        Noisy = numpy.random.normal(loc=noise[0], scale=noise[1], size=point1_2xn.shape[1])
        AddNoisy = np.repeat(Noisy.reshape(1, -1), 2, 0)
        noisePts1_2xn = point1_2xn + AddNoisy * randomAngle_deg
    else:
        noisePts1_2xn = point1_2xn + (noise[0] + noise[1]) * randomAngle_deg
    return noisePts1_2xn

def addNoiseMatrix_abs(matrix, noiseParam=(0,0), noiseNormalOrUniform=True):
    newMatrix = copy.deepcopy(matrix)
    newPose = VGL.T2Pose(newMatrix)
    if noiseNormalOrUniform:
        sign = numpy.random.uniform(low=-1, high=1, size=newPose.shape)
        sign[sign>=0] = 1
        sign[sign<0] = -1
        if noiseParam[1] == 0:
            newPose = newPose + sign*noiseParam[0]
        else:
            newPose = newPose + sign*numpy.random.normal(loc=noiseParam[0], scale=noiseParam[1], size=newPose.shape)
    else:
        newPose = newPose + numpy.random.uniform(low=noiseParam[0], high=noiseParam[1], size=newPose.shape)
    return VGL.Pose2T(newPose)

def addNoisePoses(pose_6xn, x=(0, 0), y=(0, 0), z=(0, 0), u=(0, 0), v=(0, 0), w=(0, 0)):
    Pose_6xn = pose_6xn.copy()
    PoseSplit = np.vsplit(Pose_6xn, Pose_6xn.shape[0])
    NoiseParam = [x, y, z, u, v, w]
    NewPose = []
    for axis, param in zip(PoseSplit, NoiseParam):
        sign = numpy.random.uniform(low=-1, high=1, size=axis.shape)
        sign[sign>=0] = 1
        sign[sign<0] = -1
        if param[1] == 0:
            axis = axis + sign*param[0]
        else:
            axis = axis + sign*numpy.random.normal(loc=param[0], scale=param[1], size=axis.shape)
        NewPose.append(axis)
    NewPoseArray_6xn = np.vstack([axis.reshape(1, -1) for axis in NewPose])
    return NewPoseArray_6xn


def addNoiseMatrix_percent(matrix, noiseParam=(0,0), noiseNormalOrUniform=True):
    newMatrix = copy.deepcopy(matrix)
    newPose = VGL.T2Pose(newMatrix)
    if noiseNormalOrUniform:
        sign = numpy.random.uniform(low=-1, high=1, size=newPose.shape)
        sign[sign>=0] = 1
        sign[sign<0] = -1
        if noiseParam[1] == 0:
            newPose = newPose + sign*noiseParam[0] * newPose
        else:
            newPose = newPose + newPose * sign * numpy.random.normal(loc=noiseParam[0], scale=noiseParam[1], size=newPose.shape)
    else:
        newPose = newPose + numpy.random.uniform(low=noiseParam[0], high=noiseParam[1], size=newPose.shape) * newPose
    return VGL.Pose2T(newPose)

def addNoiseMatrix2(matrix, noisePercent=0.01, noiseRandomOrMax=False):
    newMatrix = copy.deepcopy(matrix)
    if newMatrix.shape == (4,4):
        R, t = VGL.T2RT(newMatrix)
        r = cv2.Rodrigues(R)[0]
        if noiseRandomOrMax is None:
            for i in range(r.size):
                r[i, 0] = r[i, 0] * (1-noisePercent)
            for i in range(t.size):
                t[i] = t[i] * (1-noisePercent)
        elif not noiseRandomOrMax:
            for i in range(r.size):
                if random.random() < 0.5:
                    r[i, 0] = r[i, 0] * (1-noisePercent)
                else:
                    r[i, 0] = r[i, 0] * (1+noisePercent)
            for i in range(t.size):
                if random.random() < 0.5:
                    t[i] = t[i] * (1-noisePercent)
                else:
                    t[i] = t[i] * (1+noisePercent)
        else:
            for i in range(r.size):
                r[i, 0] = r[i, 0] * (1-(random.random()-0.5)*2*noisePercent)
            for i in range(t.size):
                t[i] = t[i] * (1-(random.random()-0.5)*2*noisePercent)
        newMatrix = VGL.Rt2T(cv2.Rodrigues(r)[0], t.reshape(3,1))
    else:
        for i in range(newMatrix.shape[0]-1):
            for j in range(newMatrix.shape[1]):
                if random.random() < 0.5:
                    newMatrix[i, j] = newMatrix[i, j] * (1-noisePercent)
                else:
                    newMatrix[i, j] = newMatrix[i, j] * (1+noisePercent)

    return newMatrix
