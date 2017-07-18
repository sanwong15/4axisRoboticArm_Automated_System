#!/usr/bin/python2.7
# -*- coding:utf-8 -*-
__author__ = 'lh'
__version__ = '1.0'
__date__ = '06/07/2016'


import numpy as np
import cv2
import math


class Quaternion(object):
    @classmethod
    def R2Quaternion(cls, R_3x3):
        RotVec = cv2.Rodrigues(R_3x3)[0]
        theta = np.linalg.norm(RotVec)
        RotVec = RotVec / np.linalg.norm(RotVec)
        # print RotVec
        # print theta
        Quaternion = np.zeros((4,1))
        Quaternion[0,0] = math.cos(theta/2)
        Quaternion[1:4,0] = (math.sin(theta/2)*RotVec).reshape(3)
        return Quaternion

    @classmethod
    def T2Quaternion2(cls, T):
        isinstance(T, np.ndarray), 'T must be ndarray'
        assert T.shape == (4,4)
        Quaternion = cls.R2Quaternion(T[:3, :3])
        # RotVec = cv2.Rodrigues(T[0:3, 0:3])[0]
        # theta = np.linalg.norm(RotVec)
        # RotVec = RotVec / np.linalg.norm(RotVec)
        # print RotVec
        # print theta
        # Quaternion = np.zeros((4,1))
        # Quaternion[0,0] = math.cos(theta/2)
        # Quaternion[1:4,0] = (math.sin(theta/2)*RotVec).reshape(3)
        return Quaternion
        # ------------------------------------------

    @classmethod
    def T2Quaternion(cls, T):
        Quaternion = np.zeros((4,1), np.float128)
        r4 = (math.sqrt(T[0,0]+T[1,1]+T[2,2]+1))*0.5
        # print 'r4:', r4
        if r4 != 0:
            # r = 1.0 / (4*r4) * np.array([T[2,1]-T[1,2], T[0,2]-T[2,0], T[1,0]-T[0,1]], np.float128).T
            r = 0.25 * np.array([T[2,1]-T[1,2], T[0,2]-T[2,0], T[1,0]-T[0,1]], np.float128).T / r4
        else:
            RotVec = cv2.Rodrigues(T[0:3, 0:3])[0]
            RotVec = RotVec / np.linalg.norm(RotVec)
            r = RotVec.reshape(3)
            # !!!!! wrong
            # a = 0.5 * (T[0:3, 0:3] + 1)
            # for i in range(3):
            #     if (a[:,i]).all:
            #         break
            # r = a[:,i] / np.linalg.norm(a[:,i])

        Quaternion[0, 0] = r4
        Quaternion[1:4, 0] = r
        return Quaternion

    @classmethod
    def Quaternion2T(cls, quaternion):
        quaternion_4x1 = np.array(quaternion).reshape(4, 1)
        a = quaternion_4x1[0,0]
        b = quaternion_4x1[1,0]
        c = quaternion_4x1[2,0]
        d = quaternion_4x1[3,0]
        R = np.array([[a*a+b*b-c*c-d*d,     2*b*c-2*a*d,     2*b*d+2*a*c],
                      [    2*b*c+2*a*d, a*a-b*b+c*c-d*d,     2*c*d-2*a*b],
                      [    2*b*d-2*a*c,     2*c*d+2*a*b, a*a-b*b-c*c+d*d]])
        return R

    @classmethod
    def Pose2Quaternion(cls, pose):
        Pose_array = np.array(pose).reshape(3)
        assert Pose_array.size == 3
        Quaternion = np.zeros((4,1))
        Quaternion[1:4, 0] = Pose_array
        return Quaternion

    @classmethod
    def Quaternion2Pose(cls, quaternion):
        pose = quaternion[1:4,0]
        return pose.reshape(3,1)

    @classmethod
    def QuaternionConj(cls, quaternion):
        """
        unit quaternion's inv is its conj
        :param quaternion:
        :return:
        """
        QuaternionConj = np.copy(quaternion)
        QuaternionConj[1:4] = -quaternion[1:4]
        return QuaternionConj

    @classmethod
    def QuaternionProduct(cls, quaternionA, quaternionB):
        s1 = quaternionA[0]
        q1 = np.float128(quaternionA[1:4]).reshape(3)
        s2 = quaternionB[0]
        q2 = np.float128(quaternionB[1:4]).reshape(3)
        QuaternionResult = np.zeros((4,1))
        QuaternionResult[0,0] = s1*s2 - sum(q1*q2)
        QuaternionResult[1:4, 0] = s1*q2 + s2*q1 + np.cross(q1, q2)
        # a = s1
        # b = q1[0]
        # c = q1[1]
        # d = q1[2]
        # t = s2
        # x = q2[0]
        # y = q2[1]
        # z = q2[2]
        # QuaternionResult[1,0] = b*t + a*x + c*z - d*y
        # QuaternionResult[2,0] = c*t + a*y + d*x - b*z
        # QuaternionResult[3,0] = d*t + a*z + b*y - c*x
        # print QuaternionResult

        return QuaternionResult

    @classmethod
    def rotateByQuaternion(cls, quaternion, pointOrVector):
        if pointOrVector.size == 3:
            PointQuaternion = cls.Pose2Quaternion(pointOrVector)
        elif pointOrVector.size == 4:
            PointQuaternion = pointOrVector
        else:
            raise ValueError, 'pointOrVector must have 3 element or A quaternion!'
        QuaternionConj = cls.QuaternionConj(quaternion)
        rotPointQuaternion = cls.QuaternionProduct( cls.QuaternionProduct(quaternion, PointQuaternion),
                                                    QuaternionConj )
        return rotPointQuaternion


if __name__ == '__main__':
    from VisionGeometryLib import VisionGeometryLib as VGL
    # Q = [-1.01626543091e-06, -0.715943704579, -1.58704303643e-06, 0.69815801354]
    # Q = [-3.1182393982e-06, -0.674988400215, -4.33318884296e-06, 0.737828340162]
    Q = [-0.474377280056, 0.374704062424, -0.636015368443, -0.479632685369]
    R = Quaternion.Quaternion2T(Q)
    print 'R\n', R
    t = np.array([0, 0, 0]).reshape(3)
    print VGL.Rt2T(R, t)
    Pose = VGL.T2pose(VGL.Rt2T(R, t)) - np.array([0, 0, 0, 180, 90, 180])
    print 'pppppppp:', VGL.T2pose(VGL.Rt2T(R, t))
    for i in xrange(len(Pose)):
        if Pose[i] > 180:
            Pose[i] = 360 - Pose[i]
        if Pose[i] < -180:
            Pose[i] = 360 + Pose[i]
    print 'Pose:', Pose


    # t = np.random.random((3,1))
    Point = np.random.random((3,1))
    PointHomo = VGL.Homo(Point)
    print 'PointHomo:\n', PointHomo.T
    t = np.zeros((3,1))
    r = np.random.random((3,1))
    # r = np.array([math.pi, 0, 0]).reshape(3,1)
    # print 'r norm', np.linalg.norm(r) / (math.pi)
    R = cv2.Rodrigues(r)[0]
    T = np.zeros((4,4))
    T[0:3, 0:3] = R
    T[0:3, 3] = t.T
    T[3, 3] = 1
    q = Quaternion.T2Quaternion(T)
    print 'Quaternion:\n', q.T
    newT = Quaternion.Quaternion2T(q)
    print 'T:\n', T
    print 'newT:\n', newT
    print 'rotBynewQ:\n', np.dot(newT, Point).T

    rotMatrixPoint = np.dot(T, PointHomo)
    print 'RotMatrixPoint:\n', rotMatrixPoint.T

    rotQPointQuaternion = Quaternion.rotateByQuaternion(q, Point)
    print rotQPointQuaternion
    rotQPointVec = Quaternion.Quaternion2Pose(rotQPointQuaternion)
    print 'rotQPointVec:\n', rotQPointVec.T
    print 'err: ', rotQPointVec.T - rotMatrixPoint[0:3, 0]
    # print q