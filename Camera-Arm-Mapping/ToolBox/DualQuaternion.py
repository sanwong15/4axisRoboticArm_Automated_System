#!/usr/bin/python2.7
# -*- coding:utf-8 -*-
__author__ = 'lh'
__version__ = '1.0'
__date__ = '06/07/2016'


import numpy as np
import cv2

from Quaternion         import  Quaternion


class DualQuaternion(object):
    @classmethod
    def T2DualQuaternion(cls, T):
        # RealPart = Quaternion.T2Quaternion(T)
        RealPart = Quaternion.T2Quaternion2(T)
        t = T[0:3, 3]
        tQ = Quaternion.Pose2Quaternion(t)
        ImagePart = 0.5 * Quaternion.QuaternionProduct(tQ, RealPart)

        return (RealPart, ImagePart)

    @classmethod
    def CheckNormalized(cls, dualQuaternion):
        Q = dualQuaternion[0]
        QConj = Quaternion.QuaternionConj(Q)
        QImage = dualQuaternion[1]
        QImageConj = Quaternion.QuaternionConj(QImage)
        RealPart = Quaternion.QuaternionProduct(Q, QConj)
        ImagePart = Quaternion.QuaternionProduct(Q, QImageConj) + Quaternion.QuaternionProduct(QImage, QConj)

        if np.linalg.norm(RealPart) == 1 and -1.0e-10 < np.linalg.norm(ImagePart) < 1.0e-10:
            return True
        else:
            print 'RealPart:\n', RealPart
            print 'ImagePart:\n', ImagePart
            return False

    @classmethod
    def TransLine(cls, lineDualQuaternion, dualQuaternion):
        Lb = lineDualQuaternion[0]
        Mb = lineDualQuaternion[1]
        Q = dualQuaternion[0]
        QImage = dualQuaternion[1]
        QConj = Quaternion.QuaternionConj(Q)
        t = 2*Quaternion.QuaternionProduct(QImage, QConj)
        tConj = Quaternion.QuaternionConj(t)

        La = Quaternion.rotateByQuaternion(Q, Lb)
        Ma = Quaternion.rotateByQuaternion(Q, Mb) \
             + 0.5*(Quaternion.QuaternionProduct(La, tConj) + Quaternion.QuaternionProduct(t, La))

        return (La, Ma)


    @classmethod
    def Line2DualQuaternion(cls, lineVector, point):
        lineQuaternion = Quaternion.Pose2Quaternion(lineVector)
        pointQuaternion = Quaternion.Pose2Quaternion(point)

        return (lineQuaternion, pointQuaternion)

    @classmethod
    def DualQuaternion2T (cls, DQ):
        '''

        :param DQ: This is a tuple, 2 elements tuple. Real part and Imagine Part
        :return:
        '''
        q = DQ[0,0]
        p = DQ[1:4,0]

        q_Real = q[0]
        q_Imagine = list(q[1])
        p_Real = p[0]
        p_Imagine = list(p[1])
        # p_Imagine = p[1]

        # q_Real = cos(theta/2)
        half_Theta = np.arccos(q_Real)
        Theta = np.float64( 2.0 * half_Theta)

        # q_Imagine = sin(theta/2) * r_hat
        r_hat = np.array(q_Imagine,dtype=np.float64) / np.sin(half_Theta)

        Rot_Axis = r_hat * Theta

        R,_ = cv2.Rodrigues(Rot_Axis)
        R = R.reshape((3,3))

        #------------- Now solve the t, q_prime = 0.5*t*q -----------------
        Q = [q_Real] + q_Imagine
        P = [p_Real] + p_Imagine


        A = np.zeros((3,3),dtype=np.float64)
        A[0,0] = Q[0];      A[0,1] = Q[3];     A[0,2] = -Q[2]
        A[1,0] = -Q[3];     A[1,1] = Q[0];     A[1,2] = Q[1]
        A[2,0] = Q[2];      A[2,1] = -Q[1];    A[2,2] = Q[0]

        B = np.ones((3,1),dtype=np.float64)
        B[0,0] = B[0,0] * P[1]
        B[1,0] = B[1,0] * P[2]
        B[2,0] = B[2,0] * P[3]

        # At = B
        t = 2.0 * np.dot( np.linalg.inv(A), B )
        t = t.reshape((3,1))
        print t

        T_row02 = np.hstack((R,t))
        T_row3 = np.zeros((1,4))
        T_row3[0,3] = 1
        T = np.vstack((T_row02,T_row3))
        return T



if __name__ == '__main__':
    from Src.ToolBox.VisionGeometryLib import VisionGeometryLib as VGL
    # -------------------- data prepare --------------------
    # random T matrix
    # t = np.random.random((3,1))
    # r = np.random.random((3,1))
    # r = r / np.linalg.norm(r)
    # R = cv2.Rodrigues(r)[0]
    #
    # T = np.zeros((4,4))
    # T[0:3, 0:3] = R
    # T[0:3, 3] = t.T
    # T[3, 3] = 1
    # # random point
    # Vector = np.random.random((3,1))
    # Vector = Vector / np.linalg.norm(Vector)
    # Point = np.array([0, 0, 0]).reshape(3,1)
    # VectorHomo = VGL.Homo(Vector)
    # startPointHomo = VGL.Homo(Point)
    # print 'VectorHomo:\n', VectorHomo.T
    #
    # rotMatrixVector = np.dot(T, VectorHomo)
    # rotMatrixPoint = np.dot(T, startPointHomo)
    # print 'transByT:\n'
    # print 'rotMatrixVector:\n', rotMatrixVector.T
    # rotMatrixVector = rotMatrixVector - rotMatrixPoint
    # rotMatrixVectorNorm = (rotMatrixVector / np.linalg.norm(rotMatrixVector)).T
    # print 'norm vector:\n', rotMatrixVectorNorm
    # print 'rotMatrixPoint:\n', rotMatrixPoint.T
    # # ------------------------------------------------------
    #
    # LineDualQuaternion = DualQuaternion.Line2DualQuaternion(Vector, Point)
    # # print LineDualQuaternion
    # DQ = DualQuaternion.T2DualQuaternion(T)
    # # print 'DualQuaternion:\n', DQ
    # print 'Normalized? ', DualQuaternion.CheckNormalized(DQ)
    #
    # transLine = DualQuaternion.TransLine(LineDualQuaternion, DQ)
    # # print 'transByDQ:\n', transLine
    # L = transLine[0][1:4, 0]
    # print 'LineDQ:\n', L
    # M = transLine[1]
    # print 'MDQ:\n', M.T
    # from GeometryHelper         import  GeometryHelper
    # skTT = -GeometryHelper.skewMatrix(rotMatrixVectorNorm[0,0:3])
    # print 'skTT:\n', skTT
    # mmmT = np.dot(skTT, rotMatrixPoint[0:3,0].reshape(3,1))
    # P = np.dot(np.linalg.inv(skTT), mmmT)
    # print 'mmmT:\n', mmmT.T
    # mT = np.cross(rotMatrixPoint[0:3,0], rotMatrixVectorNorm[0,0:3])
    # print 'mT:\n', mT
    # print 'RecoverP:\n', P.T
    # 
    # SkL = GeometryHelper.skewMatrix(L)
    # print 'Skl:\n', SkL.T
    # m = M[1:4, 0].reshape(3,1)
    # print 'm:\n', m.T
    # # print m.shape
    # point = np.dot(np.linalg.inv(SkL), m)
    # print 'point:', point
    # # print 'm:\n', M.T

    # A = np.array([[  9.24167888e-01,  -1.86301897e-01,  -3.33474615e-01,  3.82995350e+02],
    #               [  1.45044142e-01,   9.78766177e-01,  -1.44841180e-01,  1.23806927e+02],
    #               [  3.53377861e-01,   8.54890282e-02,   9.31566269e-01, -9.60853810e+00],
    #               [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,  1.00000000e+00]])
    # B = np.array([[ 0.98476289, -0.16405414,  0.05769127,  0.07995729],
    #               [ 0.13197809,  0.92106009,  0.36637426,  0.40785189],
    #               [-0.11324234, -0.35317779,  0.92867735, -0.03274141],
    #               [ 0.        ,  0.        ,  0.        ,  1.        ]])
    A = np.array([[  9.24167888e-01,   1.45044142e-01,   3.53377861e-01,
         -3.68514029e+02],
       [ -1.86301897e-01,   9.78766177e-01,   8.54890282e-02,
         -4.90038478e+01],
       [ -3.33474615e-01,  -1.44841180e-01,   9.31566269e-01,
          1.54602558e+02],
       [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          1.00000000e+00]])
    B = np.array([[ 0.9847629 ,  0.13197804, -0.11324236, -0.15769604],
       [-0.1640541 ,  0.9210601 , -0.35317778, -0.3723729 ],
       [ 0.05769131,  0.36637425,  0.92867735, -0.12424077],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
    DQ_B = DualQuaternion.T2DualQuaternion(A)
    DQ_A = DualQuaternion.T2DualQuaternion(B)
    print DQ_A
    print DQ_B
