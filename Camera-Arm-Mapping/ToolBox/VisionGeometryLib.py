#!/usr/bin/python2.7
# -*- coding:utf-8 -*-
__author__ = 'lh'
__version__ = '1.0'
__date__ = '22/08/2015'


import numpy as np
import math
import cv2


class VisionGeometryLib(object):
    @classmethod
    def isMat(cls, mat, checkSize=None):
        if not isinstance(mat, np.ndarray):
            return False
        if checkSize is None:
            return True
        elif mat.shape != tuple(checkSize):
            return False
        return True

    @classmethod
    def Rt2T(cls, R_3x3, t):
        """
        combine R,t to transform matrix
        :param R: 3x3 matrix ndarray
        :param t: 3x1 matrix ndarray, 1x3 matrix ndarray, 3 element array, list, tuple
        :return: 4x4  ndarray
        """
        assert cls.isMat(mat=R_3x3, checkSize=(3,3))
        assert isinstance(t, (list, tuple, np.ndarray))

        T = np.eye(4,4)
        T[0:3,0:3] = R_3x3
        T[:3, -1] = np.array(t).reshape(3)
        return T

    @classmethod
    def T2Rt(cls, T_4x4):
        """
        :param T: 4x4 transform matrix, np.ndarray
        :return : (R, t)
        """
        assert cls.isMat(mat=T_4x4, checkSize=(4,4))

        R = T_4x4[0:3,0:3].copy()
        t = T_4x4[0:3,-1].copy()
        return (R,t)

    @classmethod
    def Pose2T(cls, pose):
        """
        :param pose: list, tuple, np.array--(6,) , np.ndarray--(1,6), np.ndarray--(6,1)
                     pose = [x, y, z, u_deg, v_deg, w_deg]
        :return: the 4X4 matrix T
        """
        assert isinstance(pose, (list, tuple, np.ndarray))

        X, Y, Z, U_deg, V_deg, W_deg = np.array(pose, dtype=np.float32).reshape(6)
        U_rad = np.deg2rad(U_deg)
        V_rad = np.deg2rad(V_deg)
        W_rad = np.deg2rad(W_deg)

        Rx = np.array([[               1,               0,               0],
                       [               0, math.cos(W_rad),-math.sin(W_rad)],
                       [               0, math.sin(W_rad), math.cos(W_rad)]], np.float)

        Ry = np.array([[ math.cos(V_rad),               0, math.sin(V_rad)],
                       [               0,               1,               0],
                       [-math.sin(V_rad),               0, math.cos(V_rad)]], np.float)

        Rz = np.array([[ math.cos(U_rad),-math.sin(U_rad),               0],
                       [ math.sin(U_rad), math.cos(U_rad),               0],
                       [               0,               0,               1]], np.float)
        Ryx = np.dot(Ry,Rx)
        R = np.dot(Rz, Ryx)
        t = np.array([[X],[Y],[Z]],np.float)
        T = cls.Rt2T(R_3x3=R, t=t)
        return T

    @classmethod
    def T2Pose(cls, T_4x4):
        """
        :param T_4x4: The transform matrix
        :return: 6x1 position vector [x, y, z, u_deg, v_deg, w_deg]
        """
        assert cls.isMat(mat=T_4x4, checkSize=(4,4))

        R, t = cls.T2Rt(T_4x4=T_4x4)
        Angy_rad = math.atan2(-R[2,0], math.sqrt(R[0,0] ** 2 + R[1,0] ** 2))
        Angx_rad = math.atan2(R[1,0] / math.cos(Angy_rad), R[0,0] / math.cos(Angy_rad))
        Angz_rad = math.atan2(R[2,1] / math.cos(Angy_rad), R[2,2] / math.cos(Angy_rad))
        Angz_deg = np.rad2deg(Angz_rad)
        Angy_deg = np.rad2deg(Angy_rad)
        Angx_deg = np.rad2deg(Angx_rad)
        Angle_xyz = np.array([Angx_deg, Angy_deg, Angz_deg])
        Pose = np.hstack((t, Angle_xyz))
        return np.array(Pose, dtype=np.float32).reshape(-1)

    @classmethod
    def Homo(cls, mat):
        assert cls.isMat(mat=mat)
        assert mat.ndim < 3

        if 2 == mat.ndim:
            return np.vstack((mat, np.ones((1, mat.shape[1]))))
        else:
            return np.hstack([mat, [1]])

    @classmethod
    def unHomo(cls, mat):
        assert cls.isMat(mat=mat)
        assert mat.ndim < 3

        Temp = mat.copy().astype('float32')
        Temp /= Temp[-1]
        return Temp[0:-1]

    @classmethod
    def computeVectorAngle_rad (cls, vec0, vec1):
        Vec0 = np.array(vec0).reshape(-1)
        Vec1 = np.array(vec1).reshape(-1)
        assert Vec0.size == Vec1.size

        VecNorm0 = np.linalg.norm(Vec0)
        VecNorm1 = np.linalg.norm(Vec1)
        Acos = np.inner(Vec0, Vec1) / (VecNorm0 * VecNorm1)
        if Acos > 1:
            Acos = 1
        elif Acos < -1:
            Acos = -1
        Result =  math.acos(Acos)

        return Result

    @classmethod
    def computeRotateVec(cls, vec0, vec1):
        Vec0 = np.array(vec0).reshape(-1)
        Vec1 = np.array(vec1).reshape(-1)
        assert Vec0.size == Vec1.size

        Theta = cls.computeVectorAngle_rad(Vec0, Vec1)
        OrthogonalVec = np.cross(Vec0, Vec1)
        NormOrthogonalVec = OrthogonalVec / np.linalg.norm(OrthogonalVec)
        RodriguesVec = NormOrthogonalVec * Theta
        return RodriguesVec

    @classmethod
    def getTransformWith2LineSegment(cls, lineSegPts0_3x2, lineSegPts1_3x2):
        assert cls.isMat(mat=lineSegPts0_3x2, checkSize=(3,2))
        assert cls.isMat(mat=lineSegPts1_3x2, checkSize=(3,2))

        Vec0 = lineSegPts0_3x2[:,1] - lineSegPts0_3x2[:,0]
        Vec1 = lineSegPts1_3x2[:,1] - lineSegPts1_3x2[:,0]
        RotateVec = cls.computeRotateVec(vec0=Vec0, vec1=Vec1)
        RotateMatrix,_ = cv2.Rodrigues(RotateVec)
        t = lineSegPts1_3x2[:, 0] - np.dot(RotateMatrix , lineSegPts0_3x2[:, 0])
        T = cls.Rt2T(RotateMatrix, t)
        return T

    @classmethod
    def projectPts(cls, pts, projectMatrix):
        assert 2 == projectMatrix.ndim
        assert 2 == pts.ndim
        assert projectMatrix.shape[1] - pts.shape[0] < 2

        if projectMatrix.shape[1] == pts.shape[0]:
            return projectMatrix.dot(pts)
        else:
            return cls.unHomo(projectMatrix.dot(cls.Homo(pts)))

    @classmethod
    def projectPtsToImg(cls, pts_3xn, Tx2Cam, cameraMatrix, distCoeffs):
        assert cls.isMat(Tx2Cam, checkSize=(4,4))
        assert cls.isMat(cameraMatrix, checkSize=(3,3))

        R, t = cls.T2Rt(T_4x4=Tx2Cam)
        Rvec, _ = cv2.Rodrigues(R)
        ProjImgPts_nx2, _ = \
            cv2.projectPoints(objectPoints=pts_3xn.T.reshape(-1,3).astype('float32'),
                              rvec=Rvec.astype('float32'), tvec=t.astype('float32').reshape(-1, 3),
                               cameraMatrix=cameraMatrix, distCoeffs=distCoeffs)
        ProjImgPts_2xn = ProjImgPts_nx2.T.reshape(2, -1)
        return ProjImgPts_2xn

    @classmethod
    def unDistortPts(cls, imgPts_2xn, cameraMatrix, distCoeffs):
        """
        :param imgPts_2xn:
        :param cameraMatrix:
        :param distCoeffs:
        :return: UnDistorPts_2xn, UnDistortRay_2xn(z = 1)
        """
        assert cls.isMat(imgPts_2xn)
        assert 2 == imgPts_2xn.shape[0]
        assert cls.isMat(cameraMatrix, checkSize=(3,3))
        assert isinstance(distCoeffs, (tuple, list, np.ndarray))

        ImgPts_1xnx2 = imgPts_2xn.T.reshape(1, -1, 2)
        UnDistortRay_1xnx2 = cv2.undistortPoints(src=ImgPts_1xnx2.astype('float32'),
                                                 cameraMatrix=cameraMatrix,
                                                 distCoeffs=distCoeffs)
        UnDistortRay_2xn = UnDistortRay_1xnx2[0].T
        UnDistortPts_2xn = cls.unHomo(cameraMatrix.dot(cls.Homo(UnDistortRay_2xn)))
        return UnDistortPts_2xn, UnDistortRay_2xn

    @classmethod
    def unDistort(cls, img, cameraMatrix, distCoeffs):
        cls.isMat(cameraMatrix, checkSize=(3,3))
        unDistortImg = cv2.undistort(src=img, cameraMatrix=cameraMatrix, distCoeffs=distCoeffs)
        return unDistortImg

    @classmethod
    def reProj3dPts(cls, imgPtsA_2xn, imgPtsB_2xn, cameraMatrixA, cameraMatrixB,
                    distCoeffsA, distCoeffsB, Tx2CamA, Tx2CamB, calcReprojErr=False):
        """
        :param imgPtsA_2xn:
        :param imgPtsB_2xn:
        :param cameraMatrixA:
        :param cameraMatrixB:
        :param distCoeffsA:
        :param distCoeffsB:
        :param Tx2CamA:
        :param Tx2CamB:
        :param calcReprojErr:
        :return: ReProj3dPts_3xn in X coordination
        """

        UnDistortPtsA_2xn, UnDistortRaysA_2xn\
            = cls.unDistortPts(imgPts_2xn=imgPtsA_2xn, cameraMatrix=cameraMatrixA, distCoeffs=distCoeffsA)
        UnDistortPtsB_2xn, UnDistortRaysB_2xn\
            = cls.unDistortPts(imgPts_2xn=imgPtsB_2xn, cameraMatrix=cameraMatrixB, distCoeffs=distCoeffsB)

        PtsInCoordinateX_4xn = \
            cv2.triangulatePoints(projMatr1=Tx2CamA[0:3],
                                  projMatr2=Tx2CamB[0:3],
                                  projPoints1=UnDistortRaysA_2xn.astype('float32'),
                                  projPoints2=UnDistortRaysB_2xn.astype('float32'))
        PtsInCoordinateX_3xn = cls.unHomo(PtsInCoordinateX_4xn)

        if calcReprojErr:
            PtsInCoordinateX_3xn_Norm = PtsInCoordinateX_3xn
            ProjImgPtsA_2xn = \
                cls.projectPtsToImg(pts_3xn=PtsInCoordinateX_3xn_Norm,
                                    Tx2Cam=Tx2CamA, cameraMatrix=cameraMatrixA, distCoeffs=distCoeffsA)
            RpErrA = np.linalg.norm(UnDistortPtsA_2xn - ProjImgPtsA_2xn, axis=0).mean()

            ProjImgPtsB_2xn = \
                cls.projectPtsToImg(pts_3xn=PtsInCoordinateX_3xn_Norm,
                                    Tx2Cam=Tx2CamB, cameraMatrix=cameraMatrixB, distCoeffs=distCoeffsB)
            RpErrB = np.linalg.norm(UnDistortPtsB_2xn - ProjImgPtsB_2xn, axis=0).mean()

            return PtsInCoordinateX_3xn, RpErrA, RpErrB
        return PtsInCoordinateX_3xn, None, None


if __name__ == "__main__":
    import yaml
    VGL = VisionGeometryLib


    with open('../../res/input/CameraCalibrationData.yaml', 'r') as File:
        CalibrationData = yaml.load(File)
    CameraMatrixA = np.array(CalibrationData['Intrinsic_1414']) # Camera0
    CameraMatrixB = np.array(CalibrationData['Intrinsic_1313']) # Camera1
    DistCoeffsA = np.array(CalibrationData['DistCoeffs_1414'])
    DistCoeffsB = np.array(CalibrationData['DistCoeffs_1313'])
    TcAcB = np.array(CalibrationData['Tc14c13'])
    TcBcA = np.array(CalibrationData['Tc13c14'])
    TctA = np.array(CalibrationData['Tct_1414'])
    TctB = np.array(CalibrationData['Tct_1313'])

    PointCam14_2xn = np.array([[6.366322275682084637e+02, 7.899394910272621928e+02],
                               [7.416458478168997317e+02, 6.542366444992233028e+02]], dtype=np.float32)
    PointCam13_2xn = np.array([[8.149478495193284289e+02, 9.375415089453317705e+02],
                               [3.091887147386404422e+02, 3.632621606795080424e+02]], dtype=np.float32)

    # print VGL.unDistortPts(imgPts_2xn=PointCam13_2xn, cameraMatrix=CameraMatrixA, distCoeffs=())
    # print VGL.unDistortPts(imgPts_2xn=PointCam13_2xn, cameraMatrix=CameraMatrixA, distCoeffs=DistCoeffsA)

    Pts_3xn, RpErrA, RpErrB = \
        VGL.reProj3dPts(imgPtsA_2xn=PointCam14_2xn, imgPtsB_2xn=PointCam13_2xn,
                        cameraMatrixA=CameraMatrixA, cameraMatrixB=CameraMatrixB,
                        distCoeffsA=DistCoeffsA, distCoeffsB=DistCoeffsB,
                        # Tx2CamA=np.eye(4), Tx2CamB=TcAcB, calcReprojErr=True)
                        Tx2CamA=np.linalg.inv(TctA), Tx2CamB=np.linalg.inv(TctB), calcReprojErr=True)
    print "PtsInTool_3xn:\n", Pts_3xn
    print "Dis: ", np.linalg.norm(Pts_3xn[:,0] - Pts_3xn[:,1])
    print "RpErrA:\n", RpErrA
    print "RpErrB:\n", RpErrB

    # Rt2T, T2Rt
    # R = np.eye(3)
    # t = [1,1,1]
    # T = VGL.Rt2T(R_3x3=R, t=t)
    # print T
    # R, t = VGL.T2Rt(T_4x4=T)
    # print R
    # print t

    # Pose2T, T2Pose
    # T = VGL.Pose2T(pose=[1,2,3,0,0,0])
    # print T
    # print VGL.T2Pose(T_4x4=T)

    # Homo, unHomo
    # print VGL.Homo(mat=np.eye(3))
    # print VGL.Homo(mat=np.array([3,3]))
    # print VGL.unHomo(mat=np.array([[1,2,3], [3,4,5]]))
    # print VGL.unHomo(mat=VGL.Homo(mat=np.array([3,3])))

    # RotateAngle, RotateVec
    # a = [[0, 1]]
    # b = [[3, 0]]
    # a = [0, 1, 1]
    # b = [3, 0, 1]
    # print VGL.computeVectorAngle_rad(vec0=a, vec1=b)
    # print VGL.computeRotateVec(b, a)

    # getTransformWith2LineSegment

    PinPts3d_3xn = np.array([[0,1],
                             [0,0],
                             [0,0]])
    HolePts3d_3xn = np.array([[0,0],
                              [0,1],
                              [0,1]])
    # T = VGL.getTransformWith2LineSegment(PinPts3d_3xn.copy(), HolePts3d_3xn.copy())
    # print "T:\n", T
    # NewPinPts_3xn = VGL.projectPts(pts=PinPts3d_3xn, projectMatrix=T)
    # print "NewPinPts_3xn: ", NewPinPts_3xn
	#
    # Ang = \
    #     VGL.computeVectorAngle_rad(vec0=NewPinPts_3xn[:,1]-NewPinPts_3xn[:,0],
    #                                vec1=HolePts3d_3xn[:,1]-HolePts3d_3xn[:,0])
    # print "Ang: ", Ang
	#
    # Pose = VGL.T2Pose(T_4x4=T)
    # print "Pose: ", Pose
    # Mark = [1,1,1,1,0,0]
    # Pose *= Mark
    Pose = [10,10,10,0,0,0]
    T = VGL.Pose2T(pose=Pose)
    print "T:\n", T
    # NewPinPts_3xn = VGL.projectPts(pts=PinPts3d_3xn, projectMatrix=T)
    NewPinPts_3xn = VGL.projectPts(pts=HolePts3d_3xn, projectMatrix=T)
    print "NewPinPts_3xn: ", NewPinPts_3xn
    Ang = \
        VGL.computeVectorAngle_rad(vec0=NewPinPts_3xn[:,1]-NewPinPts_3xn[:,0],
                                   vec1=HolePts3d_3xn[:,1]-HolePts3d_3xn[:,0])
    print "Ang: ", Ang



