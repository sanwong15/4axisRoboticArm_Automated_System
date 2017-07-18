#!/usr/bin/python2.7
# -*- coding:utf-8 -*-
__author__ = 'pi'
__version__ = '1.0'
__date__ = '09/08/2016'


import os
import sys
CurrentPath = os.path.dirname(__file__) or '.'
sys.path.insert(0, os.path.abspath(os.path.join(CurrentPath, os.path.pardir)))
sys.path.insert(0, os.path.abspath(os.path.join(CurrentPath, os.path.pardir, os.path.pardir, os.path.pardir)))

import cv2
import numpy as np
import numpy.testing as testNumpy
from unittest import TestCase

from Src.ToolBox.MyQuaternion import Quaternion, EulerAngles
from Src.ToolBox.VisionGeometryLib import VisionGeometryLib as VGL


class TestFromRotationAxisAngle(TestCase):
    def testPose(self):
        Pose = np.array([10, 20, 30, 40, 50, 60])
        # Pose = np.array([10, 20, 30, 90, 90, 90])
        Q1 = Quaternion.fromEulerAngles(Pose[3], Pose[4], Pose[5])
        T = VGL.Pose2T(Pose)
        R, t_vec  = VGL.T2Rt(T)
        # NewR = EulerAngles.toRotationMatrix(Pose[3], Pose[4], Pose[5])
        # print R
        # print NewR
        r_vec = cv2.Rodrigues(R)[0]
        Q2 = Quaternion.fromRotationAxisAngle(r_vec)
        Q3 = Quaternion.T2Quaternion(T).reshape(-1)
        Q4 = Quaternion.T2Quaternion2(T).reshape(-1)
        print Q1
        print Q2
        print Q3
        print Q4
        assert np.allclose(Q1, Q2)
        assert 1 == 2
        # testNumpy.assert_array_equal(Q1, Q2)



if __name__ == '__main__':
    import nose
    nose.run()