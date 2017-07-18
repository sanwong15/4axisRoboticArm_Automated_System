#!/usr/bin/python2.7
# -*- coding:utf-8 -*-
__author__ = 'hkh'
__version__ = 1.3
__date__ = 07/02/2017

import com_interface as Com
import numpy as np
import time
import math
import logging
import logging.config
import sys
sys.path.append('../../')

class RobotError(Exception):
    pass


class RobotRunTimeError(RobotError):
    pass


class RobotPosError(RobotError):
    pass


class RobotControl(object):
    def __init__(self, logger=None):
        self._my_logger = logging.getLogger(self.__class__.__name__)
        self.__ShmBuf = None
        self.__IsSendingCommand = False
        self.__IsGettingStatus = False

    def link(self, fileName):
        if not self.__ShmBuf:
            self._my_logger.info('link to robot...')
            self.__ShmBuf = Com.link_port(filename=fileName)
        if self.__ShmBuf:
            self._my_logger.info('Connected !!')
            return True
        self._my_logger.warn('Failed to Connect !!')
        return False

    def getCurPos(self):
        CurPos = Com.get_curr_pos(self.__ShmBuf)[0:6]
        return np.array([CurPos]).T

    def sendRobotPos(self, robotPos, error=None, type="GO", wait=True, timeout=0.1, resendTimes=3):
        assert isinstance(robotPos, (np.ndarray, list, tuple))
        # assert error is None or len(error) == 6
        T0 = time.time()
        if 6 == len(robotPos) and not self.isPoseOK(robotPos):
            raise RobotPosError('go pose is error')

        while self.__IsSendingCommand:
            self._my_logger.info('command ' + type + str(robotPos) + 'is waitting')
            time.sleep(0.01)

        self.__IsSendingCommand = True
        self._my_logger.info(type + "   sendRobotPos: " + str(robotPos))

        if isinstance(robotPos, np.ndarray) :
            robotPos.shape=(len(robotPos),)
            robotPos = robotPos.tolist()
        Com.command_to_robot(shm_obj
            =self.__ShmBuf, cmd_str=type, param_list=robotPos,
                             ThreadOrNot=1, error=error, wait=wait)
        self._my_logger.info('Command to robot end')
        self.__IsSendingCommand = False

        if error is None:
            return False
        PrePos = None
        while True:
            CurPos = self.getCurPos()
            time.sleep(0.01)
            if (abs(np.array(robotPos).reshape(-1) - CurPos.reshape(-1)) < np.array(error).reshape(-1)).all():
                return True

            if time.time() - T0 > timeout:

                if PrePos is not None:
                    AbsPos = np.abs(PrePos - CurPos)
                    self._my_logger.info('AbsPos: \n%s', AbsPos)
                    if (AbsPos < 0.05).all(): # robot is stop if the abs dis less than 0.05
                        if resendTimes <= 0:
                            raise RobotRunTimeError('Robot run time out')
                        self._my_logger.warn('command %s resending...', type)
                        self._my_logger.info('cur pos: %s', CurPos)
                        self._my_logger.info('error: %s', error)
                        self.sendRobotPos(robotPos=robotPos, error=error, type=type,
                                          wait=wait, timeout=timeout, resendTimes=resendTimes-1)
                        self._my_logger.warn('end of resend')
                PrePos = CurPos.copy()

            time.sleep(0.1)

    def goDelta(self, delta=(0,0,0,0,0,0), error=None, type="GO", wait=True):
        self._my_logger.info("goDelta: " + str(delta))

        CurPos = self.getCurPos()
        print 'curpos:', CurPos
        print 'delta:', delta
        CurPos += np.array(delta).reshape(6,1)
        self.sendRobotPos(CurPos, type=type, error=error, wait=wait)

    # def moveInTool(self, delta=(0,0,0,0,0,0), error=None, type="GO", wait=True):
    #     self._my_logger.info("moveInTool: %s"%(str(delta)))
    #     CurPos = self.getCurPos()
    #     CurTtr = vgl.Pose2T(pose=CurPos)
    #     M = vgl.Pose2T(pose=delta)
    #     NewTtr = CurTtr.dot(M)
    #     NewPose = vgl.T2Pose(T_4x4=NewTtr)
    #     self.sendRobotPos(NewPose, type=type, error=error, wait=wait)

    # def moveInRob(self, delta=(0,0,0,0,0,0), error=None, type="GO", wait=True):
    #     self._my_logger.info("moveInRob: %s"%(str(delta)))
    #     CurPos = self.getCurPos()
    #     CurTtr = vgl.Pose2T(pose=CurPos)
    #     M = vgl.Pose2T(pose=delta)
    #     NewTtr = M.dot(CurTtr)
    #     NewPose = vgl.T2Pose(T_4x4=NewTtr)
    #     self.sendRobotPos(NewPose, type=type, error=error, wait=wait)

    def setSpeed(self, SpeedValList):
        assert isinstance(SpeedValList, ( list, tuple))
        assert len(SpeedValList) == 3, 'SpeedValList has 3 values'
        self._my_logger.info('set speed')
        SpeedValList = np.int32(SpeedValList).tolist()
        Com.command_to_robot(self.__ShmBuf, "SPEED",  SpeedValList)

    def setAcce(self, AcceValList):
        assert isinstance(AcceValList, ( list, tuple))
        assert len(AcceValList) == 6, 'SpeedValList has 6 values'
        self._my_logger.info('set acce')
        AcceValList = np.int32(AcceValList).tolist()
        Com.command_to_robot(self.__ShmBuf, "ACCEL",  AcceValList)

    def getStatus(self, io):
        while self.__IsGettingStatus:
            self._my_logger.info('command ' + 'io' + str(io) + 'is waiting...')
            time.sleep(0.01)
        self.__IsGettingStatus = True
        io_status = Com.get_status(shm_obj=self.__ShmBuf, enq_str="SW", param_list=io)
        self.__IsGettingStatus = False
        time.sleep(0.01)
        return io_status[0]

    def isPoseOK(self, robotPos):
        assert isinstance(robotPos, (np.ndarray, list, tuple))
        assert self.__ShmBuf is not None, 'Shm is not connected!'

        while self.__IsGettingStatus:
            self._my_logger.info('command isPoseOK' + 'is waiting...')
            time.sleep(0.01)
        self.__IsGettingStatus = True

        if isinstance(robotPos, np.ndarray):
            GoPosition = robotPos.reshape(-1).tolist()
        else:
            GoPosition = list(robotPos)
        TargetOk = Com.get_status(self.__ShmBuf, "TARGET_OK", GoPosition)
        self._my_logger.info('TargetOk?%s', TargetOk[0])

        self.__IsGettingStatus = False
        time.sleep(0.01)

        return (0 == TargetOk[0])

    def setConveyorSpeed(self, speed):
        "speed: [0~28]"
        self._my_logger.info('set Conveyor Speed: ' + str(speed))
        Com.command_to_robot(self.__ShmBuf, "CONV_SPEED", speed)

    def runConveyor(self, dis_mm=None):
        self._my_logger.info('run Conveyor')
        Com.command_to_robot(self.__ShmBuf, "CONV_ON", [1])
        if dis_mm is not None:
            origin_dis_mm = self.getConveyorDis_mm()
            while True:
                time.sleep(0.1)
                now_dis = self.getConveyorDis_mm()
                if now_dis - origin_dis_mm > dis_mm:
                    self.stopConveyor()
                    break

    def stopConveyor(self):
        self._my_logger.info('stop Conveyor')
        Com.command_to_robot(self.__ShmBuf, "CONV_ON", [0])

    def clearConveyor(self):
        self._my_logger.info('clear Conveyor')
        Com.command_to_robot(self.__ShmBuf, "CONV_CLEAR", [1])
        Com.command_to_robot(self.__ShmBuf, "CONV_CLEAR", [0])

    def getConveyorDis_mm(self):
        ConveyorDis = Com.get_status(self.__ShmBuf, "CONV_POS")
        self._my_logger.info('get Conveyor Dis(mm): ' + str(math.pi * ConveyorDis[0]))
        return math.pi * ConveyorDis[0]

    def setHandType(self, flag):
        flag_map = {'left': [0], 'right': [1], 'auto': [2]}

        assert flag in flag_map
        self._my_logger.info('set hand type: %s', flag_map[flag])
        Com.command_to_robot(self.__ShmBuf, "HAND", flag_map[flag])

    def waitConveyorStop(self):
        self._my_logger.info('wait conveyor stop')
        pre_dis_mm = self.getConveyorDis_mm()
        while True:
            time.sleep(0.1)
            new_dis_mm = self.getConveyorDis_mm()
            self._my_logger.info('new_dis_mm: %s', new_dis_mm)
            if pre_dis_mm == new_dis_mm:
                break
            else:
                pre_dis_mm = new_dis_mm
        self._my_logger.info('end of wait conveyor stop')

if __name__ == '__main__':
    # LogConfig = io.loadYaml(fileName='../../res/input/logging_config.yaml')
    # logging.config.dictConfig(LogConfig)

    Control = RobotControl()
    Control.link(fileName="shm_VS.bin")

    print Control.getCurPos()
    #------------------test send Pos------------------
    Control.sendRobotPos(robotPos=[350,153.8,121.0,51.5,0,0],type="MOVE")
    raw_input('input anykey to next')
    Control.sendRobotPos(robotPos=[299,154,76,82,0,0],type="MOVE", error=[0.5,0.5,0.5,0.5,0.1,0.1])
    print Control.getCurPos()
    # IOData = io.loadYaml('../../res/input/robot_config.yaml')

    #------------------test IO control------------------
    # Control.sendRobotPos(robotPos=IOData['OUT_FrontGripper'],type="ON")
    # Control.sendRobotPos(robotPos=IOData['OUT_BackGripper'],type="ON")
    # Control.sendRobotPos(robotPos=IOData['OUT_RotateGripper'],type="ON")
    # Control.sendRobotPos(robotPos=IOData['OUT_Block'],type="ON")
    # Control.sendRobotPos(robotPos=IOData['OUT_Slider'],type="ON")
    # Control.sendRobotPos(robotPos=IOData['OUT_FrontCutterBlock'],type="ON")
    # Control.sendRobotPos(robotPos=IOData['OUT_FrontCutter'],type="ON")
    # Control.sendRobotPos(robotPos=IOData['OUT_Blow'],type="ON")
    # Control.sendRobotPos(robotPos=IOData['OUT_BackCutter'],type="ON")

    # Control.sendRobotPos(robotPos=IOData['OUT_FrontGripper'],type="OFF")
    # Control.sendRobotPos(robotPos=IOData['OUT_BackGripper'],type="OFF")
    # Control.sendRobotPos(robotPos=IOData['OUT_RotateGripper'],type="OFF")
    # Control.sendRobotPos(robotPos=IOData['OUT_Block'],type="OFF")
    # Control.sendRobotPos(robotPos=IOData['OUT_Slider'],type="OFF")
    # Control.sendRobotPos(robotPos=IOData['OUT_FrontCutterBlock'],type="OFF")
    # Control.sendRobotPos(robotPos=IOData['OUT_FrontCutter'],type="OFF")
    # Control.sendRobotPos(robotPos=IOData['OUT_Blow'],type="OFF")
    # Control.sendRobotPos(robotPos=IOData['OUT_BackCutter'],type="OFF")
    # time.sleep(1)
    # print Control.getStatus(io=IOData['IN_FrontGripper_OFF'])
    # print Control.getStatus(io=IOData['IN_FrontGripper_ON'])
    # print Control.getStatus(io=IOData['IN_BackGripper_OFF'])
    # print Control.getStatus(io=IOData['IN_BackGripper_ON'])
    # print Control.getStatus(io=IOData['IN_RotateGripper_OFF'])
    # print Control.getStatus(io=IOData['IN_RotateGripper_ON'])
    # print Control.getStatus(io=IOData['IN_Block_OFF'])
    # print Control.getStatus(io=IOData['IN_Block_ON'])
    # print Control.getStatus(io=IOData['IN_Slider_OFF'])
    # print Control.getStatus(io=IOData['IN_Slider_ON'])
    # print Control.getStatus(io=IOData['IN_FrontCutterBlock_OFF'])
    # print Control.getStatus(io=IOData['IN_FrontCutterBlock_ON'])
    # print Control.getStatus(io=IOData['IN_FrontCutter_OFF'])
    # print Control.getStatus(io=IOData['IN_FrontCutter_ON'])
    # print Control.getStatus(io=IOData['IN_BackCutter_OFF'])
    # print Control.getStatus(io=IOData['IN_BackCutter_ON'])

    #------------------test IO Movement------------------
    # COMPONENT_DIR = False
    # time.sleep(5)
    # Control.sendRobotPos(robotPos=IOData['OUT_FrontGripper'],type="OFF")
    # Control.sendRobotPos(robotPos=IOData['OUT_BackGripper'],type="OFF")
    # Control.sendRobotPos(robotPos=IOData['OUT_Block'],type="OFF")
    # while True:
    #     if Control.getStatus(io=IOData['IN_FrontGripper_OFF']) & \
    #         Control.getStatus(io=IOData['IN_BackGripper_OFF']):
    #         break
    # Control.sendRobotPos(robotPos=IOData['OUT_Slider'],type="OFF")
    # while True:
    #     if Control.getStatus(io=IOData['IN_Slider_OFF']):
    #         break
    # Control.sendRobotPos(robotPos=IOData['OUT_FrontCutterBlock'],type="OFF")
    # Control.sendRobotPos(robotPos=IOData['OUT_FrontCutter'],type="OFF")
    # Control.sendRobotPos(robotPos=IOData['OUT_Blow'],type="OFF")
    # Control.sendRobotPos(robotPos=IOData['OUT_BackCutter'],type="OFF")
    # Control.sendRobotPos(robotPos=IOData['OUT_RotateGripper'],type="OFF")
    # time.sleep(0.5)
    # if not (Control.getStatus(io=IOData['IN_FrontGripper_OFF'])& \
    #     Control.getStatus(io=IOData['IN_BackGripper_OFF'])     & \
    #     Control.getStatus(io=IOData['IN_RotateGripper_OFF'])   & \
    #     Control.getStatus(io=IOData['IN_Block_OFF'])           & \
    #     Control.getStatus(io=IOData['IN_Slider_OFF'])          & \
    #     Control.getStatus(io=IOData['IN_FrontCutterBlock_OFF'])& \
    #     Control.getStatus(io=IOData['IN_FrontCutter_OFF'])     & \
    #     Control.getStatus(io=IOData['IN_BackCutter_OFF'])):
    #     raise RuntimeError
    # exetime = 10
    # while exetime > 0:
    #
    #     #MOVE1
    #     Control.sendRobotPos(robotPos=IOData['OUT_FrontGripper'],type="ON")
    #     Control.sendRobotPos(robotPos=IOData['OUT_BackGripper'],type="ON")
    #     while True:
    #         if Control.getStatus(io=IOData['IN_FrontGripper_ON']) & \
    #             Control.getStatus(io=IOData['IN_BackGripper_ON']):
    #             break
    #     #MOVE2
    #     Control.sendRobotPos(robotPos=IOData['OUT_FrontCutter'],type="OFF")
    #     Control.sendRobotPos(robotPos=IOData['OUT_FrontCutterBlock'],type="OFF")
    #     while True:
    #         if Control.getStatus(io=IOData['IN_FrontCutter_OFF']) & \
    #             Control.getStatus(io=IOData['IN_FrontCutterBlock_OFF']):
    #             break
    #     # time.sleep(0.5)
    #     #MOVE2.5
    #     if not COMPONENT_DIR:
    #         if Control.getStatus(io=IOData['IN_RotateGripper_OFF']) & \
    #             (not Control.getStatus(io=IOData['IN_RotateGripper_ON'])):
    #             Control.sendRobotPos(robotPos=IOData['OUT_RotateGripper'],type="ON")
    #             while True:
    #                 if Control.getStatus(io=IOData['IN_RotateGripper_ON']):
    #                     break
    #         elif Control.getStatus(io=IOData['IN_RotateGripper_ON']) & \
    #             (not Control.getStatus(io=IOData['IN_RotateGripper_OFF'])):
    #             Control.sendRobotPos(robotPos=IOData['OUT_RotateGripper'],type="OFF")
    #             while True:
    #                 if Control.getStatus(io=IOData['IN_RotateGripper_OFF']):
    #                     break
    #         else:
    #             raise RuntimeError
    #     time.sleep(0.5)
    #     #MOVE3
    #     Control.sendRobotPos(robotPos=IOData['OUT_Slider'],type="ON")
    #     while True:
    #         if Control.getStatus(io=IOData['IN_Slider_ON']):
    #             break
    #     time.sleep(0.5)
    #     #MOVE4
    #     Control.sendRobotPos(robotPos=IOData['OUT_FrontCutterBlock'],type="ON")
    #     while True:
    #         if Control.getStatus(io=IOData['IN_FrontCutterBlock_ON']):
    #             break
    #     time.sleep(0.5)
    #     Control.sendRobotPos(robotPos=IOData['OUT_FrontCutter'],type="ON")
    #     while True:
    #         if Control.getStatus(io=IOData['IN_FrontCutter_ON']):
    #             break
    #     Control.sendRobotPos(robotPos=IOData['OUT_Blow'],type="ON")
    #     time.sleep(0.5)
    #     Control.sendRobotPos(robotPos=IOData['OUT_Blow'],type="OFF")
    #     Control.sendRobotPos(robotPos=IOData['OUT_BackCutter'],type="ON")
    #     while True:
    #         if Control.getStatus(io=IOData['IN_BackCutter_ON']):
    #             break
    #     time.sleep(0.5)
    #     #MOVE5
    #     Control.sendRobotPos(robotPos=IOData['OUT_FrontGripper'],type="OFF")
    #     Control.sendRobotPos(robotPos=IOData['OUT_BackGripper'],type="OFF")
    #     while True:
    #         if Control.getStatus(io=IOData['IN_FrontGripper_OFF']) & \
    #             Control.getStatus(io=IOData['IN_BackGripper_OFF']):
    #             break
    #     Control.sendRobotPos(robotPos=IOData['OUT_Slider'],type="OFF")
    #     while True:
    #         if Control.getStatus(io=IOData['IN_Slider_OFF']):
    #             break
    #     time.sleep(1)
    #     #MOVE6 ROBOT LOAD COMPONENT
    #     #ROBOT COME
    #     time.sleep(2)
    #     #ROBOT GRIPPER ON
    #     time.sleep(2)
    #     Control.sendRobotPos(robotPos=IOData['OUT_BackCutter'],type="OFF")
    #     while True:
    #         if Control.getStatus(io=IOData['IN_BackCutter_OFF']):
    #             break
    #     #ROBOT LEAVE
    #     time.sleep(2)
    #
    #     exetime -= 1

    #---------------test Conveyor-------------------
    # Control.setConveyorSpeed([10])
    # Control.clearConveyor()
    # Control.runConveyor()
    # T1 = time.time()
    # time.sleep(3)
    # # Dis = Control.getConveyorDis_mm()
    # # print Dis
    # # Control.clearConveyor()
    # Control.stopConveyor()
    # Control.waitConveyorStop()
    # T2 = time.time()
    # # time.sleep(5)
    # Dis = Control.getConveyorDis_mm()
    # # Control.clearConveyor()
    # time.sleep(10)
    # Dis = Control.getConveyorDis_mm()

    # print Dis
    # print 'T2-T1:', T2-T1
