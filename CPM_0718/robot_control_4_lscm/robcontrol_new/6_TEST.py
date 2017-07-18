import com_interface as Com
import random
import time
from Operation_Code import *
import unittest
import math
import operator
import numpy as np
import copy

# Annotation: (20150804)

# Com.command_to_robot() is for command with a parameter list, like "GO", "JUMP", "ARC", "MOVE",
# "SPEED", "ACCEL", "ON", "OFF", "POWER";

# Com.get_status() is for enquiry with or without parameter list:
# without param: "SPEED", "ACCEL", "INPOS", "POWER", "ON", "OFF", "WHERE";
# with param: "TARGET_OK", "SW";

def testGo():
    OriginPosition = [418.4, 0, 629.89, 0, -90, -180]
    GoPosition = [464.686, -169.017, 297.854, 152.65, -4.135, 174.468]
    while True:
        radius = random.uniform(-30, 30)
        GoPosition[0] += radius
        GoPosition[1] += radius
        GoPosition[2] += radius
        print GoPosition
        TargetOk = Com.get_status(ShareMem, "TARGET_OK", GoPosition)
        if TargetOk[0] != 0:
            print "************Error!!!can not go to the position!!***************"
        else:
            Com.command_to_robot(ShareMem, "GO", GoPosition)
            ActualPosition = Com.get_curr_pos(ShareMem)[0:6]
            print "GoPosition = ", GoPosition
            print "ActualPosition = ", ActualPosition
            GoPosition[1] = -GoPosition[1]
            Com.command_to_robot(ShareMem, "GO", GoPosition)
            ActualPosition = Com.get_curr_pos(ShareMem)[0:6]
            print "GoPosition = ", GoPosition
            print "ActualPosition = ", ActualPosition
            GoPosition = [464.686, -169.017, 297.854, 152.65, -4.135, 174.468]

def testMove():
    OriginPosition = [418.4, 0, 629.89, 0, -90, -180]
    GoPosition = [464.686, -169.017, 297.854, 152.65, -4.135, 174.468]
    while True:
        radius = random.uniform(-30, 30)
        GoPosition[0] += radius
        GoPosition[1] += radius
        GoPosition[2] += radius
        print GoPosition
        TargetOk = Com.get_status(ShareMem, "TARGET_OK", GoPosition)
        if TargetOk[0] != 0:
            print "************Error!!!can not go to the position!!***************"
        else:
            Com.command_to_robot(ShareMem, "MOVE", GoPosition)
            ActualPosition = Com.get_curr_pos(ShareMem)[0:6]
            print "GoPosition = ", GoPosition
            print "ActualPosition = ", ActualPosition
            GoPosition[1] = -GoPosition[1]
            Com.command_to_robot(ShareMem, "MOVE", GoPosition)
            ActualPosition = Com.get_curr_pos(ShareMem)[0:6]
            print "GoPosition = ", GoPosition
            print "ActualPosition = ", ActualPosition
            GoPosition = [464.686, -169.017, 297.854, 152.65, -4.135, 174.468]

def testThread():
    OriginPosition = [418.4, 0, 629.89, 0, -90, -180]
    GoPosition = [424.686, -169.017, 297.854, 152.65, -4.135, 174.468]
    while True:
        radius = random.uniform(-30, 30)
        GoPosition[0] += radius
        GoPosition[1] += radius
        GoPosition[2] += radius
        print GoPosition
        TargetOk = Com.get_status(ShareMem, "TARGET_OK", GoPosition)
        if TargetOk[0] != 0:
            print "************Error!!!can not go to the position!!***************"
        else:
            Com.command_to_robot(ShareMem, "GO", GoPosition)
            ActualPosition = Com.get_curr_pos(ShareMem)[0:6]
            print "GoPosition = ", GoPosition
            print "ActualPosition = ", ActualPosition
            GoPosition[1] = -GoPosition[1]
            Com.command_to_robot(ShareMem, "MOVE", GoPosition)
            ActualPosition = Com.get_curr_pos(ShareMem)[0:6]
            print "GoPosition = ", GoPosition
            print "ActualPosition = ", ActualPosition
            GoPosition = [464.686, -169.017, 297.854, 152.65, -4.135, 174.468]
            
            JumpP1=[474.39855489, 205.01568053, 115.40855304, 152.63334917, -4.30779311, 174.62465658]
            JumpP2= [479.14825634, 207.09563971, 395.88019415, 152.40037482, -2.2939525, 172.08569708]
            JumpP3= [479.3525761, -209.63551643, 395.92318385, 152.40334516, -2.29067504, 172.0860196]
            JumpP4 = [479.39361121, -209.6325469, 146.89025608, 152.39699665, -2.27772204, 172.07260997]
            Com.command_to_robot(ShareMem, "GO", JumpP1)
            JUMP3Position = [JumpP2,JumpP3,JumpP4,0,0,0]
            Com.command_to_robot(ShareMem, "JUMP3",JUMP3Position)

            randomNum = random.uniform(-40, 40)
            JumpP1[0] = JumpP1[0]+randomNum
            JumpP1[1] = JumpP1[1]+randomNum
            JumpP1[2] = JumpP1[2]+randomNum

            randomNum = random.uniform(-40, 40)
            JumpP2[0] = JumpP2[0]+randomNum
            JumpP2[1] = JumpP2[1]+randomNum
            JumpP2[2] = JumpP2[2]+randomNum

            randomNum = random.uniform(-40, 40)
            JumpP3[0] = JumpP3[0]+randomNum
            JumpP3[1] = JumpP3[1]+randomNum
            JumpP3[2] = JumpP3[2]+randomNum

            randomNum = random.uniform(-40, 40)
            JumpP4[0] = JumpP4[0]+randomNum
            JumpP4[1] = JumpP4[1]+randomNum
            JumpP4[2] = JumpP4[2]-randomNum
            
            Com.command_to_robot(ShareMem, "GO", JumpP1)
            JUMP3Position = [JumpP2,JumpP3,JumpP4,0,0,0]
            Com.command_to_robot(ShareMem, "JUMP3",JUMP3Position)

def testUnthread():
    OriginPosition = [418.4, 0, 629.89, 0, -90, -180]
    # GoPosition = [424.686, -169.017, 297.854, 152.65, -4.135, 174.468]
    # while True:
    #     radius = random.uniform(-30, 30)
    #     GoPosition[0] += radius
    #     GoPosition[1] += radius
    #     GoPosition[2] += radius
    GoPosition=[560.501, -33.202, 390.669, 152.65, -4.135, 174.468]
    while True:
        # radius = random.uniform(-30, 10)
        # GoPosition[0] += radius
        # GoPosition[1] += radius
        # GoPosition[2] += radius
        # print GoPosition
        TargetOk = Com.get_status(ShareMem, "TARGET_OK", GoPosition)
        if TargetOk[0] != 0:
            print "************Error!!!can not go to the position!!***************"
            continue
        else:
            print "GoPosition1 = ", GoPosition
            Com.command_to_robot(ShareMem, "GO", GoPosition, 0)
            # time.sleep(2)
            ActualPosition = Com.get_curr_pos(ShareMem)[0:6]
            print "ActualPosition1 = ", ActualPosition

            # GoPosition[1] = copy.deepcopy(-GoPosition[1])
            print "GoPosition2 = ", OriginPosition
            Com.command_to_robot(ShareMem, "GO", OriginPosition, 0)
            # time.sleep(2)
            ActualPosition2 = Com.get_curr_pos(ShareMem)[0:6]
            print "ActualPosition2 = ", ActualPosition2

            # GoPosition = [464.686, -169.017, 297.854, 152.65, -4.135, 174.468]
            
            # JumpP1=[474.39855489, 205.01568053, 115.40855304, 152.63334917, -4.30779311, 174.62465658]
            # JumpP2= [479.14825634, 207.09563971, 395.88019415, 152.40037482, -2.2939525, 172.08569708]
            # JumpP3= [479.3525761, -209.63551643, 395.92318385, 152.40334516, -2.29067504, 172.0860196]
            # JumpP4 = [479.39361121, -209.6325469, 146.89025608, 152.39699665, -2.27772204, 172.07260997]
            # Com.command_to_robot(ShareMem, "GO", JumpP1, 0)
            # JUMP3Position = [JumpP2,JumpP3,JumpP4,0,0,0]
            # Com.command_to_robot(ShareMem, "JUMP3",JUMP3Position, 0)

            # randomNum = random.uniform(-40, 40)
            # JumpP1[0] = JumpP1[0]+randomNum
            # JumpP1[1] = JumpP1[1]+randomNum
            # JumpP1[2] = JumpP1[2]+randomNum

            # randomNum = random.uniform(-40, 40)
            # JumpP2[0] = JumpP2[0]+randomNum
            # JumpP2[1] = JumpP2[1]+randomNum
            # JumpP2[2] = JumpP2[2]+randomNum

            # randomNum = random.uniform(-40, 40)
            # JumpP3[0] = JumpP3[0]+randomNum
            # JumpP3[1] = JumpP3[1]+randomNum
            # JumpP3[2] = JumpP3[2]+randomNum

            # randomNum = random.uniform(-40, 40)
            # JumpP4[0] = JumpP4[0]+randomNum
            # JumpP4[1] = JumpP4[1]+randomNum
            # JumpP4[2] = JumpP4[2]-randomNum
            
            # Com.command_to_robot(ShareMem, "GO", JumpP1, 0)
            # JUMP3Position = [JumpP2,JumpP3,JumpP4,0,0,0]
            # Com.command_to_robot(ShareMem, "JUMP3",JUMP3Position, 0)

if __name__ == '__main__':

    ShareMem = Com.link_port("../../robot_control_gui/shm_VS.bin")
    assert ShareMem != None, "Can not connect share memory! Please check the share memory filename!"

    Speed = [40, 40, 40]
    Accel = [40, 40, 40, 40, 40, 40]
    Com.command_to_robot(ShareMem, "SPEED", Speed)
    Com.command_to_robot(ShareMem, "ACCEL", Accel)
    # testGo()
    # testMove()
    # testThread()
    testUnthread()
