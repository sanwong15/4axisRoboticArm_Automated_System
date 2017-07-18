import random
import time
import yaml
import sys
sys.path.append("./lib/")
sys.path.append("./")
import com_interface as Com
from Operation_Code import *

# test events: GO MOVE ARC JUMP STOP SPEED ACCEL POWER MOTOR ON OFF SW WHERE TARGET_OK QUIT JUMP3 JUMP_WITH_HIGH_SPEED
#              INPOS 

# define origin position:
global OriginPosition
OriginPosition = [400, 0, 123, 0]

def compareList(listA, listB, error=1.5):
    length=len(listA)
    for i in range(0,length):
        if abs(listA[i]-listB[i]) > error:
            assert 0>1 #when the error more than 1 mm, the movement is not a good motion.
        else:
            continue

def randomPosition(listA,randomRangeList=[15,15,20,20,15,15]):
    length = len(listA)
    for i in range(0,length):
        randomNum = random.uniform(-randomRangeList[i], randomRangeList[i])
        listA[i] = listA[i]+randomNum
    return listA

def linkShm(ShmVs):
    global ShareMem
    ShareMem = Com.link_port(ShmVs)
    assert ShareMem != None, "Can not connect share memory! Please check the share memory filename!"

def initSpeed(SpeedValue):
    Speed = [SpeedValue, SpeedValue, SpeedValue]
    Accel = [SpeedValue, SpeedValue, SpeedValue, SpeedValue, SpeedValue, SpeedValue]
    Com.command_to_robot(ShareMem, "SPEED", Speed)
    Com.command_to_robot(ShareMem, "ACCEL", Accel)

def loadYaml(yamlPath):
    with open(yamlPath, 'r') as YamlPoints:
        return yaml.load(YamlPoints)    

class RobotTest:
    def __init__(self, PointsYamlPath):
        self.YamlLoadPoints = loadYaml(PointsYamlPath)
        initSpeed(self.YamlLoadPoints['Speed'])

    def testTarget_OK(self,position):
        """
        test the point whether can get to or not
        input:  position
        return: 0 or -1, 0: TargetOk; -1: position input error
        """
        return Com.get_status(ShareMem, "TARGET_OK", position)[0]

    def TestGO(self,ThreadOrNot,testNumber,GoPosition):
        """
        command_to_robot()
            shareMem:       shmmemory name
            "GO":           target cmd 
            ToPosition:     target position 
            ThreadOrNot=1:  open thread 
            ThreadOrNot=0:  unthread
        """
        for i in range(0,testNumber):
            # GoPosition = [300,120,80,20]
            # randomList = [10,10,10,10,10,10]
            GoPosition = self.YamlLoadPoints['Go']
            GoPosition = randomPosition(GoPosition,randomRangeList=randomList)
            if not testTarget_OK(GoPosition):
                start =  time.time()
                Com.command_to_robot(ShareMem, "GO", GoPosition,ThreadOrNot)
                print "Go time = ",time.time()-start
                time.sleep(2)
                CurrentPosition = Com.get_curr_pos(ShareMem)
                print CurrentPosition[0:6]
                compareList(GoPosition,CurrentPosition,error=1)
            else:
                continue

    def TestMove(self,ThreadOrNot,testNumber):
        """
        command_to_robot()
            shareMem:       shmmemory name
            "GO":           target cmd 
            ToPosition:     target position 
            ThreadOrNot=1:  open thread 
            ThreadOrNot=0:  unthread
        """
        Com.command_to_robot(ShareMem, "GO", OriginPosition, ThreadOrNot=0)
        for i in range(0,testNumber):
            MovePosition = [264,120,123,0]
            randomList = [10,10,0,0,0,0]
            MovePosition = randomPosition(MovePosition,randomRangeList=randomList)
            if not testTarget_OK(MovePosition):
                start =  time.time()
                Com.command_to_robot(ShareMem, "MOVE", MovePosition,ThreadOrNot)
                print "MOVE time = ",time.time()-start
                time.sleep(2)
                CurrentPosition = Com.get_curr_pos(ShareMem)
                print CurrentPosition[0:6]
                compareList(MovePosition,CurrentPosition,error=1)
            else:
                continue    

    def TestJump(self,ThreadOrNot,testNumber):
        """
        command_to_robot()
            shareMem:       shmmemory name
            "GO":           target cmd 
            ToPosition:     target position 
            ThreadOrNot=1:  open thread 
            ThreadOrNot=0:  unthread
        """
        Com.command_to_robot(ShareMem, "GO", OriginPosition, ThreadOrNot=0)
        for i in range(0,testNumber):
            JumpPosition = [312,204,60,0,120,0]
            randomList = [25,25,30,30,0,0]
            JumpPosition = randomPosition(JumpPosition,randomRangeList=randomList)
            if not testTarget_OK(JumpPosition):
                start =  time.time()
                Com.command_to_robot(ShareMem, "JUMP", JumpPosition,ThreadOrNot)
                print "JUMP time = ",time.time()-start
                time.sleep(4)
                CurrentPosition = Com.get_curr_pos(ShareMem)
                CurrentPosition[4] = 120
                print CurrentPosition[0:6]
                compareList(JumpPosition,CurrentPosition,error=1)
            else:
                continue    

    def TestJumpWithHighSpeed(self, ThreadOrNot,testNumber):
        """
        command_to_robot()
            shareMem:       shmmemory name
            "GO":           target cmd 
            ToPosition:     target position 
            ThreadOrNot=1:  open thread 
            ThreadOrNot=0:  unthread
        """
        Com.command_to_robot(ShareMem, "GO", OriginPosition, ThreadOrNot=0)
        for i in range(0,testNumber):
            JumpHighPosition = [310,197,70,0,115,0]
            randomList = [25,25,30,30,0,0]
            JumpHighPosition = randomPosition(JumpHighPosition,randomRangeList=randomList)
            if not testTarget_OK(JumpHighPosition):
                start =  time.time()
                Com.command_to_robot(ShareMem, "JUMP_WITH_HIGH_SPEED", JumpHighPosition,ThreadOrNot)
                print "JUMP_HIGH time = ",time.time()-start
                time.sleep(10)
                CurrentPosition = Com.get_curr_pos(ShareMem)
                CurrentPosition[4] = 115
                print CurrentPosition[0:6]
                compareList(JumpHighPosition,CurrentPosition,error=1)
            else:
                continue    

# def TestIO():


if __name__ == '__main__':
    linkShm(ShmVs = "../../shm_VS.bin")
    # initSpeed(SpeedValue=100)
    Test = RobotTest("./fourPoints.yaml")
    # TestGO(ThreadOrNot=0,testNumber=10)
    # TestMove(ThreadOrNot=0,testNumber=10)
    # TestJump(ThreadOrNot=0,testNumber=10)
    # TestJumpWithHighSpeed(ThreadOrNot=0,testNumber=10)