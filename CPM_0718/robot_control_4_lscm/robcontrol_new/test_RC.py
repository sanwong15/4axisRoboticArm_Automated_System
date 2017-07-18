from RobotControl import RobotControl

if __name__ == "__main__":
	RC = RobotControl()
	RC.link(fileName="shm_VS.bin")
	RC.setSpeed([10,10,10])
	RC.setAcce([10,10,10,10,0,0])

	print '[350,153.8,121.0,51.5,0,0] isPoseOK?', RC.isPoseOK([350,153.8,121.0,51.5,0,0])
	print RC.getCurPos()
	RC.sendRobotPos(robotPos=[350,153.8,121.0,51.5,0,0],type="MOVE",error=0.5)
	print RC.getCurPos()
	raw_input('input anykey to next')	
        RC.sendRobotPos(robotPos=[299,154,76,82,0,0],type="GO")
	print RC.getCurPos()
