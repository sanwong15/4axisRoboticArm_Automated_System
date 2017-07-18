from RobotControl import RobotControl

if __name__ == "__main__":
	RC = RobotControl()
	RC.link(fileName="shm_VS.bin")
	RC.setSpeed([10,10,10])
	RC.setAcce([10,10,10,10,0,0])

	# zx, oi, begin
	#print '[350,153.8,121.0,51.5,0,0] isPoseOK?', RC.isPoseOK([350,153.8,121.0,51.5,0,0])
	#print RC.getCurPos()
	RC.sendRobotPos(robotPos=[350,153.8,121.0,51.5,0,0],type="MOVE",error=0.5)
	# zx, oi, end

	#RC.sendRobotPos(robotPos=[200.0, -226.0, 395.07, -37.89, 0.0, 89.92], type="MOVE", error=0.5)
	print RC.getCurPos()
	
	raw_input('input anykey to next')	
 	# zx, oi, begin
        #RC.sendRobotPos(robotPos=[299,154,60,82,0,0],type="GO")
	# zx, oi, end

	#RC.sendRobotPos(robotPos=[221.13211468, 144.40044912, 60, 82, 0, 0], type="GO")
	#RC.sendRobotPos(robotPos=[364.29494662, -52.58334555, 54.57142639, 0.66162154, 0.0, 0.0], type="GO")  # p0
	#RC.sendRobotPos(robotPos=[340.12691049, -55.48038278, 54.57142639, 0.66162154, 0.0, 0.0], type="GO")  # p1
	#RC.sendRobotPos(robotPos=[337.16612335, -30.80669559, 54.57142639, 0.66162154, 0.0, 0.0], type="GO")  # p8
	#RC.sendRobotPos(robotPos=[239.50279732, -42.36209227, 54.57142639, 0.66162154, 0.0, 0.0], type="GO")  # p12
	#RC.sendRobotPos(robotPos=[233.35125407, 7.32913015, 54.57142639, 0.66162154, 0.0, 0.0], type="GO")   # p26
	#RC.sendRobotPos(robotPos=[223.98607617, 82.37886273, 54.57142639, 0.66162154, 0.0, 0.0], type="GO")   # p47
	
	# 1000.jpg, begin	
	#RC.sendRobotPos(robotPos=[196.21386192+18, 32.29851444-16.7, 57.57142639, 0.66162154, 0.0, 0.0], type="GO")  # for_test_1000.jpg

	# point2
	#RC.sendRobotPos(robotPos=[342.24323602+18, 60.29855881-16.7, 57.57142639, 0.66162154, 0.0, 0.0], type="GO")  # for_test_1000.jpg
        #RC.sendRobotPos(robotPos=[352.57355879, 46.13205324, 57.57142639, 0.66162154, 0.0, 0.0], type="GO")  # point#29_afterH_times
	# 1000.jpg, end

        # 1001.jpg, begin
        #RC.sendRobotPos(robotPos=[275.49137749+13, 111.68435498-16.7, 57.57142639, 0.66162154, 0.0, 0.0], type="GO")
	#raw_input('input anykey to next')
	#RC.sendRobotPos(robotPos=[279.00826071+18, 50.20335986-16.7, 57.57142639, 0.66162154, 0.0, 0.0], type="GO")
	#raw_input('input anykey to next')
	#RC.sendRobotPos(robotPos=[334.55578991+18, -50.14409025-16.7, 57.57142639, 0.66162154, 0.0, 0.0], type="GO")
	#raw_input('input anykey to next')
	#RC.sendRobotPos(robotPos=[391.79899781+18, -103.54947049, 57.57142639, 0.66162154, 0.0, 0.0], type="GO")
        #raw_input('input anykey to next')
        #RC.sendRobotPos(robotPos=[355.90477777+18, -152.27927785-16.7, 57.57142639, 0.66162154, 0.0, 0.0], type="GO")
        # 1001.jpg, end


        # 1002.jpg, begin
        RC.sendRobotPos(robotPos=[276.23754593+18, 119.91703619-16.7, 57.57142639, 0.66162154, 0.0, 0.0], type="GO")
	raw_input('input anykey to next')
	RC.sendRobotPos(robotPos=[351.35577807+18, 101.4178561-16.7, 57.57142639, 0.66162154, 0.0, 0.0], type="GO")
	raw_input('input anykey to next')
	RC.sendRobotPos(robotPos=[360.79541748+18, 36.44643979-16.7, 57.57142639, 0.66162154, 0.0, 0.0], type="GO")
	raw_input('input anykey to next')
	RC.sendRobotPos(robotPos=[147.95403952+18, 26.47254402-16.7, 57.57142639, 0.66162154, 0.0, 0.0], type="GO")
        raw_input('input anykey to next')
        RC.sendRobotPos(robotPos=[191.24360245+18, -10.18593234-16.7, 57.57142639, 0.66162154, 0.0, 0.0], type="GO")
 	raw_input('input anykey to next')
        RC.sendRobotPos(robotPos=[300.82204+18, -31.64355219-16.7, 57.57142639, 0.66162154, 0.0, 0.0], type="GO")
  	raw_input('input anykey to next')
        RC.sendRobotPos(robotPos=[235.68927155+18, -50.23715651-16.7, 57.57142639, 0.66162154, 0.0, 0.0], type="GO")
   	raw_input('input anykey to next')
        RC.sendRobotPos(robotPos=[297.90945592+18, -115.74260043-16.7, 57.57142639, 0.66162154, 0.0, 0.0], type="GO")
  	raw_input('input anykey to next')
        RC.sendRobotPos(robotPos=[353.48296938+18, -114.41162392-16.7, 57.57142639, 0.66162154, 0.0, 0.0], type="GO")
	raw_input('input anykey to next')
        RC.sendRobotPos(robotPos=[307.80999916+18, -155.70215989-16.7, 57.57142639, 0.66162154, 0.0, 0.0], type="GO")
        # 1002.jpg, end

	print RC.getCurPos()
