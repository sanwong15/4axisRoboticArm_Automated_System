RobotControl使用

1. sendRobotPos(self, robotPos, error=None, type="GO", wait=True, timeout=0.1, resendTimes=3)
发送指令给机械手
参数:
	robotPos: 要发送的目的坐标
	error: 等待机械手相距目的坐标小于error值后返回,默认为None表示不等待
	type: 表示"GO"或者"MOVE",默认"GO"
	wait: 表示是否阻塞等待指令发送完毕,默认为True即可
	timeout: 表示动作完成超时时间,error为None时不生效
	resendTimes: 超时时重新发送命令次数,默认3次

2. getCurPos(self)
获取机械手当前坐标,返回（6,1）数组

3. setSpeed(self, SpeedValList)
设置机械手速度

4. setAcce(self, AcceValList)
设置加速度

5. isPoseOK(self, robotPos)
测试Target是否ok

附 测试代码
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

测试结果
~/Desktop/robcontrol$sudo python test_RC.py 
linking.....
11   SPEED
12   ACCEL
[350,153.8,121.0,51.5,0,0] isPoseOK? True
[[ 299.00442505]
 [ 153.99404907]
 [  75.99682617]
 [  81.99938202]
 [   0.        ]
 [   0.        ]]
3   MOVE
[[ 349.90701294]
 [ 153.86479187]
 [ 120.94458008]
 [  51.55408859]
 [   0.        ]
 [   0.        ]]
input anykey to next
1   GO
[[ 350.00039673]
 [ 153.79899597]
 [ 120.99963379]
 [  51.49969482]
 [   0.        ]
 [   0.        ]]


