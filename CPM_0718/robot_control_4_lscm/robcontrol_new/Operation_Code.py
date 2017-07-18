# last modification: Oct 27, 2016 by Jason PENG
# Both Cmd and Enq: SPEED, ACCEL, ON, OFF, ELBOW
# Motion related
GO = 1
TGO = 2
MOVE = 3
ARC = 4
JUMP = 5
JUMP3 = 6
JUMP_WITH_HIGH_SPEED = 7
GO_DEG = 8
STOP = 10

# Settings:
SPEED = 11
ACCEL = 12
ELBOW = 13
HAND = 14

# I/O related
ON = 21
OFF = 22
SW = 23	# FOR READING IN_PORT

# Enquiries
WHERE = 41
INPOS = 42
TARGET_OK = 43
EXEC_TIME = 44
WHERE_DEG = 45
MOVE_OK = 46

# Other Controls
POWER = 51
MOTOR = 52
SFREE = 53
SLOCK = 54
QUIT = 60

# Special case for conveyor control 20161213:
CONV_SPEED = 61
CONV_ON = 62
CONV_CLEAR = 63
CONV_POS = 64