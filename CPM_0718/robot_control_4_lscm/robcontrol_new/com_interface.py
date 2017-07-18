# Last modified: Jason PENG, Jan 6, 2017, add MOVE_OK

# For Current version we will have to separate the interface into 4 and 6 axis parts, 
# because I do not know how to implement run-time configuration to tell the program how many
# joints we are currently controlling
# -- Jason  2015 July 24
################ COMMENT OVER ########################
# ***** Last Modified: 20170209 by Jason ******

# import the share memory library
from shm_module import *
from Operation_Code    import *
from datetime import datetime #20160803
import time
import random #20160616
import threading #20170209

JOINTS_MAX = 6
JOINTS = 4 #CHANGE THE NUMBER OF JOINTS HERE!

#shm file SIZE N INDEX configuration: (ver 150723)
# set the memory size, always 40 bytes, defined by Jason 
shm_size = 256 #expanded from 192, 20161018

CMD_CNTER_START = 0
I_CMD_CODE = 4
CMD_ARG_START = 5
CNTER_LOCK = 30 # added 20160616, dealing with python cmd in multi-threads
SIGNAL_START = 31 # 20161026 for custom signals, Occupy range: 31-36
FB_POS_START = 37
ENQ_IN_FLAG = 63
ENQ_RESPOND_FLAG = 64
I_ENQ_CODE = 65
ENQ_RESULT_START = 66
O_START = 90
I_START = 94
JUMP3_PNTS = 102 # added 20150917
RESP_CNTER_START = 185 #added 20160419, need 4 Bytes
ENQ_PARAM_START = 189 # added 20161018, not using CMD Area and Mode 0 for enq

NO_SOL_FLAG = shm_size - 4
MOV_STATE_IN = shm_size - 3
I_MOV_STATE = shm_size - 2
I_TERMINATION = shm_size - 1

# global lock for enquiry/cmd operation at python side, in case of two enqs arrive together, and mess up each other's parameter zone
sem_cmd = threading.Lock() #20170209 add lock for thread safe
sem_enq = threading.Lock() #20170209 add lock for thread safe

# Arc operation requires only (X2, Y2, X3, Y3) or (X2, Y2, Z2, X3, Y3, Z3).
# Jump is now only for 4Axis SCARA, and No LimZ provided.
# This is about input value? and for TARGET_OK and SW(single port enquiry), it is returning values 
cmd_chars = {    "GO": JOINTS*4, "TGO": JOINTS*4, "MOVE": JOINTS*4, "ARC": JOINTS_MAX*4, "GO_DEG": JOINTS*4,\
                "JUMP": 4*(JOINTS+1), "STOP": 0, "SPEED": 12, "ACCEL": 24, "POWER": 4, "MOTOR": 4, \
                "SFREE": JOINTS*4, "SLOCK": JOINTS*4, "ON": 4, "OFF": 4, "SW": 4, "ELBOW": 4, \
                "WHERE": 0, "QUIT": 0, "JUMP3": (3*6*4+4*3),"JUMP_WITH_HIGH_SPEED": 4*(JOINTS+1), \
                "CONV_SPEED": 4, "CONV_ON": 4, "CONV_CLEAR" : 4, "HAND": 4 }

# For Enquiry, how many chars to extract from shm_VS to form the result list
enq_chars = {    "STOP": 0, "SPEED": 12, "ACCEL": 24, "POWER": 4, "MOTOR": 4, "INPOS": 4,\
                "SFREE": JOINTS*4, "SLOCK": JOINTS*4, "ON": 16, "OFF": 16, "SW": 4, "ELBOW": 4, "MOVE_OK": 4, \
                "WHERE": JOINTS*4, "TARGET_OK": 4, "EXEC_TIME": 4, "QUIT": 0, "WHERE_DEG": JOINTS*4, "CONV_POS": 4 }
# For cmd and enq:
get_code = {"GO": GO, "TGO": TGO, "MOVE": MOVE, "ARC": ARC, "JUMP": JUMP, "STOP": STOP, "GO_DEG": GO_DEG,\
            "SPEED": SPEED, "ACCEL": ACCEL, "POWER": POWER, "MOTOR": MOTOR, "INPOS": INPOS, "ELBOW": ELBOW, \
            "SFREE": SFREE, "SLOCK": SLOCK, "ON": ON, "OFF": OFF, "SW": SW, "WHERE": WHERE, "WHERE_DEG": WHERE_DEG, \
            "TARGET_OK": TARGET_OK, "EXEC_TIME": EXEC_TIME, "QUIT": QUIT, "JUMP3": JUMP3,  "JUMP_WITH_HIGH_SPEED": JUMP_WITH_HIGH_SPEED, \
            "CONV_SPEED": CONV_SPEED, "CONV_ON": CONV_ON, "CONV_CLEAR" : CONV_CLEAR,  "CONV_POS": CONV_POS, "HAND": HAND, "MOVE_OK": MOVE_OK }

#For cmd only, watch out for difference between int and uint            
get_type = {"GO": "float", "TGO": "float", "MOVE": "float", "ARC": "float", "JUMP": "float", "STOP": "none", \
            "SPEED": "int", "ACCEL": "int", "POWER": "int", "MOTOR": "int", "INPOS": "none", "GO_DEG": "float", "ELBOW": "int", \
            "SFREE": "int", "SLOCK": "int", "ON": "int", "OFF": "int", "SW": "int", "WHERE": "none", "WHERE_DEG": "none",\
            "TARGET_OK": "none", "EXEC_TIME": "none", "QUIT": "none", "JUMP3": "none", "JUMP_WITH_HIGH_SPEED": "float", \
            "CONV_SPEED": "int", "CONV_ON": "int", "CONV_CLEAR" : "int", "HAND" : "int" } # Jump3 not listed, 2 different types

# Link the share memory segment, return None if the file does not exist
def link_port(filename):
    print 'linking.....'
    shm_obj = shm_link(filename, shm_size)
    #20161014: clean-up the dirty flags previous program left
    shm_write(shm_obj, 0, shm_size, ENQ_IN_FLAG, 1, 1)
    time.sleep(0.002) # 2ms
    shm_write(shm_obj, 0, shm_size, ENQ_RESPOND_FLAG, 1, 1)
    return shm_obj

# command counter adder, updated 20160419
def check_count(shm_obj):
    shm_count = shm_read(shm_obj, shm_size, CMD_CNTER_START, 4, 0) #return list of char data
    shm_respond_cnt = shm_read(shm_obj, shm_size, RESP_CNTER_START, 4, 0)

    count_i = uchar2uint(shm_count,4) #return list of unsigned int
    count_re = uchar2uint(shm_respond_cnt,4)

    # wait for the last cmd to be finished (respond cnter to catch up with cmd cnter)
    kcnt = 0
    while not count_i[0] == count_re[0]:
        time.sleep(0.0001)
        kcnt += 1
        if(kcnt%100000 == 0):
            print "Stuck 1: count_i = ", count_i[0], "count_re = ", count_re
            with open("./VS_log.txt", 'a+') as f:
                    f.write((str(datetime.now())+"Stuck 1: count_i = "+str(count_i[0])+"count_re = "+str(count_re[0])+"\n"))
        # print count_i[0], count_re[0]
        shm_count = shm_read(shm_obj, shm_size, CMD_CNTER_START, 4, 0)
        shm_respond_cnt = shm_read(shm_obj, shm_size, RESP_CNTER_START, 4, 0)
        count_i = uchar2uint(shm_count,4) #return list of unsigned int
        count_re = uchar2uint(shm_respond_cnt,4)

    return count_i

def add_count(shm_obj, count_i):

    # to prevent counter overflow:
    if count_i[0] < (2**32)-1:
        count_i[0] += 1     #### QUESTION: WHY USE [0]????
    else:
        count_i[0] = 0
    shm_count = uint2uchar(count_i, 4)
    shm_write(shm_obj, shm_count, shm_size, CMD_CNTER_START, 4, 0)
    return

# get the current robot pos, 
# return is a c_float array, the value must read with []
# i.e. fpos[0:4]  
# modified by Terence on 20150622
# read 6 parameter instead of 4
def get_curr_pos(shm_obj):
    uc_result = shm_read(shm_obj, shm_size, FB_POS_START, 24, 0)
    #convert uchar to float
    fpos = uchar2float(uc_result, 24)
    return fpos

# 20161026: For special signals for VS application, signal region index: [31]-[36]
def get_signal(shm_obj, signal_num):
    if signal_num<1 or signal_num>6:
        print "[ERROR]: get_signal() signal_num out of range :", signal_num
        return -1;
    status = shm_read(shm_obj, shm_size, (SIGNAL_START+signal_num-1), 1, 0)
    return status[0]

def set_signal(shm_obj, signal_num, on_off):
    if signal_num<1 or signal_num>6:
        print "[ERROR]: set_signal() signal_num out of range :", signal_num
        return -1;
    shm_write(shm_obj, on_off, shm_size, (SIGNAL_START+signal_num-1), 1, 0)


def push_cmd(shm_obj, cmd_str, param_list, param_type, ThreadOrNot, cmd_code):
    curr_cnt = check_count(shm_obj) # 20160615: wait for the last command responded first
    # 20160616: add lock to prevent race condition
    time.sleep(0.00001*random.randrange(1,10))
    kcnt = 0
    while (shm_read(shm_obj, shm_size, CNTER_LOCK, 1, 0)[0]): # wait for C-Core(InterfaceModule) to finish last cmd
        time.sleep(0.0001*random.randrange(1,10))
        kcnt += 1
        if(kcnt%100000 == 0):
            print "Stuck 2: CNTER_LOCKED"
            with open("./VS_log.txt", 'a+') as f:
                    f.write(str(datetime.now())+" Stuck 2: CNTER_LOCKED\n")
    shm_write(shm_obj, 1, shm_size, CNTER_LOCK, 1, 0)
    # below moved from command_to_robot() 20160616
    start_index = 29
    shm_write(shm_obj, ThreadOrNot, shm_size, start_index, 1, 0)
    shm_write(shm_obj, cmd_code, shm_size, I_CMD_CODE, 1, 0) # write cmd_code first 20150727 error
    print get_code[cmd_str], " ", cmd_str
    # above moved from command_to_robot() 20160616

    c_num = cmd_chars[cmd_str]
    param1 = [0]*18
    # if param_list not empty, write the params to shm in form of uchar.
    # or if len(param_list > 0):
    if JOINTS == 4 and cmd_str == "ARC":
        param_list.insert(2,0)
        param_list.append(0) #insert 2 elements to form a 3*2 input number
    if cmd_str == "JUMP3": # JUMP3 related,added 20150917
        # param part1: 3 point coordinate(6D) given to jump3 operation
        for i in range(6):
            param1[i] = param_list[0][i]
            param1[i+6] = param_list[1][i]
            param1[i+12] = param_list[2][i]
        print param1
        param2 = [param_list[3],param_list[4],param_list[5]]
        ucparam_pt1 = float2uchar(param1, (6*4*3))
        # param part2: 3 int configuration parameters
        ucparam_pt2 = uint2uchar(param2,4*3)
        shm_write(shm_obj, ucparam_pt2, shm_size, CMD_ARG_START, 4*3, 0)
        shm_write(shm_obj, ucparam_pt1, shm_size, JUMP3_PNTS, (6*4*3), 0)

    elif c_num > 0:
        if param_type == "int": ############## Do we need to discreminate int and uint???? ####################
            uc_param = uint2uchar(param_list,c_num) # convert param_list to uchar_list
        elif param_type == "float":
            uc_param = float2uchar(param_list,c_num)
        else:
            print "Error with parameter type !\n"
            return
        shm_write(shm_obj, uc_param, shm_size, CMD_ARG_START, cmd_chars[cmd_str], 0)

    add_count(shm_obj, curr_cnt) # modified 20160615
    shm_write(shm_obj, 0, shm_size, CNTER_LOCK, 1, 0) #20160616: release the lock

# send next pos to robot, 
# modified by Terence 20150623 # add global lock
def command_to_robot(shm_obj, cmd_str, param_list, ThreadOrNot=0, error=[2, 2, 1], wait=True):
    # modified here 20160616
    cmd_code = get_code[cmd_str]
    # 20161025: If thdOrNot is 1, NO WAIT for Motion even in wait_movement! 
    if ThreadOrNot:
        wait = False
    sem_cmd.acquire() #20170209, for thread safe
    push_cmd(shm_obj, cmd_str, param_list, get_type[cmd_str],ThreadOrNot, cmd_code) #20150727 error
    sem_cmd.release()
    wait_movement(shm_obj, cmd_code, param_list, error, wait=wait) # wait for the last movement to be finished
    # wait_movement(shm_obj,cmd_code, param_list, error=error)
    return

def get_move_flag(shm_obj):
    flag = shm_read(shm_obj, shm_size, I_MOV_STATE,1, 0)
    clear_to_robot = 0 
    shm_write(shm_obj, clear_to_robot, shm_size, MOV_STATE_IN, 1, 0)
    return flag[0]


# this includes the processing of "TARGET_OK" and "SW" command. Return list of unsigned integer
def Normal_enq(shm_obj,enq_str):
    num_of_chars = enq_chars[enq_str]
    uc_result = shm_read(shm_obj, shm_size, ENQ_RESULT_START, num_of_chars, 1) # 4 is sizeof(int)
    if not (enq_str == "WHERE_DEG" or enq_str == "CONV_POS"):
        result_list = uchar2uint(uc_result, num_of_chars) # ATTN: this is UINT list
    else:
        result_list = uchar2float(uc_result, num_of_chars) # ATTN: this is FLOAT list
    return result_list

def get_status(shm_obj, enq_str, param_list=None): # For normal enquiry other than "WHERE"
    sem_enq.acquire() #20170209, for thread safe, removed prev lock
    # CHECK !! Do we need to force transfer get_code[str] to be char type??
    if enq_str == "ON" or enq_str == "OFF":
        uc_result = shm_read(shm_obj, shm_size, O_START, 4, 0) # 32 outports (in bit) to form 4 bytes
        enq_result = uchar2uint(uc_result, 4) # form to an uint
    # elif enq_str == "SW":
    #     uc_result = shm_read(shm_obj, shm_size, I_START, 8, 1) # 64 inports (in bit) to form 4 bytes
    #     enq_result = uchar2ulong(uc_result, 8) # form to an ulong
    elif enq_str == "WHERE":
        enq_result = get_curr_pos(shm_obj)
    
    else:
        shm_write(shm_obj, get_code[enq_str], shm_size, I_ENQ_CODE, 1, 1) # Write the enq_str first
        # push the params of enquiry into cmd_argument area. 20150731
        if (enq_str == "TARGET_OK") or (enq_str == "EXEC_TIME") or (enq_str == "MOVE_OK"):
            uc_param = float2uchar(param_list, JOINTS*4);
            shm_write(shm_obj, uc_param, shm_size, ENQ_PARAM_START, JOINTS*4, 1)
        if enq_str == "SW":
            uc_param = uint2uchar(param_list, 4);
            shm_write(shm_obj, uc_param, shm_size, ENQ_PARAM_START, 4, 1)

        # then set the enq_IN flag to notify the other side
        # shm_write(shm_obj, 1, shm_size, ENQ _IN_FLAG, 1, 0)

        if shm_read(shm_obj, shm_size, ENQ_RESPOND_FLAG, 1, 1)[0] == 1:
            # shm_write(shm_obj, 1, shm_size, ENQ_IN_FLAG, 1, 0)

            print "[WARNING:]", enq_str, " Respond flag ready before waiting for it!!! "
            with open("./VS_log.txt", 'a+') as f:
                    f.write(str(datetime.now())+" [WARNING:] "+enq_str+" Respond flag ready before waiting for it!!! \n")
            # Jason add 20161014, in case of unfinished last enquiry
            timer2 = 0
            while(shm_read(shm_obj, shm_size, ENQ_RESPOND_FLAG, 1, 1)[0]):
                time.sleep(0.00005) # 50us
                timer2 += 1
                if timer2%10000 == 0 :
                    print "Stuck of [", enq_str, "] Waiting for Last Enq to finish~..."
                    print shm_read(shm_obj, shm_size, ENQ_RESPOND_FLAG, 1, 1)[0]

            # Jason add 20161014, finish
        #20161025: wait if there is another enq being served:
        timer2 = 0
        while (shm_read(shm_obj, shm_size, ENQ_IN_FLAG, 1, 1)[0]):
        	time.sleep(0.00005) # 50us
                timer2 += 1
                if timer2%10000 == 0 :
                    print "Stuck of [", enq_str, "] Waiting for Last Enq to finish..."
                    
        shm_write(shm_obj, 1, shm_size, ENQ_IN_FLAG, 1, 1)
        kcnt = 0
        while shm_read(shm_obj, shm_size, ENQ_RESPOND_FLAG, 1, 1)[0] == 0: # add[0], 20160705, wait for result to be updated by the other side
            time.sleep(0.00005) # 50us
            kcnt += 1
            if(kcnt%100000 == 0):
                print "Stuck 3: waiting for ENQ_RESPOND_FLAG"
                with open("./VS_log.txt", 'a+') as f:
                    f.write(str(datetime.now())+" Stuck 3: waiting for ENQ_RESPOND_FLAG\n")
        enq_result = Normal_enq(shm_obj,enq_str) # will get list of ints    

        shm_write(shm_obj, 0, shm_size, ENQ_RESPOND_FLAG, 1, 1)    # Done with the enquiry, clear the result flag
        # Jason add 20161014, to ensure clean finish:
        # timer3 = 0
        # while(shm_read(shm_obj, shm_size, ENQ_RESPOND_FLAG, 1, 0)[0]):
        #         time.sleep(0.00005) # 50us
        #         timer3 += 1
        #         if timer3%10000 == 0 :
        #             print "Not Clean finish of enquiry: [", enq_str, "] waiting..."
        # Jason add 20161014, finish
    sem_enq.release() #20170209, for thread safe, removed prev lock
    return enq_result


def wait_movement(shm_obj, cmd_code, param_list, error=[2, 2, 1], wait=True):
    time.sleep(0.005)
    # if cmd_code == 11 or cmd_code == 12:
    if cmd_code >= 10: # changed 20160615: may cause ON/OFF effective before actual stop
        return True
    else:
        if not wait:
            return [1]
        inpos = [0]
        while not inpos[0]:
            inpos = get_status(shm_obj, "INPOS", None)
            time.sleep(0.002) # added 20161028
            # if inpos[0]:
            #     if cmd_code<10:
            #         with open("./VS_log.txt", 'a+') as f:
            #             f.write(str(datetime.now())+" INPOS OK: "+str(param_list[0])+" "+str(param_list[1])+" "+str(param_list[2])+" "+str(param_list[3])+"\n")
            #     # print "INPOS OK: ", param_list[0:4]
            #     elif len(param_list)>0:
            #         with open("./VS_log.txt", 'a+') as f:
            #             f.write(str(datetime.now())+" cmd code: ["+str(cmd_code)+"] Done, with param: ("+str(param_list[0])+")\n")

        return inpos


