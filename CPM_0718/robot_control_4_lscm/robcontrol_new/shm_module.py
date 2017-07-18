# Last modified date 2015 June 10 by Li Ying Chi

# shm_module contain the following fucntion

# define BUSY 126	//!< vaule represent busy for read/write flags
# define DONE 127	//!< vaule represent free for read/write flags

# //!  create the share memory segment
# int shm_create(unsigned char **shm_buf, const char *filename, const int shm_size);

# //! initialize the lock in the share memory
# void shm_lock_init(unsigned char *shm_buf, const int shm_size);

# //! Link the share memory segment to file descriptor
# int shm_link(unsigned char **shm_buf, const char *filename, const int shm_size);

# //! read the content from the share memory and put it into rbuf
# void shm_read(unsigned char *rbuf, unsigned char *shm_buf, const int shm_size, const int start, const int length);

# //! write the content from the wbuf into shm_buf
# void shm_write(unsigned char *shm_buf, unsigned char *wbuf, const int shm_size, const int start, const int length);

# //! Remove the share memory
# void shm_destroy(unsigned char **shm_buf, const char *filename, const int shm_size);

# now you can call the function in shm_moudle.c
# byref is pass byref in C

########################################################
import sys
import time
# import the ctypes library
from ctypes import *
import os
curr_path = os.path.dirname(os.path.abspath(__file__))
lib_path = curr_path + "/lib/libDataConv.so"
libc = cdll.LoadLibrary(lib_path)
# import the C++ shm_module library
sys.path.append(curr_path + '/lib')
import shm

# create a share memory segment
# input:    filename = file descriptor for the share memory
#           shm_size = memory buffer size
# output:   shm_buf = pointer for the share memory
def shm_create(filename, shm_size):
    shm_obj = shm.SharedMem(filename, shm_size)
    create_rt = shm_obj.create()
    assert(not create_rt) #if create_rt != 0, failed to create
    shm_obj.lock()
    return shm_obj

# Link the share memory segment to file descriptor
# input:    filename = file descriptor for the share memory
# output:   shm_buf = pointer for the share memory
def shm_link(filename, shm_size):
    shm_obj = shm.SharedMem(filename, shm_size)
    link_rt = shm_obj.link()
    assert(not link_rt) #if link_rt != 0, failed to link
    return shm_obj

# read a segment of the share memory
# input:    shm_buf = the memory buffer
#           shm_size = memory buffer size
#           start = starting position of the segment
#           length = the length of the segment
# output:   rbuf = an array contain the segment
# modified by Jason 20150723, add mode as parameter
def shm_read(shm_obj, shm_size, start, length, mode): # interface changed!
    # assert ((length+start) <= shm_size) #   check if the start position and length is valid
    rbuf = [0]*length
    shm_obj.readData(rbuf, start, length, mode)
    return rbuf


# write date to an segment of the share memory
# input:    shm_buf = the memory buffer going to write
#           wbuf_list = array or list contain the data to be written
#           shm_size = memory buffer size
#           start = starting position of the segment
#           length = the length of the segment
# modified by Jason 20150723, add mode as parameter
def shm_write(shm_obj, wbuf_list, shm_size, start, length, mode): # interface changed!
    # assert(start+length <= shm_size)
    wbuf_uchar = [0]*length
    # print "inner type is:", type(wbuf_uchar) #<class 'shm_module.c_ubyte_Array_1'>
    if (length == 1):
        wbuf_uchar[0] = wbuf_list
    else:
        for i in range(0, length):
            wbuf_uchar[i]=wbuf_list[i]

    shm_obj.writeData(wbuf_uchar, start, length, mode)
    return

# destroy the share memory
# input:    shm_buf = pointer to the memory buffer
#           filename = file descriptor for the share memory
#           shm_size = memory buffer size
def shm_destroy(shm_obj): # interface changed!
    shm_obj.destroy()
    return


## data type conversion
def uchar2float(buf_any, ucbufno):
    ucbuf = (c_ubyte*ucbufno)()
    for i in range(0, ucbufno):
        ucbuf[i]=buf_any[i]
    fbufno = ucbufno/4
    fbuf = (c_float*int(fbufno))()
    libc.uchar2float(fbuf, ucbuf, ucbufno)
    return fbuf

def float2uchar(buf_any, ucbufno):
    ucbuf = (c_ubyte*ucbufno)()
    fbufno = ucbufno/4
    fbuf = (c_float*int(fbufno))()
    for i in range(0, fbufno):
        fbuf[i]= buf_any[i]
    libc.float2uchar(ucbuf, fbuf, ucbufno)
    return ucbuf

def uchar2uint(buf_any, ucbufno):
    ucbuf = (c_ubyte*ucbufno)()
    for i in range(0, ucbufno):
        ucbuf[i]=buf_any[i]
    ibufno = ucbufno/4
    ibuf = (c_uint*int(ibufno))()
    libc.uchar2uint(ibuf, ucbuf, ucbufno)
    return ibuf

# added by Jason 20150724
def uchar2ulong(buf_any,ucbufno):
    ucbuf = (c_ubyte*ucbufno)()
    for i in range(0, ucbufno):
        ucbuf[i]=buf_any[i]
    lbufno = ucbufno/8
    lbuf = (c_ulong*long(lbufno))()
    libc.uchar2ulong(lbuf, ucbuf, ucbufno)
    return lbuf

def uint2uchar(buf_any, ucbufno):
    ucbuf = (c_ubyte*ucbufno)()
    ibufno = ucbufno/4
    ibuf = (c_uint*int(ibufno))()
    for i in range(0, ibufno):
        ibuf[i]= buf_any[i]
    libc.uint2uchar(ucbuf, ibuf, ucbufno)
    return ucbuf



