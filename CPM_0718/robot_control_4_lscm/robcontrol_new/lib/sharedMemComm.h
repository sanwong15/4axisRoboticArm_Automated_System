#ifndef __SHAREDMEMCOMM_H__
#define __SHAREDMEMCOMM_H__

#include "stdlib.h"
#include <stdio.h>
#include "fcntl.h"
#include "sys/mman.h"
#include "sys/stat.h"
#include "unistd.h"
#include "sys/types.h" 
#include <semaphore.h>
#include "errno.h"
#include "string.h"
#include <stdexcept>
#include <boost/python.hpp>

class sharedMem{
    public:
        sharedMem() = default;
        sharedMem(/*unsigned char* c,*/ const char *f, int i){
            /*sharedMemBuffer = c;*/
            fileName = f;
            sharedMemSize = i;
        }
        int create();
        int lock();
        int link();
        int writeData(unsigned char *content, int start, int length, int mode);
        int writeData(int index, unsigned char value);
        int readData(unsigned char *readBuffer, int start, int length, int mode);
        unsigned char readData(int index);
        int destroy();
    
    protected:
        unsigned char *sharedMemBuffer;
        const char *fileName;
        int sharedMemSize;
        union SEM_UC_POINTER{
            sem_t *sem;
            unsigned char *uc;
        };
};

class sharedMemPy : public sharedMem{
    
    public:
        sharedMemPy() = default;
        sharedMemPy(/*unsigned char* c,*/ const char *f, int i){
            /*sharedMemBuffer = c;*/
            fileName = f;
            sharedMemSize = i;
        }
        int writeData(boost::python::list content, int start, int length, int mode);
        int readData(boost::python::list readBuffer, int start, int length, int mode);
};
#endif
