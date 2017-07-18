#include "sharedMemComm.h"
#include "timer.h"
#include "realtime.h"
#define LOCK_NUM 6
#define READ_COUNT_NUM 2
#define SEM_SIZE 32
#define MIN_MEM_SIZE 16
#define SEM_TIMEOUT 500000 //time for sem_timedwait timeout,ns, changed to 500us 20160812

using namespace boost::python;

int sharedMem::create()
{
    int fileDescriptor;
    int fileSize = sharedMemSize + LOCK_NUM * SEM_SIZE + READ_COUNT_NUM * MIN_MEM_SIZE;

    fprintf(stderr,"%cFile descriptor size:%d\n",getlocaltime(), fileSize);
    shm_unlink(fileName);
    fileDescriptor = shm_open(fileName, O_RDWR | O_CREAT, S_IRWXU);

    // int result = lseek(fileDescriptor, fileSize*sizeof(unsigned char)-1, SEEK_SET);

    // if (result == -1){
    //     close(fileDescriptor);
    //     perror("Error calling lseek() to 'stretch' the file");
    //     return -1;
    // }
    // write(fileDescriptor, "", 1);
    if(ftruncate(fileDescriptor,fileSize*sizeof(unsigned char)) == -1)
    {
        fprintf(stderr,"%cftruncate failed\n",getlocaltime());
        return -1;
    }

    sharedMemBuffer = (unsigned char *)mmap(NULL, fileSize*sizeof(unsigned char), PROT_READ | PROT_WRITE, MAP_SHARED, fileDescriptor, 0);
    // int retval = mlock(sharedMemBuffer,fileSize*sizeof(unsigned char));                                                                                                   
    // if(retval != 0)
    //     printf("mlock failed\n");
    // else
    //     printf("mlcok succeed\n");
    // close(fileDescriptor);
    
    return 0;
}

int sharedMem::lock()
{
    union SEM_UC_POINTER mutex_rdcnt_1, rd_lock_1, wt_lock_1;
    union SEM_UC_POINTER mutex_rdcnt_2, rd_lock_2, wt_lock_2;
    unsigned char *readCount_1, *readCount_2;


    
    mutex_rdcnt_1.uc = &sharedMemBuffer[sharedMemSize];
    rd_lock_1.uc = &sharedMemBuffer[sharedMemSize + SEM_SIZE];
    wt_lock_1.uc = &sharedMemBuffer[sharedMemSize + 2 * SEM_SIZE];
    readCount_1 = &sharedMemBuffer[sharedMemSize + 3 * SEM_SIZE];

    mutex_rdcnt_2.uc = &sharedMemBuffer[sharedMemSize + MIN_MEM_SIZE + 3 * SEM_SIZE];
    rd_lock_2.uc = &sharedMemBuffer[sharedMemSize + MIN_MEM_SIZE + 4 * SEM_SIZE];
    wt_lock_2.uc = &sharedMemBuffer[sharedMemSize + MIN_MEM_SIZE + 5 * SEM_SIZE];
    readCount_2 = &sharedMemBuffer[sharedMemSize + MIN_MEM_SIZE + 6 * SEM_SIZE];

    sem_init(mutex_rdcnt_1.sem, 1, 1);
    sem_init(rd_lock_1.sem, 1, 1);
    sem_init(wt_lock_1.sem, 1, 1);


    *readCount_1 = 0;

    sem_init(mutex_rdcnt_2.sem, 1, 1);
    sem_init(rd_lock_2.sem, 1, 1);
    sem_init(wt_lock_2.sem, 1, 1);
 
    *readCount_2 = 0;
    
    return 0;
}

int sharedMem::link()
{
    int fileDescriptor;
    // int fileSize = sharedMemSize + LOCK_NUM * MIN_MEM_SIZE + READ_COUNT_NUM * MIN_MEM_SIZE;
    struct stat shmstat; 
    fileDescriptor = shm_open(fileName, O_RDWR, 0);
    
    if (fileDescriptor < 0){
        fprintf(stderr,"%cError: file %s does not exist, please create it first!\n",getlocaltime(), fileName);
        return -1;
    }
    
    if(fstat(fileDescriptor,&shmstat) == -1){
        fprintf(stderr,"%cget shared memory stat error\n",getlocaltime());
        return -1;
    }
    
    sharedMemBuffer = (unsigned char *)mmap(NULL, shmstat.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, fileDescriptor, 0);
    // close(fileDescriptor);

    return 0;
}


int sharedMem::writeData(unsigned char *content, int start, int length, int mode)
{
    union SEM_UC_POINTER rd_lock, wt_lock;
    struct timespec ts;
    float td;
    int s;
    // int sval,retval;


    if (mode == 0){
        rd_lock.uc = &sharedMemBuffer[sharedMemSize + SEM_SIZE];
        wt_lock.uc = &sharedMemBuffer[sharedMemSize + 2 * SEM_SIZE];
    } else if (mode == 1){
        rd_lock.uc = &sharedMemBuffer[sharedMemSize + MIN_MEM_SIZE + 4 * SEM_SIZE];
        wt_lock.uc = &sharedMemBuffer[sharedMemSize + MIN_MEM_SIZE + 5 * SEM_SIZE];
    } else {
        fprintf(stderr,"%cMode %d is not supported!", getlocaltime(),mode);
        return -1;
    }


    semtimeout(&ts,SEM_TIMEOUT);

    if(sem_timedwait(rd_lock.sem,&ts) == -1)
    {
        fprintf(stderr,"%c%s\n",getlocaltime(),strerror(errno) );
        return -1;
    }

    semtimeout(&ts,SEM_TIMEOUT);

    if(sem_timedwait(wt_lock.sem,&ts) == -1)
    {
        fprintf(stderr,"%c%s\n",getlocaltime(),strerror(errno) );
        sem_post(rd_lock.sem);
        return -1;
    }

    int index = 0;
    int i;

    for (  i = start; i < start + length; i++){
        sharedMemBuffer[i] = content[index];
        index++;
    }

    sem_post(wt_lock.sem);
    sem_post(rd_lock.sem);


    return 0;
}


int sharedMem::writeData(int index, unsigned char value)
{
    if (index >= sharedMemSize){
        throw std::invalid_argument("Index is out of boundary!");
        return -1;
    }
    sharedMemBuffer[index] = value;
    return 0;
}


int sharedMem::readData(unsigned char *readBuffer, int start, int length, int mode)
{
    union SEM_UC_POINTER mutex_rdcnt, rd_lock, wt_lock;
    unsigned char *readCount;
    struct timespec ts;


    if (start + length > sharedMemSize){
        fprintf(stderr,"%cError: start plus length is more than the size of shared memory!",getlocaltime());
        exit(-1);
    }

    if (mode == 0){
        mutex_rdcnt.uc = &sharedMemBuffer[sharedMemSize];
        rd_lock.uc = &sharedMemBuffer[sharedMemSize + SEM_SIZE];
        wt_lock.uc = &sharedMemBuffer[sharedMemSize + 2 * SEM_SIZE];
        readCount = &sharedMemBuffer[sharedMemSize + 3 * SEM_SIZE];
    } else if (mode == 1){
        mutex_rdcnt.uc = &sharedMemBuffer[sharedMemSize + MIN_MEM_SIZE + 3 * SEM_SIZE];
        rd_lock.uc = &sharedMemBuffer[sharedMemSize + MIN_MEM_SIZE + 4 * SEM_SIZE];
        wt_lock.uc = &sharedMemBuffer[sharedMemSize + MIN_MEM_SIZE + 5 * SEM_SIZE];
        readCount = &sharedMemBuffer[sharedMemSize + MIN_MEM_SIZE + 6 * SEM_SIZE];
    } else {
        fprintf(stderr,"%cMode %d is not supported!",getlocaltime(), mode);
        exit(-1);
    }

    semtimeout(&ts,SEM_TIMEOUT);

    if (sem_timedwait(rd_lock.sem,&ts) == -1){
        fprintf(stderr,"%crd_lock sem_wait: %s\n",getlocaltime(), strerror(errno));
        return -1;
    }

    // semtimeout(&ts,SEM_TIMEOUT);

    sem_wait(mutex_rdcnt.sem);
    // if (sem_timedwait(mutex_rdcnt.sem,&ts) == -1){
    //     fprintf(stderr,"%cmutex_rdcnt sem_wait: %s\n",getlocaltime(),strerror(errno));
    //     sem_post(rd_lock.sem);
    //     return -1;
    // }

   

    (*readCount)++;
    if (*readCount == 1){
        semtimeout(&ts,SEM_TIMEOUT);
        if (sem_timedwait(wt_lock.sem,&ts) == -1){
            fprintf(stderr,"%csem_wait: %s\n",getlocaltime(), strerror(errno));
            (*readCount)--;
            sem_post(mutex_rdcnt.sem);
            sem_post(rd_lock.sem);
            return -1;
        }
    }

    // if (*readCount > 1)
    //     printf("readCount = %d\n", *readCount); /* commented by jason 160105*/

    sem_post(mutex_rdcnt.sem);
    sem_post(rd_lock.sem);
    


    int index = 0;
    for (int i = start; i < start + length; i++){
        readBuffer[index] = sharedMemBuffer[i];
        index++;
    }

    //semtimeout(&ts,SEM_TIMEOUT);
    sem_wait(mutex_rdcnt.sem);
    // if (sem_timedwait(mutex_rdcnt.sem,&ts) == -1){
    //     fprintf(stderr,"%c%s\n", getlocaltime(),strerror(errno));
    //     return -1;
    // }
    (*readCount)--;
    if (*readCount == 0) 
        sem_post(wt_lock.sem);
    sem_post(mutex_rdcnt.sem);

    return 0;
}

unsigned char sharedMem::readData(int index)
{
    if (index >= sharedMemSize){
        throw std::invalid_argument("Index is out of boundary!");
    }
    return sharedMemBuffer[index];
}

int sharedMem::destroy()
{
    union SEM_UC_POINTER mutex_rdcnt_1, rd_lock_1, wt_lock_1;
    union SEM_UC_POINTER mutex_rdcnt_2, rd_lock_2, wt_lock_2;

    mutex_rdcnt_1.uc = &sharedMemBuffer[sharedMemSize];
    rd_lock_1.uc = &sharedMemBuffer[sharedMemSize + SEM_SIZE];
    wt_lock_1.uc = &sharedMemBuffer[sharedMemSize + 2 * SEM_SIZE];

    mutex_rdcnt_2.uc = &sharedMemBuffer[sharedMemSize + 1 + 3 * SEM_SIZE];
    rd_lock_2.uc = &sharedMemBuffer[sharedMemSize + 1 + 4 * SEM_SIZE];
    wt_lock_2.uc = &sharedMemBuffer[sharedMemSize + 1 + 5 * SEM_SIZE];

    int fd = open(fileName,O_CREAT | O_WRONLY,S_IWUSR);

    int retval = lseek(fd,0,SEEK_SET);
    if(retval == -1 ){
        fprintf(stderr,"%cseek file error\n",getlocaltime());
    }

    retval = write(fd,sharedMemBuffer,sharedMemSize);
    if(retval == -1 )
        fprintf(stderr,"%cwrite to file error\n",getlocaltime());
    else
        fprintf(stderr,"%cwrite %d bytes\n",getlocaltime(),retval );

    fprintf(stderr,"%cunlink shared memory and destroy sem\n",getlocaltime());

    sem_destroy(mutex_rdcnt_1.sem);
    sem_destroy(rd_lock_1.sem);
    sem_destroy(wt_lock_1.sem);
    sem_destroy(mutex_rdcnt_2.sem);
    sem_destroy(rd_lock_2.sem);
    sem_destroy(wt_lock_2.sem);
    
    int fileSize = sharedMemSize + 194; // where does 194 come from?
    munmap(sharedMemBuffer, fileSize * sizeof(unsigned char));
    // remove(fileName);
    shm_unlink(fileName);
    return 0;
}


int sharedMemPy::writeData(boost::python::list content, int start, int length, int mode)
{
    union SEM_UC_POINTER rd_lock, wt_lock;
    struct timespec ts;
    float td;
    int s;

    if (mode == 0){
        rd_lock.uc = &sharedMemBuffer[sharedMemSize + SEM_SIZE];
        wt_lock.uc = &sharedMemBuffer[sharedMemSize + 2 * SEM_SIZE];
    } else if (mode == 1){
        rd_lock.uc = &sharedMemBuffer[sharedMemSize + MIN_MEM_SIZE + 4 * SEM_SIZE];
        wt_lock.uc = &sharedMemBuffer[sharedMemSize + MIN_MEM_SIZE + 5 * SEM_SIZE];
    } else {
        fprintf(stderr,"%cMode %d is not supported!",getlocaltime(), mode);
        return -1;
    }

    semtimeout(&ts,SEM_TIMEOUT);

    if(sem_timedwait(rd_lock.sem,&ts) == -1)
    {
        fprintf(stderr,"%c%s\n",getlocaltime(),strerror(errno) );
        return -1;
    }

    // clock_gettime(CLOCK_REALTIME, &ts);
    // ts.tv_nsec+=10000;
    // tsnorm(&ts);
    semtimeout(&ts,SEM_TIMEOUT);

    if(sem_timedwait(wt_lock.sem,&ts) == -1)
    {
        fprintf(stderr,"%c%s\n",getlocaltime(),strerror(errno) );
        sem_post(rd_lock.sem);
        return -1;
    }
     
    // t1 = tic();
    int index = 0;
    int i;
    // t2 = toc(t2,&td);

    for (i = start; i < start + length; i++){
        sharedMemBuffer[i] = extract<unsigned char>(content[index]);
        index++;
    }
    sem_post(wt_lock.sem);
    sem_post(rd_lock.sem);

    return 0;
}

int sharedMemPy::readData(boost::python::list readBuffer, int start, int length, int mode)
{
    union SEM_UC_POINTER mutex_rdcnt, rd_lock, wt_lock;
    unsigned char *readCount;
    struct timespec ts;


    if (start + length > sharedMemSize){
        fprintf(stderr,"%cError: start plus length is more than the size of shared memory!",getlocaltime());
        exit(-1);
    }

    if (mode == 0){
        mutex_rdcnt.uc = &sharedMemBuffer[sharedMemSize];
        rd_lock.uc = &sharedMemBuffer[sharedMemSize + SEM_SIZE];
        wt_lock.uc = &sharedMemBuffer[sharedMemSize + 2 * SEM_SIZE];
        readCount = &sharedMemBuffer[sharedMemSize + 3 * SEM_SIZE];
    } else if (mode == 1){
        mutex_rdcnt.uc = &sharedMemBuffer[sharedMemSize + MIN_MEM_SIZE + 3 * SEM_SIZE];
        rd_lock.uc = &sharedMemBuffer[sharedMemSize + MIN_MEM_SIZE + 4 * SEM_SIZE];
        wt_lock.uc = &sharedMemBuffer[sharedMemSize + MIN_MEM_SIZE + 5 * SEM_SIZE];
        readCount = &sharedMemBuffer[sharedMemSize + MIN_MEM_SIZE + 6 * SEM_SIZE];
    } else {
        fprintf(stderr,"%cMode %d is not supported!",getlocaltime(), mode);
        exit(-1);
    }
    semtimeout(&ts,SEM_TIMEOUT);

    if (sem_timedwait(rd_lock.sem,&ts) == -1){
        fprintf(stderr,"%crd_lock sem_wait: %s\n",getlocaltime(), strerror(errno));
        return -1;
    }

    // semtimeout(&ts,SEM_TIMEOUT);

    // if (sem_timedwait(mutex_rdcnt.sem,&ts) == -1){
    //     fprintf(stderr,"%cmutex_rdcnt sem_wait: %s\n", getlocaltime(),strerror(errno));
    //     return -1;
    // }
    sem_wait(mutex_rdcnt.sem);
    

    (*readCount)++;
    if (*readCount == 1){
        semtimeout(&ts,SEM_TIMEOUT);
        if (sem_timedwait(wt_lock.sem,&ts) == -1){
            fprintf(stderr,"%csem_wait: %s\n",getlocaltime(), strerror(errno));
            (*readCount)--;
            sem_post(mutex_rdcnt.sem);
            sem_post(rd_lock.sem);
            return -1;
        }
    }

    // if (*readCount > 1)
    //     printf("readCount = %d\n", *readCount); /* commented by jason 160105*/

    sem_post(mutex_rdcnt.sem);
    sem_post(rd_lock.sem);
    


    int index = 0;
    for (int i = start; i < start + length; i++){
        readBuffer[index] = sharedMemBuffer[i];
        index++;
    }

    // semtimeout(&ts,SEM_TIMEOUT);

    // if (sem_timedwait(mutex_rdcnt.sem,&ts) == -1){
    //     fprintf(stderr,"%c%s\n", getlocaltime(),strerror(errno));
    //     return -1;
    // }
    sem_wait(mutex_rdcnt.sem);
    (*readCount)--;
    if (*readCount == 0) 
        sem_post(wt_lock.sem);
    sem_post(mutex_rdcnt.sem);

    return 0;
}


BOOST_PYTHON_MODULE(shm)
{
    class_<sharedMemPy>("SharedMem", init<const char*, int>())
        .def("create", &sharedMemPy::create)
        .def("destroy", &sharedMemPy::destroy)
        .def("link", &sharedMemPy::link)
        .def("lock", &sharedMemPy::lock)
        .def("readData", &sharedMemPy::readData)
        .def("writeData", &sharedMemPy::writeData)
        ;
}