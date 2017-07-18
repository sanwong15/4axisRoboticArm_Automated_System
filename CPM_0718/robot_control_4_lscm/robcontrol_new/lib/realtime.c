/*!
	This module defines API function to make the program running with
	real-time ability base on POSIX sched.h

	Company: PI Electronic (H.K.) Ltd
	Author : Terence
	Date: 13 Jan 2015

	v1: Revise stack prefault function // Terence on 26 Aug 2015
*/
#include "realtime.h"
#include "timer.h"

void set_dedicated_cpu(int mask){

	//unsigned long mask = 1; /* processor 0 */
 
 	cpu_set_t cpuset;
	CPU_ZERO(&cpuset);
	CPU_SET(mask, &cpuset); // cpu 0
    /* bind process to processor 0 */
    if (sched_setaffinity(0, sizeof(cpuset), &cpuset) < 0) {
        fprintf(stderr,"%csched_setaffinity",getlocaltime());
    }
    fprintf(stderr,"%cCurrently running on CPU core: %d!\n",getlocaltime(), sched_getcpu());
 
}

void set_dedicated_cpu_thread(pthread_t tid, int mask){

	//unsigned long mask = 1; /* processor 0 */
 
 	cpu_set_t cpuset;
	CPU_ZERO(&cpuset);
	CPU_SET(mask, &cpuset); // cpu 0
    /* bind thread to processor 0 */
    if (pthread_setaffinity_np(tid, sizeof(cpuset), &cpuset) < 0) {
        fprintf(stderr,"%cpthread_setaffinity_np",getlocaltime());
    }
    fprintf(stderr,"Currently running on CPU core: %d!\n", sched_getcpu());
 
}

//! set the realtime priority to the process
/*!
	input: 
		pid: process ID, can get it by calling getpid()
		priority: range from 0-99,
				  99 is he most realtime priority, 0 is the lowest   
*/
void make_realtime(int pid, int priority)
{

	fprintf(stderr,"%cgetpid () = %d\n",getlocaltime(), pid);	//!< print out the pid
	
	struct sched_param param = { priority }; //!< define the priority
	//! set the priority
	if (sched_setscheduler (pid, SCHED_FIFO, &param) < 0){
		fprintf (stderr,"%ccould not get real-time priority",getlocaltime());
		exit (1);
	}
	//! lock the process in memory
	// if (mlockall (MCL_CURRENT | MCL_FUTURE) < 0){
	// 	perror ("could not lock process in memory");
	// 	exit (2);
 //    }
}

//! set the realtime priority to the thread
/*!
	input: 
		tid: thread ID
		priority: range from 0-99,
				  99 is he most realtime priority, 0 is the lowest   
*/
void make_realtime_thread(pthread_t tid, int priority){
	// printf("thread ID = %d\n", (unsigned int)tid);	//!< print out the pid
	struct sched_param param = { priority }; //!< define the priority
	//! set the priority
	if (pthread_setschedparam (tid, SCHED_FIFO, &param) < 0){
		fprintf (stderr,"%ccould not get real-time priority for the thread",getlocaltime());
		exit (1);
	}
	//! lock the process in memory
	// if (mlockall (MCL_CURRENT | MCL_FUTURE) < 0){
	// 	perror ("could not lock process in memory");
	// 	exit (2);
 //    }

}

//!	Stack prefault function
/*!
	input:
		max_safe_stack: size of prefault stack in byte
*/

void stack_prefault(int max_safe_stack){

	unsigned char *dummy;
	dummy = (unsigned char*)malloc(max_safe_stack*sizeof(unsigned char));
	memset(dummy, 0, max_safe_stack);

	// New stack prefault code // Add by Terence on 26 Aug 2015

	/* buffer will now be released. As Glibc is configured such that it 
         never gives back memory to the kernel, the memory allocated above is
         locked for this process. All malloc() and new() calls come from
         the memory pool reserved and locked above. Issuing free() and
         delete() does NOT make this locking undone. So, with this locking
         mechanism we can build C++ applications that will never run into
         a major/minor pagefault, even with swapping enabled. */
    free(dummy);
}

// lock the physical memory page in RAM to prevent stack prefualt // Add by Terence on 26 Aug 2015
void lock_mem_page(void){

    /* Now lock all current and future pages 
         from preventing of being paged  */
    if (mlockall(MCL_CURRENT | MCL_FUTURE))
        fprintf(stderr,"%cmlockall failed:",getlocaltime());
   
    /* Turn off malloc trimming.*/
    mallopt(M_TRIM_THRESHOLD, -1);
   
    /* Turn off mmap usage. */
    mallopt(M_MMAP_MAX, 0);
}

// Check the stack prefault times // Added by Terence on 26 Aug 2015
void pagefault_count(const char* logtext)
{
    static int last_majflt = 0, last_minflt = 0;
    struct rusage usage;
 
    getrusage(RUSAGE_SELF, &usage);
 
    fprintf(stderr,"%c%-30.30s: Pagefaults, Major:%ld, " \
           "Minor:%ld \n", getlocaltime(),logtext,
           usage.ru_majflt - last_majflt, usage.ru_minflt - last_minflt);
      
    last_majflt = usage.ru_majflt; 
    last_minflt = usage.ru_minflt;
 }


//!	normalize the time from 1E9 ns to 1s
void tsnorm(struct timespec *ts){
	while (ts->tv_nsec >=NSEC_PER_SEC){
		ts->tv_nsec -= NSEC_PER_SEC;
		ts->tv_sec++;
	}
}

//!	nanosleep function
void custom_nanosleep(int nanocount){
		struct timespec ts;
		clock_gettime(CLOCK_MONOTONIC, &ts);
		ts.tv_nsec+=nanocount;
		tsnorm(&ts);
		clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, NULL);
}


