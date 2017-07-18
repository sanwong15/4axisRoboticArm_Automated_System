/*!
	This module defines API function to make the program running with
	real-time ability base on POSIX sched.h

	Company: PI Electronic (H.K.) Ltd
	Author : Terence
	Date: 13 Jan 2015

	v1: Revise stack prefault function // Terence on 26 Aug 2015

*/
#ifndef __REALTIME_H__
#define __REALTIME_H__
#ifndef _GNU_SOURCE
#define _GNU_SOURCE // neccessaru for gcc complier to recognize the sched.h library
#endif

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h> // needed for memset
#include <sched.h>
#include <sys/mman.h>
#include <time.h>
#include "pthread.h"
#include <sys/time.h>	// needed for getrusage
#include <sys/resource.h>	// needed for getrusage
#include <limits.h>
#include <unistd.h>     // needed for sysconf(int name);
#include <malloc.h>


#define NSEC_PER_SEC 1000000000 //!< define 1s = 1E9 ns
void set_dedicated_cpu(int mask);
void set_dedicated_cpu_thread(pthread_t tid, int mask);
//! set the realtime priority to the process
/*!
	input: 
		pid: process ID, can get it by calling getpid()
		priority: range from 0-99,
				  99 is he most realtime priority, 0 is the lowest   
*/
void make_realtime (int pid, int priority);
//! set the realtime priority to the thread
/*!
	input: 
		tid: thread ID
		priority: range from 0-99,
				  99 is he most realtime priority, 0 is the lowest   
*/
void make_realtime_thread(pthread_t tid, int priority);

//!	Stack prefault function
/*!
	input:
		max_safe_stack: size of prefault stack in byte
*/
void stack_prefault(int max_safe_stack) ;


// lock the physical memory page to prevent stack prefualt // Add by Terence on 26 Aug 2015
void lock_mem_page(void);

// Check the stack prefault times // Added by Terence on 26 Aug 2015
void pagefault_count(const char* logtext);

//!	normalize the time from 1E9 ns to 1s
void tsnorm(struct timespec *ts);
//!	nanosleep function
void custom_nanosleep(int nanocount);


#endif

