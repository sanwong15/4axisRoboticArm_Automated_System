/*!
	This module defines function for ticking time

	Company: PI Electronic (H.K.) Ltd
	Author : Terence
	Date: 13 Jan 2015
*/
#ifndef __TIMER_H__
#define __TIMER_H__

#include "stdio.h"
#include "stdlib.h"
#include <time.h>

extern struct timespec t_on;
extern struct timespec t_off;
extern float ts_diff;
extern struct timespec t_loc;

//! tick current time
struct timespec tic();
// tick cuurent time and calculate the time difference with ts
/*!
	input:
		ts: time variable
		tdif: time difference between current time and ts
*/
struct timespec toc(struct timespec ts, float *tdif);

void semtimeout(struct timespec *ts,int timeout);

char getlocaltime(void);

#endif