/*!
	This module defines the structure of Rx and Tx data from and to DSP.
	
	Structure of Rx Data:
		byte 0: must be 0
		byte 1: time stamp 
		byte 2: control byte
		byte[3..60]: 24 bytes data + 12 bytes IO state
		byte[61..62]: check sum of byte[3..60]
		byte[63]: must be 255

	Structure of Tx data:
		byte 0: must be 0
		byte 1: time stamp 
		byte 2: control byte
		byte[3..156]: 120 bytes data + 12 bytes IO state
		byte[157..158]: check sum of byte[3..156]
		byte[159]: must be 255

	Company: PI Electronic (H.K.) Ltd
	Author : Terence
	Date: 13 Jan 2015
*/
#ifndef __DATAPACKETSTRUCT_H__
#define __DATAPACKETSTRUCT_H__

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include "WinTypes.h"


#define Tx_DataPacketSize 160	//!< send 160 bytes out
#define Rx_DataPacketSize 64	//!< recieved 64 bytes from DSP

//! define a union for 2 8-bit unsigned char and 16-bit unsigned short
union char2short{
	unsigned short sint;
	unsigned char c[2];
};

//! define a union for 4 8-bit unsigned char and 32-bit float
union f2uc{
	float f;
	unsigned char c[4]; 
};

//! define a union for 4 8-bit unsigned char and 32-bit integer
union i2uc{
	int ii;
	unsigned char c[4]; 
};

union ui2uc{
	unsigned int ui;
	unsigned char c[4];
};

union ul2uc{
	unsigned long ul;
	unsigned char c[8];
};
//!	combine 4 char to 1 float  
/*!
	input:
		cbuf: 	unsigned char array with size bufno
		bufno: 	no of unsigned char, must be multiple of 4
	output:
		fbuf: 	float array with size bufno/4
*/
void uchar2float(float *fbuf, const unsigned char *cbuf,  int bufno);
//! seperate 1 float to 4 char
/*!
	input:
		fbuf: 	float array with size bufno/4
		bufno: 	no of unsigned char, must be multiple of 4
	output:
		cbuf: 	unsigned char array with size bufno
*/
void float2uchar(unsigned char *cbuf, float *fbuf, int bufno);
//! combine 4 char to 1 integer
void uchar2int(int *ibuf, const unsigned char *cbuf, int bufno);

void uchar2uint(unsigned int *ibuf, const unsigned char *cbuf,int bufno);
//! separate 1 integer to 4 char
void int2uchar(unsigned char *cbuf, int *ibuf, int bufno);

void uchar2uint(unsigned int *ibuf, const unsigned char *cbuf, int bufno); // added by jason 20150212

void uint2uchar(unsigned char *cbuf, unsigned int *ibuf, int bufno); // added by jason 20150212

void uchar2ulong(unsigned long *lbuf, const unsigned char *cbuf, int bufno); // added by jason 20150318

void ulong2uchar(unsigned char *cbuf, const unsigned long* lbuf, int bufno);
//! get a bit value from a byte
/*!
	input: 
		ctlbyte: control byte
		pos: bit position, start from 0

*/
unsigned char getbitvalue(unsigned char ctlbyte, int pos);
//! Check the Rx data,
/*! 
	if pass return 0, if data corrupted return 1
*/
int check_Rx_data(const unsigned char *Rxbuf);
//! Check the control byte
/*! 
	if pass return 0, return 1 if adknowledge by DSP
*/
int check_ctlbyte(unsigned char ctlbyte);
//! extract float data from Rxbuf
/*! 
	input:
		Rxbuf: Rx buffer
	output:
		data: data in float type	
*/
void extract_Rx_data(float *data, const unsigned char *Rxbuf);
//! create Tx data from float data
/*! 
	input:
		ts: time stamp, same as received current time stamp
		ctlbyte: control byte 
		data: data going to be converted into unsigned char
		no_of_float: size of data
	output:
		Txbuf: data in unsigned char type	
*/
//! original from Terence:
//void create_Tx_data(unsigned char *Txbuf, unsigned char ts, unsigned char ctlbyte, float *data, int no_data_set);
//! Revised by Jason on 20150116, pass raw data directly, rather than float
void create_Tx_data(unsigned char *Txbuf, unsigned char ts, unsigned char ctlbyte, unsigned char *data, int no_of_char);
#endif
