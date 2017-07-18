/*!
	This module defines the basic API to access the FTDI device

	Company: PI Electronic (H.K.) Ltd
	Author : Terence
	Date: 13 Jan 2015
*/

#include "DataPacketStruct.h"

//!	combine 4 char to 1 float  
void uchar2float(float *fbuf, const unsigned char *cbuf, int bufno){
	union f2uc x;
	for (int i=0; i<bufno/4; i++){
		x.c[0] = cbuf[4*i];
    	x.c[1] = cbuf[4*i+1];
    	x.c[2] = cbuf[4*i+2];
    	x.c[3] = cbuf[4*i+3]; 
    	fbuf[i] = x.f;
  	}
}

void uchar2int(int *ibuf, const unsigned char *cbuf, int bufno){
	union i2uc x;
	for (int i=0; i<bufno/4; i++){
		x.c[0] = cbuf[4*i];
    	x.c[1] = cbuf[4*i+1];
    	x.c[2] = cbuf[4*i+2];
    	x.c[3] = cbuf[4*i+3]; 
    	ibuf[i] = x.ii;
  	}
}

void uchar2uint(unsigned int *ibuf, const unsigned char *cbuf, int bufno){
	union ui2uc x;
	for (int i=0; i<bufno/4; i++){
		x.c[0] = cbuf[4*i];
    	x.c[1] = cbuf[4*i+1];
    	x.c[2] = cbuf[4*i+2];
    	x.c[3] = cbuf[4*i+3]; 
    	ibuf[i] = x.ui;
  	}
}

void uchar2ulong(unsigned long *lbuf, const unsigned char *cbuf, int bufno){
	union ul2uc x;
	for (int i=0; i<bufno/8; i++){
		for(int j = 0; j<8; j++)
			x.c[j] = cbuf[8*i+j];
    	
    	lbuf[i] = x.ul;
  	}
}

void ulong2uchar(unsigned char *cbuf, const unsigned long* lbuf, int bufno){
	union ul2uc x;

	for (int i=0; i<bufno/8; i++){
		x.ul = lbuf[i];
    	cbuf[8*i] = x.c[0];
    	cbuf[8*i+1] = x.c[1];
    	cbuf[8*i+2] = x.c[2];
    	cbuf[8*i+3] = x.c[3];
    	cbuf[8*i+4] = x.c[4];
    	cbuf[8*i+5] = x.c[5];
    	cbuf[8*i+6] = x.c[6];
    	cbuf[8*i+7] = x.c[7];

  	}
}
/*
	seperate 1 float to 4 char
*/ 
void float2uchar(unsigned char *cbuf, float *fbuf, int bufno){
	union f2uc x;
	for (int i=0; i<bufno/4; i++){
		x.f = fbuf[i];
    	cbuf[4*i] = x.c[0];
    	cbuf[4*i+1] = x.c[1];
    	cbuf[4*i+2] = x.c[2];
    	cbuf[4*i+3] = x.c[3];
  	}
}

void int2uchar(unsigned char *cbuf, int *ibuf, int bufno){
	union i2uc x;
	for (int i=0; i<bufno/4; i++){
		x.ii = ibuf[i];
    	cbuf[4*i] = x.c[0];
    	cbuf[4*i+1] = x.c[1];
    	cbuf[4*i+2] = x.c[2];
    	cbuf[4*i+3] = x.c[3];
  	}
}

void uint2uchar(unsigned char *cbuf, unsigned int *ibuf, int bufno){
	union ui2uc x;
	for (int i=0; i<bufno/4; i++){
		x.ui = ibuf[i];
    	cbuf[4*i] = x.c[0];
    	cbuf[4*i+1] = x.c[1];
    	cbuf[4*i+2] = x.c[2];
    	cbuf[4*i+3] = x.c[3];
  	}
}

//! get a bit value from a byte
/*!
	input: 
		ctlbyte: control byte
		pos: bit position, start from 0

*/
unsigned char getbitvalue(unsigned char ctlbyte, int pos){
	return ( (ctlbyte >> pos)&1 );
}

//! Check the Rx data,
/*! 
	if pass return 0, if data corrupted return 1
*/
int check_Rx_data(const unsigned char *Rxbuf){

	unsigned char head, tail;
	unsigned short checksum = 0;
	union char2short Rxchecksum;	

	head = Rxbuf[0];
	Rxchecksum.c[1] = Rxbuf[Rx_DataPacketSize-3];	//!MSB
	Rxchecksum.c[0] = Rxbuf[Rx_DataPacketSize-2];	//!LSB
	tail = Rxbuf[Rx_DataPacketSize-1];	

	if (tail != 255){
		//printf("Error: Invalid Rx data: The tail of Rx data is not 255!\n");
		return 1;
	}

	if (head != 0) {
		//printf("Error: Invalid Rx data: The head of Rx data is not 0!\n");
		return 1;
	}


	for (int i=3; i<Rx_DataPacketSize-3; i++){
		checksum += Rxbuf[i];
	}

	if (Rxchecksum.sint != checksum){
		//printf("Error: Invalid Rx data: ");
		//printf("The check sum of data is not correct!");
		//printf("Rx checksum = %d, Calcualted checksum = %d!\n", 
		//Rxchecksum.sint, checksum);
		return 1;
	}

	return 0;

}

//! Check the control byte
/*! 
	if pass return 0, return 1 if adknowledge by DSP
*/
// int check_ctlbyte(unsigned char ctlbyte){

// 	unsigned char err_bit = getbitvalue(ctlbyte, 1);
// 	unsigned char err_disc_lsb = getbitvalue(ctlbyte, 2);
// 	unsigned char err_disc_msb = getbitvalue(ctlbyte, 3);

	
// 	// if ((err_disc_msb == 1) && (err_disc_lsb ==1)){
// 	// 	printf("AKW from DSP: Control Byte: checksum Error!\n");
// 	// 	return 1;
// 	// }
	
// 	// if((err_disc_msb == 0) && (err_disc_lsb ==1)){
// 	// 	printf("AKW from DSP: Control Byte: head or tail mismatch!\n");
// 	// 	return 1;
// 	// }

// 	// if((err_disc_msb == 1) && (err_disc_lsb ==0)){
// 	// 	printf("AKW from DSP: Control Byte: Time stamp mismatch!\n");
// 	// 	return 1;
// 	// }

// 	if (err_bit != 0) return 1;

// 	return 0;
// }

//! extract float data from Rxbuf
/*! 
	input:
		Rxbuf: Rx buffer
	output:
		data: data in float type	
*/
void extract_Rx_data(float *data, const unsigned char* Rxbuf){
	
	//! get the degree of freedom from the ctlbyte
	int dof = 0;	
	unsigned char dof_bit = getbitvalue(Rxbuf[2], 0);
	if ( dof_bit == 0) dof = 4;
	else dof = 6;

	uchar2float(data, Rxbuf+3, dof*4);

}

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
// void create_Tx_data(unsigned char *Txbuf, unsigned char ts, unsigned char ctlbyte, float *data, int no_of_float){
	
// 	char2short checksum;	//!< check sum variable

// 	Txbuf[0] = 0;	//!< head must be 0;
// 	Txbuf[1] = ts;	//!< current time stamp
// 	Txbuf[2] = ctlbyte;	//!< ctlbyte, 0

// 	if (4*no_of_float > 120){
// 		printf("your size of data is exceed the limit\n");
// 	}
// 	float2uchar(Txbuf+3, data, 4*no_of_float); //!< convert float into unsigned char

// 	//! calculate checksum of the data
// 	checksum.sint = 0;
// 	for (int i = 3; i < Tx_DataPacketSize-3; ++i)
// 	{
// 		checksum.sint += Txbuf[i];
// 	}

// 	Txbuf[Tx_DataPacketSize-3]=checksum.c[1];	//!MSB
// 	Txbuf[Tx_DataPacketSize-2]=checksum.c[0];	//!LSB
// 	Txbuf[Tx_DataPacketSize-1]=255; 	//!< end of Txbuf, must be 255

// }

//! Revised by Jason on 20150116, pass raw data directly, rather than float
void create_Tx_data(unsigned char *Txbuf, unsigned char ts, unsigned char ctlbyte, unsigned char *data, int no_of_char){
	
	union char2short checksum;	//!< check sum variable

	Txbuf[0] = 0;	//!< head must be 0;
	Txbuf[1] = ts;	//!< current time stamp
	Txbuf[2] = ctlbyte;	//!< ctlbyte, 0

	if (no_of_char > 136)//20150630 xiaoxia add "+4" for get g_clock
	{
		printf("your size of data is exceed the limit\n");
	}
	memcpy(Txbuf+3, data, no_of_char); //!< convert float into unsigned char

	//! calculate checksum of the data
	checksum.sint = 0;
	for (int i = 3; i < Tx_DataPacketSize-3; ++i)
	{
		checksum.sint += Txbuf[i];
	}

	Txbuf[Tx_DataPacketSize-3]=checksum.c[1];	//!MSB
	Txbuf[Tx_DataPacketSize-2]=checksum.c[0];	//!LSB
	Txbuf[Tx_DataPacketSize-1]=255; 	//!< end of Txbuf, must be 255

}
