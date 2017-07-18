#!/bin/bash

g++ -I/usr/local/include -I/usr/include/python2.7 timer.c realtime.c sharedMemComm.cpp -fPIC -lpython2.7 -lboost_python -lrt -lpthread -shared -o shm.so

gcc -std=c99 -I./ -I/usr/local/include DataPacketStruct.c -fPIC -shared -o libDataConv.so