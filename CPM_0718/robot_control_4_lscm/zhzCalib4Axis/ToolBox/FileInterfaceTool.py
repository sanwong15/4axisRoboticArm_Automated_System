#!/usr/bin/python2.7
# -*- coding:utf-8 -*-
__author__ = 'hkh & lh'
__date__ = '23/08/2016'
__version__ = '2.0'


import os
import sys
try:
    import yaml
except ImportError:
    print 'Warning: please install yaml,or some function will occur error!'
try:
    import pickle as pk
except ImportError:
    print 'Warning: please install pickle,or some function will occur error!'
try:
    import numpy as np
except ImportError:
    print 'Warning: please install numpy,or some function will occur error!'
try:
    import xlwt
except ImportError:
    print 'Warning: please install xlwt,or some function will occur error!'

import glob
import subprocess
import multiprocessing


def saveArrayToXlsx(arrayLike, xlsxPath, sheetName='A', color='black', numFormatStr='#,###0.000'):
    DataArray = np.array(arrayLike)
    if DataArray.ndim > 2:
        raise ValueError, 'the dimension of arrayLike must <= 2'
    if DataArray.ndim == 1:
        DataArray = DataArray.reshape(1, -1)
    Style = xlwt.easyxf('font: name Times New Roman, color-index %s' % color,
                         num_format_str='%s' % numFormatStr)
    MyWorkBook = xlwt.Workbook()
    MySheet = MyWorkBook.add_sheet(sheetName)
    for row in range(DataArray.shape[0]):
        for col in range(DataArray.shape[1]):
            MySheet.write(row, col, DataArray[row, col], Style)
    MyWorkBook.save(xlsxPath)

def excuseCommandBase(command, shell=False):
    try:
        RetCode = subprocess.call(command, shell=shell)
        if RetCode < 0:
            print >>sys.stderr, "Child was terminated by signal", -RetCode
            return -RetCode
        else:
            print >>sys.stderr, "Child returned", RetCode
            return RetCode
    except OSError as e:
        print >>sys.stderr, "Execution failed:", e
        return -1

def excuseCommand(command, shell=False, thread=False):
    if thread:
        my_pipe = subprocess.Popen(command, stdout=subprocess.PIPE, shell=shell)
        my_process = multiprocessing.Process(target=my_pipe.communicate, args=())
        my_process.daemon = True
        my_process.start()
        return my_pipe, my_process, None
        # return my_process
    else:
        my_pipe = subprocess.Popen(command, stdout=subprocess.PIPE, shell=shell)
        Output, Err = my_pipe.communicate()
        return my_pipe, Output, Err

def checkCommandOutput(command):
    Output = subprocess.check_output(command, shell=True)
    return Output

def checkCommandOutput_PIPE(command):
    my_pipe = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    Output, Err = my_pipe.communicate()
    return Output, Err

def saveFile(fileName, data, method='w'):
    with open(fileName, method) as file:
        return file.writelines(data.__str__())

def loadYaml(fileName, method='r'):
    with open(fileName, method) as file:
        return  yaml.load(stream=file)

def loadAllYaml(fileName, method='r'):
    with open(fileName, method) as file:
        return yaml.load_all(stream=file)

def dumpYaml(fileName, data, method='w'):
    with open(fileName, method) as file:
        yaml.dump(data=data, stream=file)

def dumpAllYaml(data, fileName, method='w'):
    with open(fileName, method) as file:
        yaml.dump_all(documents=data, stream=file)

def pkLoad(fileName, method='r'):
    with open(fileName, method) as File:
        return pk.load(File)

def isExist(fileName):
    return os.path.exists(fileName)

def createFile(fileName):
    if not isExist(fileName):
        os.system('mkdir ' + fileName)

def globPath(path):
    return glob.glob(path)

def joinPath(path, paths):
    return os.path.join(path, paths)

def absPath(path):
    return os.path.abspath(path)

def isFile(path):
    if os.path.isfile(path):
        return True
    return False

def isDir(path):
    if os.path.isdir(path):
        return True
    return False


if __name__ == '__main__':
    print checkCommandOutput('ls')
    # import xlrd
    # import numpy as np
    # workbook = xlrd.open_workbook("b.xlsx")
    # sheet = workbook.sheet_by_index(0)
    #
    # Rows = []
    # for rowx in range(sheet.nrows):
    #     cols = sheet.row_values(rowx)
    #     Rows.append(cols)
    #     print(cols)
    # Data = np.array(Rows)
    # print Data[7, :]
    # print float(Data[7, 1])
    # Path = r'./'
    # print globPath(Path)
    # s = np.ones((3, 4))
    # s = [1, 2, 3]
    # saveArrayToXlsx(s, './save.xlsx')
    # import openpyxl as px
    # W = px.load_workbook('b.xlsx', use_iterators = True)
    # W._Workbook__write_only = True
    # W._Workbook__read_only = False
    # p = W.get_sheet_by_name(name = 'Sheet1')
    #
    # # p = W.get_sheet_by_name(name = u'工作表1')
    #
    # a=[]
    #
    # for row in p.iter_rows():
    #     for k in row:
    #         # print k.internal_value
    #         a.append(k.internal_value)
    #
    # # print a
    # # convert list a to matrix (for example 5*6)
    # Datas = np.array(a).reshape(-1, 10)
    # # print Datas[8, :]
    # print Datas[7, 1:4]
    # Datas[7, 0] = 0L
    #
    # # W.save('aaa.xlsx')
    #
    # # WW = px.Workbook()
    # # WW.
    # # WW._add_sheet(p)
    # # WW.save('aaa.xlsx')
    # # aa = Datas
    # # # print aa
    # # # save matrix aa as xlsx file
    # # WW = px.Workbook()
    # # pp = WW.get_active_sheet()
    # # pp.title='NEW_DATA'
    # #
    # # f={'A':0,'B':1,'C':2,'D':3,'E':4,'F':5}
    # # aa[7, 1] = 100L
    # # aa[7, 2] = 500L
    # # aa[7, 3] = 600L
    # # aa[7, 4] = 700L
    # # #insert values in six columns
    # # for (i,j) in f.items():
    # #     for k in np.arange(1,len(aa)+1):
    # #         pp.cell('%s%d'%(i,k)).value=aa[k-1][j]
    # #
    # # WW.save('newfilname.xlsx')
