#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

__author__ = 'hkh'
__data__ = '07/08/2015'

try:
    import cv2
    import numpy as np
except:
    pass

import copy

ROI_TYPE_XYWH = 0L
ROI_TYPE_XYXY = 8L
ROI_TYPE_ROTATED = 64L

ROI_CVT_XYXY2XYWH = 0L
ROI_CVT_XYWH2XYXY = 8L

def gamaTransform(Src, Gama):
    assert 2 == Src.ndim

    C = 255.0 / 255 ** Gama
    if Gama > 1:
        Src = np.uint64(Src)
    return np.uint8(C * (Src ** Gama))

def contrastStretch(gray, min=0):
    Hist = cv2.calcHist([gray], [0], None, [256], [0.0, 256.0])
    IdxMin = 0
    for i in xrange(Hist.shape[0]):
        if Hist[i] > min:
            IdxMin = i
            break
    IdxMax = 0
    for i in xrange(Hist.shape[0]):
        if Hist[255-i] > min:
            IdxMax = 255 - i
            break
    _, gray = cv2.threshold(gray, IdxMax, IdxMax, cv2.THRESH_TRUNC)
    gray = ((gray >= IdxMin) * gray) + ((gray < IdxMin) * IdxMin)
    Res = np.uint8(255.0 * (gray - IdxMin) / (IdxMax - IdxMin))
    return Res

def findMaxAreaContours(contours, num=1):
    ContoursNum = len(contours)
    assert (0 != ContoursNum)

    MaxAreaContoursIndex = []
    Times = 0
    while Times < num:
        for i in xrange(ContoursNum - 1 - Times):
            if cv2.contourArea(contour=contours[i]) > cv2.contourArea(contour=contours[i+1]):
                Temp = contours[i]
                contours[i] = contours[i+1]
                contours[i+1] = Temp
        MaxAreaContoursIndex.append(ContoursNum - 1 - Times)
        Times += 1
    return MaxAreaContoursIndex

def findMaxBoundBox(contours, num=1):
    ContoursNum = len(contours)
    assert (0 != ContoursNum)

    MaxIndex = []
    Times = 0
    while Times < num:
        for i in xrange(ContoursNum - 1 - Times):
            _, _, w1, h1 = cv2.boundingRect(contours[i])
            _, _, w2, h2 = cv2.boundingRect(contours[i+1])
            if w1*h1 > w2*h2:
                Temp = contours[i]
                contours[i] = contours[i+1]
                contours[i+1] = Temp
        MaxIndex.append(ContoursNum - 1 - Times)
        Times += 1
    return MaxIndex

def splitImageWithHSV(src):
    HSV = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    H = HSV[:,:,0]
    B = np.uint8(((H >=90) & (H < 150)) * 255)
    G = np.uint8(((H >=30) & (H < 90)) * 255)
    R = np.uint8(((H >= 150) | (H < 30)) * 255)
    img = cv2.merge([B, G, R])
    return img

def splitImgColorWithHSV(src, chanel):
    assert chanel in 'bgr', 'must input one of b,g,r'

    HSV = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    H = HSV[:,:,0]
    if 'b' == chanel:
        return np.uint8(((H >=90) & (H < 150)) * 255)
    elif 'g' == chanel:
        return np.uint8(((H >=30) & (H < 90)) * 255)
    elif 'r' == chanel:
        return np.uint8(((H >= 150) | (H < 30)) * 255)

def calcGrayGravity(gray):
    # calculate gravity of the gray image.
    assert gray.ndim == 2, "must input a gary_img"


    Row, Col = gray.shape
    GraySum = np.sum(gray)
    if GraySum != 0:
        # np.sum(img, 0),叠加为一行
        # np.sum(img, 1),叠加为一列
        SumX = np.sum(gray, 0) * (np.array(range(Col)))
        SumY = np.sum(gray, 1) * (np.array(range(Row)))
        GravityX = np.sum(SumX) / GraySum
        GravityY = np.sum(SumY) / GraySum
    else:
        GravityX, GravityY = 0.0, 0.0
    return GravityX, GravityY

def getHistImg(imgOrHist):
    """
    :param imgOrHist:
    :return:
    """
    if 1 != imgOrHist.shape[1]:
        Hist = cv2.calcHist(images=[imgOrHist], channels=[0], mask=None, histSize=[256], ranges=[0.0, 256.0])
    else:
        Hist = imgOrHist
    MaxVal = np.max(Hist)
    # init show hist image
    HistImg = np.zeros((Hist.shape[0], Hist.shape[0]))
    Hpt = Hist.shape[0] * 0.9 / MaxVal
    for i, value in enumerate(Hist):
        draw_point = (i, 255 - value * Hpt)
        cv2.line(HistImg, draw_point, (i, 0), 255, 1)
    return HistImg

def __cvtRoi2xyxy(roi, roiType):
    if ROI_TYPE_XYWH == roiType:
        return cvtRoi(roi=roi, flag=ROI_CVT_XYWH2XYXY)
    elif ROI_TYPE_XYXY == roiType:
        return copy.copy(roi)
    else:
        raise ValueError, 'flag is wrong!!!'

def cvtRoi(roi, flag):
    """
    convert roi type
    :param roi: list or ndarray
    :param flag: ROI_CVT_XYXY2XYWH or ROI_CVT_XYWH2XYXY
    :return: roi (xyxy or xywh,depends on what you set)
    """
    x0, y0, c, d = roi
    if ROI_CVT_XYWH2XYXY == flag:
        x1 = x0 + c
        y1 = y0 + d
        return [x0, y0, x1, y1]
    elif ROI_CVT_XYXY2XYWH == flag:
        w = c - x0
        h = d - y0
        return [x0, y0, w, h]
    else:
        raise ValueError, 'flag is wrong!!!'

def getRoiImg(img, roi, roiType, copy=True):
    """
    :param img: gray image or BGR image
    :param roi: list or ndarray
    :param roiType: flag - ROI_TYPE_XYWH or ROI_TYPE_XYXY
    :return: Roi image
    """
    Roi_xyxy = __cvtRoi2xyxy(roi, roiType)
    if Roi_xyxy[0] < 0:
        Roi_xyxy[0] = 0
    if Roi_xyxy[1] < 0:
        Roi_xyxy[1] = 0
    if 3 == img.ndim:
        if copy:
            RoiImg = img[Roi_xyxy[1]:Roi_xyxy[3], Roi_xyxy[0]:Roi_xyxy[2], :].copy()
        else:
            RoiImg = img[Roi_xyxy[1]:Roi_xyxy[3], Roi_xyxy[0]:Roi_xyxy[2], :]
    else:
        if copy:
            RoiImg = img[Roi_xyxy[1]:Roi_xyxy[3], Roi_xyxy[0]:Roi_xyxy[2]].copy()
        else:
            RoiImg = img[Roi_xyxy[1]:Roi_xyxy[3], Roi_xyxy[0]:Roi_xyxy[2]]
    Offset_2x1 = np.array(Roi_xyxy[:2]).reshape(2, 1)
    return Offset_2x1, RoiImg

def drawRoi(img, roi, roiType, color, thickness=2, lineType=1, shift=0, offset=(0,0)):
    """
    draw roi(rectangle) in img
    :param img: gray image or BGR image
    :param roi: list or ndarray
    :param roiType: flag - ROI_TYPE_XYWH or ROI_TYPE_XYXY
    :param color: plot color you want
    :param thickness: roi(rectangle)'s thickness
    :return: None
    """
    Offset = np.array(offset).ravel()
    if roiType == ROI_TYPE_ROTATED:
        Points_4x2 = np.array(roi)
        if Points_4x2.shape[1] == 4:
            Points_4x2 = Points_4x2.T.reshape(4, 2)
        for i in range(3, -1, -1):
            drawLine(img, Points_4x2[i]+Offset[0], Points_4x2[i-1]+Offset[1], color=color, thickness=thickness, shift=shift)
    else:
        Roi_xyxy = __cvtRoi2xyxy(roi, roiType)
        Roi_xyxy[0] += Offset[0]
        Roi_xyxy[1] += Offset[1]
        Roi_xyxy[2] += Offset[0]
        Roi_xyxy[3] += Offset[1]
        cv2.rectangle(img, (int(Roi_xyxy[0]), int(Roi_xyxy[1])), (int(Roi_xyxy[2]), int(Roi_xyxy[3])), color,
                      thickness=thickness, lineType=lineType, shift=shift)

def drawPoints(img, pts_2xn, color, radius=1, thickness=-1, offset=(0,0), shift=0):
    """
    draw points(circles) in img
    :param img: gray image or BGR image
    :param pts_2xn: 2xn ndarray
    :param color: plot color you want
    :param radius: points(circles)'s radius
    :param thickness: points(circles)'s thickness
    :return: None
    """
    Offset = np.array(offset).ravel()
    for idx in range(pts_2xn.shape[1]):
        cv2.circle(img, (int(pts_2xn[0, idx]+Offset[0]), int(pts_2xn[1, idx]+Offset[1])), radius, color, thickness, shift=0)

def drawLine(img, point1, point2, color, thickness=2, shift=0):
    """
    draw line in img
    :param img: gray image or BGR image
    :param point1: line's first point - list, tuple or ndarray
    :param point2: line's second point - list, tuple or ndarray
    :param color: line's color you want
    :param thickness: line's thickness
    :return: None
    """
    Point1 = np.array(point1).ravel()
    Point2 = np.array(point2).ravel()
    cv2.line(img=img, pt1=(int(Point1[0]), int(Point1[1])),
             pt2=(int(Point2[0]), int(Point2[1])), color=color, thickness=thickness, shift=shift)

def inRoi(pt, roi, roiType):
    Point = np.array(pt).ravel()
    if roiType == ROI_TYPE_ROTATED:
        Points_4x2 = np.array(roi)
        if Points_4x2.shape[1] == 4:
            Points_4x2 = Points_4x2.T.reshape(4, 2)
        Contour_4x1x2 = Points_4x2.reshape(4, 1, 2)
        testResult = cv2.pointPolygonTest(Contour_4x1x2, tuple(Point), False)
        if testResult == -1:
            return False
        else:
            return True
    else:
        Roi_xyxy = __cvtRoi2xyxy(roi, roiType)
        if Roi_xyxy[0] <= Point[0] < Roi_xyxy[2] and Roi_xyxy[1] <= Point[1] < Roi_xyxy[3]:
            return True
    return False


if __name__ == '__main__':
    Img = np.zeros((10, 10), np.uint8)
    Roi_xyxy = [2, 2, 5, 5]
    Roi_xywh = [2, 2, 3, 3]
    CvtRoi_xywh = cvtRoi(Roi_xyxy, flag=ROI_CVT_XYXY2XYWH)
    CvtRoi_xyxy = cvtRoi(Roi_xywh, flag=ROI_CVT_XYWH2XYXY)
    print 'xyxy: ', Roi_xyxy, 'convert to xywh->', CvtRoi_xywh, 'convert back->', cvtRoi(CvtRoi_xywh, ROI_CVT_XYWH2XYXY)
    print 'xywh: ', Roi_xywh, 'convert to xyxy->', CvtRoi_xyxy, 'convert back->', cvtRoi(CvtRoi_xyxy, ROI_CVT_XYXY2XYWH)

    _, RoiImg_xyxy = getRoiImg(Img, Roi_xyxy, roiType=ROI_TYPE_XYXY)
    print 'RoiImg_xyxy shape: ', RoiImg_xyxy.shape
    _, RoiImg_xywh = getRoiImg(Img, Roi_xywh, roiType=ROI_TYPE_XYWH)
    print 'RoiImg_xywh shape: ', RoiImg_xywh.shape

    Point = [0, 0]
    print Point, 'in roi_xyxy?', inRoi(Point, Roi_xyxy, ROI_TYPE_XYXY), '\t/\t in roi_xywh?', inRoi(Point, Roi_xywh, ROI_TYPE_XYWH)
    Point = [2, 2]
    print Point, 'in roi_xyxy?', inRoi(Point, Roi_xyxy, ROI_TYPE_XYXY), '\t/\t in roi_xywh?', inRoi(Point, Roi_xywh, ROI_TYPE_XYWH)
    Point = [3, 3]
    print Point, 'in roi_xyxy?', inRoi(Point, Roi_xyxy, ROI_TYPE_XYXY), '\t/\t in roi_xywh?', inRoi(Point, Roi_xywh, ROI_TYPE_XYWH)
    Point = [4, 4]
    print Point, 'in roi_xyxy?', inRoi(Point, Roi_xyxy, ROI_TYPE_XYXY), '\t/\t in roi_xywh?', inRoi(Point, Roi_xywh, ROI_TYPE_XYWH)
    Point = [5, 5]
    print Point, 'in roi_xyxy?', inRoi(Point, Roi_xyxy, ROI_TYPE_XYXY), '\t/\t in roi_xywh?', inRoi(Point, Roi_xywh, ROI_TYPE_XYWH)

    cv2.namedWindow('Roi2', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Roi1', cv2.WINDOW_NORMAL)
    Img1 = np.zeros((100, 100), np.uint8)
    Img2 = np.zeros((100, 100), np.uint8)
    RotatedRoi1 = [[10, 10],
                  [50, 20],
                  [70, 70],
                  [20, 30]]
    RotatedRoi2 = [[10, 50, 70, 20],
                  [10, 20, 70, 30]]
    print 'RotatedRoi:', RotatedRoi1
    Point_2x1 = np.array([10, 20]).reshape(2, 1)
    drawPoints(Img1, Point_2x1, 255)
    print Point_2x1.ravel(), 'in RotatedRoi?', inRoi(Point_2x1, RotatedRoi1, ROI_TYPE_ROTATED)
    Point_2x1 = np.array([10, 10]).reshape(2, 1)
    drawPoints(Img1, Point_2x1, 255)
    print Point_2x1.ravel(), 'in RotatedRoi?', inRoi(Point_2x1, RotatedRoi1, ROI_TYPE_ROTATED)
    Point_2x1 = np.array([70, 70]).reshape(2, 1)
    drawPoints(Img1, Point_2x1, 255)
    drawPoints(Img1, Point_2x1, 255, offset=(-10, 10))
    print Point_2x1.ravel(), 'in RotatedRoi?', inRoi(Point_2x1, RotatedRoi1, ROI_TYPE_ROTATED)
    drawRoi(Img1, RotatedRoi1, ROI_TYPE_ROTATED, color=255)
    drawRoi(Img2, RotatedRoi2, ROI_TYPE_ROTATED, color=255, offset=(-50, -50))
    Contours, _ = cv2.findContours(image=Img2.copy(), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    Rect = cv2.minAreaRect(Contours[0])
    Box = cv2.cv.BoxPoints(Rect)
    BoxImg = np.zeros((200, 200), np.uint8)
    drawRoi(Img2, Box, ROI_TYPE_ROTATED, color=255)
    drawRoi(Img1, RotatedRoi1, ROI_TYPE_ROTATED, color=255, offset=(20, 20))
    # drawRoi(Img2, RotatedRoi2, ROI_TYPE_ROTATED, color=255, offset=(20, 20))
    cv2.imshow('Roi2', Img2)
    cv2.imshow('Roi1', Img1)
    cv2.waitKey()
