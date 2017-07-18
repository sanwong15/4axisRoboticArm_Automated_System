#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

__author__ = 'hkh'
__date__ = '03/12/2015'
__version__ = '2.0'


import cv2
import math
import copy
import numpy as np


ROI_TYPE_XYWH = 0L
ROI_TYPE_XYXY = 8L

ROI_CVT_XYXY2XYWH = 0L
ROI_CVT_XYWH2XYXY = 8L


def gammaTransform(src, gamma):
    LUT = []
    C = 255.0 / (255 ** gamma)
    for i in xrange(256):
        LUT.append(C * (i**gamma))
    return cv2.LUT(src, np.array(LUT, dtype=np.uint8))

def gammaTransform_BGR(src, gamma):
    if src.ndim == 1:
        return gammaTransform(src, gamma)
    elif src.ndim == 3:
        HSVImg = cv2.cvtColor(src=src, code=cv2.COLOR_BGR2HSV)
        H = HSVImg[:, :, 0]
        S = HSVImg[:, :, 1]
        V = HSVImg[:, :, 2]
        V = gammaTransform(V, gamma)
        NewHSVImg = cv2.merge((H, S, V))
        return cv2.cvtColor(src=NewHSVImg, code=cv2.COLOR_HSV2BGR)

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
    assert (0 != ContoursNum), 'contours num is 0'

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

def calcCentroid(array, binaryImage=False):
    Moments = cv2.moments(array=array, binaryImage=binaryImage)
    try:
        MarkPt_2x1 = np.array([[Moments['m10'] / Moments['m00']],
                               [Moments['m01'] / Moments['m00']]])
        return True, MarkPt_2x1
    except ZeroDivisionError:
        return False, None

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

def cvtRoi(roi, flag):
    """
    convert roi type
    :param roi: list or ndarray
    :param flag: ROI_CVT_XYXY2XYWH or ROI_CVT_XYWH2XYXY
    :return: roi in type you want
    """
    newRoi = [0]*4
    newRoi[:] = roi[:]
    if ROI_CVT_XYWH2XYXY == flag:
        newRoi[2] = roi[0] + roi[2] + 1
        newRoi[3] = roi[1] + roi[3] + 1
    if ROI_CVT_XYXY2XYWH == flag:
        newRoi[2] = roi[2] - roi[0] - 1
        newRoi[3] = roi[3] - roi[1] - 1
    return newRoi

def getRoiImg(img, roi, roiType, copyFlag=True):
    """
    :param img: gray image or BGR image
    :param roi: list or ndarray
    :param roiType: flag - ROI_TYPE_XYWH or ROI_TYPE_XYXY
    :return: Roi image
    """
    if ROI_TYPE_XYWH == roiType:
        roi_xyxy = cvtRoi(roi=roi, flag=ROI_CVT_XYWH2XYXY)
    else:
        roi_xyxy = copy.copy(roi)
    if roi_xyxy[0] < 0:
        roi_xyxy[0] = 0
    if roi_xyxy[1] < 0:
        roi_xyxy[1] = 0
    if 3 == img.ndim:
        RoiImg = img[roi_xyxy[1]:roi_xyxy[3], roi_xyxy[0]:roi_xyxy[2], :]
    else:
        RoiImg = img[roi_xyxy[1]:roi_xyxy[3], roi_xyxy[0]:roi_xyxy[2]]
    Offset_2x1 = np.array(roi_xyxy[:2]).reshape(2, 1)
    if copyFlag:
        return Offset_2x1, RoiImg.copy()
    else:
        return Offset_2x1, RoiImg

def getContourRoi_xywh(contour):
    ContourRoi_xywh = cv2.boundingRect(contour)
    return ContourRoi_xywh

def drawRoi(img, roi, roiType, color, thickness=2):
    """
    draw roi(rectangle) in img
    :param img: gray image or BGR image
    :param roi: list or ndarray
    :param roiType: flag - ROI_TYPE_XYWH or ROI_TYPE_XYXY
    :param color: plot color you want
    :param thickness: roi(rectangle)'s thickness
    :return: None
    """
    if ROI_TYPE_XYWH == roiType:
        roi_xyxy = cvtRoi(roi=roi, flag=ROI_CVT_XYWH2XYXY)
    else:
        roi_xyxy = roi
    cv2.rectangle(img, (int(roi_xyxy[0]), int(roi_xyxy[1])), (int(roi_xyxy[2]), int(roi_xyxy[3])), color, thickness=thickness)

def drawPoints(img, pts_2xn, color, radius=1, thickness=-1):
    """
    draw points(circles) in img
    :param img: gray image or BGR image
    :param pts_2xn: 2xn ndarray
    :param color: plot color you want
    :param radius: points(circles)'s radius
    :param thickness: points(circles)'s thickness
    :return: None
    """
    if pts_2xn.size == 2:
        cv2.circle(img, (int(pts_2xn.ravel()[0].round()), int(pts_2xn.ravel()[1].round())), radius, color, thickness)
    else:
        assert pts_2xn.ndim == 2, 'points_2xn must be 2xn'
        assert pts_2xn.shape[0] == 2, 'points_2xn must be 2xn'
        for idx in range(pts_2xn.shape[1]):
            cv2.circle(img, (int(pts_2xn[0, idx].round()), int(pts_2xn[1, idx].round())), radius, color, thickness)

def drawLine(img, point1, point2, color, thickness=2):
    """
    draw line in img
    :param img: gray image or BGR image
    :param point1: line's first point - list, tuple or ndarray
    :param point2: line's second point - list, tuple or ndarray
    :param color: line's color you want
    :param thickness: line's thickness
    :return: None
    """
    assert type(point1) in (tuple, list, np.ndarray), 'point1 should be list, tuple or ndarray!'
    assert type(point2) in (tuple, list, np.ndarray), 'point1 should be list, tuple or ndarray!'
    Point1 = np.array(point1)
    Point2 = np.array(point2)
    cv2.line(img=img, pt1=(int(Point1.item(0)), int(Point1.item(1))),
             pt2=(int(Point2.item(0)), int(Point2.item(1))), color=color, thickness = thickness)

def drawContours(srcImg, contours, contourIdx, color, thickness=1, lineType=8, hierarchy=None, maxLevel=1<<31-1, offset=(0,0)):
    if hierarchy is not None:
        cv2.drawContours(image=srcImg, contours=contours, contourIdx=contourIdx,
                         color=color, thickness=thickness, lineType=lineType,
                         hierarchy=hierarchy, maxLevel=maxLevel, offset=offset)
    else:
        cv2.drawContours(image=srcImg, contours=contours, contourIdx=contourIdx,
                         color=color, thickness=thickness, lineType=lineType, offset=offset)

def sortDisBWPt2Pts(point_2x1, points_2xn, ascending=True):
    """
    sort point to points by distance
    :param point_2x1: a point - ndarray
    :param points_2xn: points
    :param ascending: True or False
    :return: index sorted by calculating the distance between every point(in points_2xn) to point_2x1
    """
    assert isinstance(points_2xn, np.ndarray),  'points must be ndarray'
    assert isinstance(point_2x1, np.ndarray),   'point must be ndarray'
    assert point_2x1.shape == (2,1),   'point must be 2-by-1'
    assert points_2xn.ndim == 2,          'points must be 2*N'
    assert points_2xn.shape[0] == 2,            'points must be 2*N'

    Dis_1xn = np.linalg.norm(point_2x1 - points_2xn, axis=0)
    sortIdx = Dis_1xn.argsort()
    if not ascending:
        sortIdx[:] = sortIdx[::-1]
    return sortIdx

def OTSU(grayImg, threshBegin=0, threshEnd=256, threshStep=1):
    assert grayImg.ndim == 2, "must input a gary_img"

    max_g = 0
    suitable_th = 0
    for threshold in xrange(threshBegin, threshEnd, threshStep):
        bin_img = grayImg > threshold
        bin_img_inv = grayImg <= threshold
        fore_pix = np.sum(bin_img)
        back_pix = np.sum(bin_img_inv)
        if 0 == fore_pix:
            break
        if 0 == back_pix:
            continue

        w0 = float(fore_pix) / grayImg.size
        u0 = float(np.sum(grayImg * bin_img)) / fore_pix
        w1 = float(back_pix) / grayImg.size
        u1 = float(np.sum(grayImg * bin_img_inv)) / back_pix
        # intra-class variance
        g = w0 * w1 * (u0 - u1) * (u0 - u1)
        if g > max_g:
            max_g = g
            suitable_th = threshold
    return suitable_th

def rotateImg(srcImg, angle_deg):
    """
    :param numpy.ndarray src: the sra image
    :param float angle_deg:
    :param float scale:
    :return: ndarray, rotated image
    """
    w1 = math.fabs(srcImg.shape[1] * math.cos(np.deg2rad(angle_deg)))
    w2 = math.fabs(srcImg.shape[0] * math.sin(np.deg2rad(angle_deg)))
    h1 = math.fabs(srcImg.shape[1] * math.sin(np.deg2rad(angle_deg)))
    h2 = math.fabs(srcImg.shape[0] * math.cos(np.deg2rad(angle_deg)))
    width = int(w1 + w2) + 1
    height = int(h1 + h2) + 1
    dstSize = (width, height)
    x = srcImg.shape[1]
    y = srcImg.shape[0]
    center = np.array([x/2, y/2]).reshape(2, 1)
    rotateMatrix = cv2.getRotationMatrix2D(center=(0, 0),
                                           angle=angle_deg,
                                           scale=1.0)
    rotateCenter = np.dot(rotateMatrix[0:2, 0:2].reshape(2, 2), center)
    rotateMatrix[0, 2] = width / 2 - rotateCenter[0]
    rotateMatrix[1, 2] = height / 2 - rotateCenter[1]
    if 0 == angle_deg % 90:
        angle_deg = angle_deg / 90 % 4
        # rotatedImg = np.array(np.rot90(srcImg, angle_deg), dtype=np.uint8)
        rotatedImg = np.rot90(srcImg, angle_deg)
    else:
        rotatedImg = cv2.warpAffine(src=srcImg,
                                    M=rotateMatrix,
                                    dsize=dstSize)
    TranFormMatrix = np.vstack((rotateMatrix, np.array([0.0, 0.0, 1.0])))
    return rotatedImg, TranFormMatrix

def ImageThin(img_bin, maxIteration = -1):
    imgthin = np.copy(img_bin)
    imgthin2 = np.copy(img_bin)
    count = 0
    rows = imgthin.shape[0]
    cols = imgthin.shape[1]
    while True:
        count += 1
        if maxIteration != -1 and count > maxIteration:
            break

        flag = 0
        for i in range(rows):
            for j in range(cols):
                # p9 p2 p3
                # p8 p1 p4
                # p7 p6 p5
                p1 = imgthin[i, j]
                p2 = 0 if i == 0 else imgthin[i-1, j]
                p3 = 0 if i == 0 or j == cols-1 else imgthin[i-1, j+1]
                p4 = 0 if j == cols-1 else imgthin[i, j+1]
                p5 = 0 if i == rows-1 or j == cols-1 else imgthin[i+1, j+1]
                p6 = 0 if i == rows-1 else imgthin[i+1, j]
                p7 = 0 if i == rows-1 or j == 0 else imgthin[i+1, j-1]
                p8 = 0 if j == 0 else imgthin[i, j-1]
                p9 = 0 if i == 0 or j == 0 else imgthin[i-1, j-1]

                if (p2+p3+p4+p5+p6+p7+p8+p9) >= 2 and (p2+p3+p4+p5+p6+p7+p8+p9) <= 6:
                    ap = 0
                    if p2 == 0 and p3 == 1:
                        ap += 1
                    if p3 == 0 and p4 == 1:
                        ap += 1
                    if p4 == 0 and p5 == 1:
                        ap += 1
                    if p5 == 0 and p6 == 1:
                        ap += 1
                    if p6 == 0 and p7 == 1:
                        ap += 1
                    if p7 == 0 and p8 == 1:
                        ap += 1
                    if p8 == 0 and p9 == 1:
                        ap += 1
                    if p9 == 0 and p2 == 1:
                        ap += 1
                    if ap == 1:
                        if p2*p4*p6 == 0:
                            if p4*p6*p8 == 0:
                                imgthin2[i, j] = 0
                                flag = 1
        if flag == 0:
            break;
        imgthin = np.copy(imgthin2)

        flag = 0
        for i in range(rows):
            for j in range(cols):
                # p9 p2 p3
                # p8 p1 p4
                # p7 p6 p5
                p1 = imgthin[i, j]
                if p1 != 1:
                    continue
                p2 = 0 if i == 0 else imgthin[i-1, j]
                p3 = 0 if i == 0 or j == cols-1 else imgthin[i-1, j+1]
                p4 = 0 if j == cols-1 else imgthin[i, j+1]
                p5 = 0 if i == rows-1 or j == cols-1 else imgthin[i+1, j+1]
                p6 = 0 if i == rows-1 else imgthin[i+1, j]
                p7 = 0 if i == rows-1 or j == 0 else imgthin[i+1, j-1]
                p8 = 0 if j == 0 else imgthin[i, j-1]
                p9 = 0 if i == 0 or j == 0 else imgthin[i-1, j-1]

                if (p2+p3+p4+p5+p6+p7+p8+p9) >= 2 and (p2+p3+p4+p5+p6+p7+p8+p9) <= 6:
                    ap = 0
                    if p2 == 0 and p3 == 1:
                        ap += 1
                    if p3 == 0 and p4 == 1:
                        ap += 1
                    if p4 == 0 and p5 == 1:
                        ap += 1
                    if p5 == 0 and p6 == 1:
                        ap += 1
                    if p6 == 0 and p7 == 1:
                        ap += 1
                    if p7 == 0 and p8 == 1:
                        ap += 1
                    if p8 == 0 and p9 == 1:
                        ap += 1
                    if p9 == 0 and p2 == 1:
                        ap += 1
                    if ap == 1:
                        if p2*p4*p8 == 0:
                            if p2*p6*p8 == 0:
                                imgthin2[i, j] = 0
                                flag = 1
        if flag == 0:
            break;
        imgthin = np.copy(imgthin2)
    for i in range(rows):
        for j in range(cols):
            if imgthin2[i, j] == 1:
                imgthin2[i, j] = 255
    return imgthin2

def enhanceWithLaplacion(gray_img):
    assert 2 == gray_img.ndim

    lap_img = cv2.Laplacian(gray_img, cv2.CV_8UC1)
    enhance_img = cv2.subtract(gray_img, lap_img)
    return enhance_img

def enhanceWithLaplacion2(SrcImg):
    if 2 == SrcImg.ndim:
        type = cv2.CV_8UC1
    elif 3 == SrcImg.ndim:
        type = cv2.CV_8UC3
    else:
        raise ValueError

    kernel = np.array([[-1, -1, -1],
                      [-1, 9, -1],
                      [-1, -1, -1]])
    return cv2.filter2D(src=SrcImg, ddepth=type, kernel=kernel)

def ULBP(src):
    assert 2 == src.ndim

    r, c = src.shape
    LBP = np.zeros(shape=src.shape, dtype=np.uint8)
    LbpHist = np.zeros(shape=(256, 1), dtype=np.float32)
    Kernel = np.array([[1,   2,  4],
                       [128, 0,  8],
                       [64, 32, 16]], dtype=np.uint8)

    for i in xrange(1, r-1):
        for j in xrange(1, c-1):
            Mask = np.zeros(shape=(3, 3), dtype=np.uint8)
            for m in xrange(-1, 2):
                for n in xrange(-1, 2):
                    Mask[m][n] = 1 if src[i+m][j+n] >= src[i][j] else 0
            LbpValue = int(np.sum(Mask * Kernel))

            if 255 == LbpValue and 0 == src[i][j]:
            # if 255 == LbpValue:
                continue
            # LBP[i][j] = LbpValue
            # ValueLeft = ((LbpValue << 1)&0xff) + (LbpValue & 0x01)
            # Temp = LbpValue ^ ValueLeft
            # JumpCount = 0
            # while Temp:
            #     Temp &= Temp - 1
            #     JumpCount += 1
            # # print '%x'%(Value), count
            # if JumpCount <= 2:
            #     LbpHist[LbpValue] += 1
            LbpHist[LbpValue] += 1
    return LBP, LbpHist

if __name__ == '__main__':
    roi_xywh = [10, 20, 50, 70]
    print 'Roi_xywh:      ', roi_xywh
    roi_xywh2xyxy = cvtRoi(roi=roi_xywh, flag=ROI_CVT_XYWH2XYXY)
    roi_xyxy2xywh = cvtRoi(roi=roi_xywh2xyxy, flag=ROI_CVT_XYXY2XYWH)
    print 'roi_xywh2xyxy: ', roi_xywh2xyxy
    print 'roi_xyxy2xywh: ', roi_xyxy2xywh

    # SrcImg = cv2.imread('../Data/girl.jpg')
    SrcImg = cv2.imread('../Data/Cam14.bmp')
    resizeImg1 = cv2.resize(SrcImg, (SrcImg.shape[1]/5, SrcImg.shape[0]/5))
    drawRoi(img=resizeImg1, roi=roi_xywh, roiType=ROI_TYPE_XYWH, color=(0,0,255))
    cv2.imshow('roi_xywh', resizeImg1)
    resizeImg2 = cv2.resize(SrcImg, (SrcImg.shape[1]/5, SrcImg.shape[0]/5))
    drawRoi(img=resizeImg2, roi=roi_xywh2xyxy, roiType=ROI_TYPE_XYXY, color=(0,0,255))
    cv2.imshow('roi_xywh2xyxy', resizeImg2)

    RotateImg = rotateImg(src=SrcImg, angle_deg=30)
    cv2.namedWindow("RotateImg", cv2.WINDOW_NORMAL)
    cv2.imshow("RotateImg", RotateImg)

    cv2.waitKey()

