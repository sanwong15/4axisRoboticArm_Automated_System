__author__ = 'Li Hao'
__version__ = '1.0'
__date__ = '01/11/2016'
__copyright__ = "Copyright 2016, PI"


import cv2
import math
import copy
import numpy as np

__all__ = [
    'ROI_TYPE_XYWH',
    'ROI_TYPE_XYXY',
    'ROI_TYPE_ROTATED',
    'ROI_CVT_XYXY2XYWH',
    'ROI_CVT_XYWH2XYXY',
    'IPTError',
    'cvtRoi',
    'getRoiImg',
    'inRoi',
    'drawRoi',
    'drawPoints',
    'drawLine',
    'drawContours',
    'gammaTransform',
    'fillHoles',
    'contrastStretch',
    'findTopNAreaContours',
]


ROI_TYPE_XYWH = 0L
ROI_TYPE_XYXY = 8L
ROI_TYPE_ROTATED = 64L

ROI_CVT_XYXY2XYWH = 0L
ROI_CVT_XYWH2XYXY = 8L


# Error object
class IPTError(Exception):
    """
    Generic Python-exception-derived object raised by Image Process Tool functions.

    General purpose exception class, derived from Python's exception.Exception
    class, programmatically raised in Image Process Tool functions when a Image
    Processing-related condition would prevent further correct execution of the
    function.

    Parameters
    ----------
    None

    Examples
    --------
    >>> import ImageProcessTool as IPT
    >>> IPT.getRoiImg(None, [1, 2, 3, 4], IPT.ROI_TYPE_XYWH)
    Traceback (most recent call last):
    ...
    IPTError: ...
    """
    pass

def __raiseError(msg=None):
    if msg is None:
        raise IPTError
    else:
        raise IPTError, msg

def __checkNone(img):
    if img is None:
        raise IPTError('img can not be [None]')

def __cvtRoi2xyxy(roi, roiType):
    if ROI_TYPE_XYWH == roiType:
        return cvtRoi(roi=roi, flag=ROI_CVT_XYWH2XYXY)
    elif ROI_TYPE_XYXY == roiType:
        return np.array(roi).ravel()
    else:
        raise ValueError, 'flag is wrong!!!'

def cvtRoi(roi, flag):
    """
    Convert roi to different type.

    Parameters
    ----------
    roi : array_like
        4 element to represent roi.
    flag : int
        The option for converting roi, must be the following values:
        - ROI_CVT_XYXY2XYWH, convert roi from XYXY format to XYWH
        - ROI_CVT_XYWH2XYXY, convert roi from XYWH format to XYXY

    Returns
    -------
    roi : ndarray
        Roi in desired type.

    Examples
    --------
    >>> import numpy as np
    >>> import ImageProcessTool as IPT
    >>> Roi_xyxy = [1, 3, 5, 15]
    >>> IPT.cvtRoi(Roi_xyxy, IPT.ROI_CVT_XYXY2XYWH)
    array(...1, ...3, ...4, ...12])

    """
    x0, y0, c, d = roi
    if ROI_CVT_XYWH2XYXY == flag:
        x1 = x0 + c
        y1 = y0 + d
        return np.array([x0, y0, x1, y1])
    elif ROI_CVT_XYXY2XYWH == flag:
        w = c - x0
        h = d - y0
        return np.array([x0, y0, w, h])
    else:
        __raiseError('flag is wrong!')

def getRoiImg(img, roi, roiType, copy=True):
    """
    Cut roi from source image.

    Parameters
    ----------
    img : ndarray
        2-D or 3-D image.
    roi : array_like
        4 element to represent roi.
    roiType : int
        The type of roi, must be the following values:
        - ROI_TYPE_XYXY, roi with XYXY format
        - ROI_TYPE_XYWH, roi with XYWH format
    copy : bool, optional
        - True, return the copy of the roi in source image.
        - False, return a view of the roi in source image.

    Returns
    -------
    Offset_2x1 : ndarray
        Offset for the roi, 2x1.
    RoiImg : ndarray

    Raises
    ------
    IPTError:
        If image is not 2-D or 3-D array, or invalid roiType.

    Examples
    --------
    >>> import numpy as np
    >>> import ImageProcessTool as IPT
    >>> Img = np.array([[1, 2, 3, 4], [11, 12, 13, 14], [21, 22, 23, 24], [31, 32, 33, 34]], np.uint8)
    >>> Roi_xyxy = [1, 1, 3, 3]
    >>> Offset_2x1, RoiImg = IPT.getRoiImg(Img, Roi_xyxy, IPT.ROI_TYPE_XYXY)
    >>> Offset_2x1
    array([[1],
           [1]])
    >>> RoiImg
    array([[12, 13],
           [22, 23]], dtype=uint8)
    """
    if img is None:
        raise IPTError, 'img is None'
    Roi_xyxy = __cvtRoi2xyxy(roi, roiType)
    if Roi_xyxy[0] < 0:
        Roi_xyxy[0] = 0
    if Roi_xyxy[1] < 0:
        Roi_xyxy[1] = 0
    if Roi_xyxy[2] < 0 or Roi_xyxy[3] < 0:
        raise IPTError, 'roi data invalid'
    if 3 == img.ndim:
        if copy:
            RoiImg = img[Roi_xyxy[1]:Roi_xyxy[3], Roi_xyxy[0]:Roi_xyxy[2], :].copy()
        else:
            RoiImg = img[Roi_xyxy[1]:Roi_xyxy[3], Roi_xyxy[0]:Roi_xyxy[2], :]
    elif 2 == img.ndim:
        if copy:
            RoiImg = img[Roi_xyxy[1]:Roi_xyxy[3], Roi_xyxy[0]:Roi_xyxy[2]].copy()
        else:
            RoiImg = img[Roi_xyxy[1]:Roi_xyxy[3], Roi_xyxy[0]:Roi_xyxy[2]]
    else:
        raise IPTError, 'img data error'
    Offset_2x1 = np.array(Roi_xyxy[:2]).reshape(2, 1)
    return Offset_2x1, RoiImg

def inRoi(pt, roi, roiType):
    """
    Check if point in the roi.

    Parameters
    ----------
    pt : array_like
        Point in image.
    roi : array_like
        4 element to represent roi.
    roiType : int
        The type of roi, must be the following values:
        - ROI_TYPE_XYXY, roi with XYXY format
        - ROI_TYPE_XYWH, roi with XYWH format

    Returns
    -------
    ret : bool
        - True, point in the roi.
        - False, point not in the roi.

    Examples
    --------
    >>> import ImageProcessTool as IPT
    >>> Point = [2, 2]
    >>> IPT.inRoi(Point, [0, 0, 2, 2], roiType=IPT.ROI_TYPE_XYXY)
    False
    >>> IPT.inRoi(Point, [0, 0, 3, 3], roiType=IPT.ROI_TYPE_XYXY)
    True
    """
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

def drawRoi(img, roi, roiType, color, thickness=2, lineType=1, shift=0, offset=(0,0)):
    """
    Draw the border of roi in the image.

    Parameters
    ----------
    img : ndarray
        2-D or 3-D image.
    roi : array_like
        4 element to represent roi.
    roiType : int
        The type of roi, must be the following values:
        - ROI_TYPE_XYXY, roi with XYXY format
        - ROI_TYPE_XYWH, roi with XYWH format
    color : tuple
    thickness : int, optional
    lineType : int, optional
    shift : int, optional
    offset : array_like, optional

    Returns
    -------
    img : ndarray
        Source image with drawn.

    Raises
    ------
    IPTError:
        Invalid roiType.

    Examples
    --------
    >>> import numpy as np
    >>> import ImageProcessTool as IPT
    >>> Img = np.zeros((4, 4), np.uint8)
    >>> Roi_xyxy = [1, 1, 4, 4]
    >>> IPT.drawRoi(Img, Roi_xyxy, IPT.ROI_TYPE_XYXY, 255, thickness=1)
    array([[  0,   0,   0,   0],
           [  0, 255, 255, 255],
           [  0, 255,   0, 255],
           [  0, 255, 255, 255]], dtype=uint8)
    """
    if img is None:
        raise IPTError, 'img is None'
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
        cv2.rectangle(img, (int(Roi_xyxy[0]), int(Roi_xyxy[1])), (int(Roi_xyxy[2]-1), int(Roi_xyxy[3]-1)), color,
                      thickness=thickness, lineType=lineType, shift=shift)
    return img

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
    Offset = np.array(offset).reshape(2, 1)
    Pts_2xn = np.round(np.array(pts_2xn).reshape(2, -1) + Offset).astype(np.int)
    for idx in range(Pts_2xn.shape[1]):
        cv2.circle(img, tuple(Pts_2xn[:, idx]), radius, color, thickness, shift=0)

def drawLine(img, point1, point2, color, thickness=2, shift=0, offset=(0, 0)):
    """
    draw line in img
    :param img: gray image or BGR image
    :param point1: line's first point - list, tuple or ndarray
    :param point2: line's second point - list, tuple or ndarray
    :param color: line's color you want
    :param thickness: line's thickness
    :return: None
    """
    Offset = np.array(offset).ravel()
    Point1 = np.round(point1).ravel() + Offset
    Point2 = np.round(point2).ravel() + Offset
    cv2.line(img=img, pt1=tuple(Point1), pt2=tuple(Point2), color=color, thickness=thickness, shift=shift)

def drawContours(srcImg, contours, contourIdx, color, thickness=1, lineType=8, hierarchy=None, maxLevel=1<<31-1, offset=(0,0)):
    if hierarchy is not None:
        cv2.drawContours(image=srcImg, contours=contours, contourIdx=contourIdx,
                         color=color, thickness=thickness, lineType=lineType,
                         hierarchy=hierarchy, maxLevel=maxLevel, offset=offset)
    else:
        cv2.drawContours(image=srcImg, contours=contours, contourIdx=contourIdx,
                         color=color, thickness=thickness, lineType=lineType, offset=offset)

def __gammaTransform(src, gamma):
    LUT = []
    C = 255.0 / (255 ** gamma)
    for i in xrange(256):
        LUT.append(C * (i**gamma))
    return cv2.LUT(src, np.round(LUT).astype(np.uint8))

def gammaTransform(src, gamma):
    """
    Adjust image through gamma transform.

    Parameters
    ----------
    src : ndarray
        Color image or gray scale image.
    gamma : float
        Gamma scalar.

    Returns
    -------
    img : ndarray
        Image after gamma transform.

    References
    ----------
    [1] https://www.mathworks.com/help/images/gamma-correction.html
    [2] https://en.wikipedia.org/wiki/Gamma_correction

    Examples
    --------
    >>> import numpy as np
    >>> import ImageProcessTool as IPT
    >>> Img = np.array([[5,  15], [20, 10]], np.uint8)
    >>> IPT.gammaTransform(Img, 0.5)
    array([[36, 62],
           [71, 50]], dtype=uint8)
    """
    if src.ndim == 2:
        return __gammaTransform(src, gamma)
    elif src.ndim == 3:
        HSVImg = cv2.cvtColor(src=src, code=cv2.COLOR_BGR2HSV)
        H = HSVImg[:, :, 0]
        S = HSVImg[:, :, 1]
        V = HSVImg[:, :, 2]
        V = __gammaTransform(V, gamma)
        NewHSVImg = cv2.merge((H, S, V))
        return cv2.cvtColor(src=NewHSVImg, code=cv2.COLOR_HSV2BGR)

def fillHoles(binImg, backGroundValue, foreGroundValue, loDiff=None, upDiff=None):
    DstImg = binImg.copy()
    MaskImg = binImg.copy()
    MaxRow, MaxCol = MaskImg.shape[0]-1, MaskImg.shape[1]-1
    for i in range(MaskImg.shape[0]):
        if MaskImg[i, 0] == backGroundValue:
            cv2.floodFill(MaskImg, mask=None, seedPoint=(0, i), newVal=foreGroundValue, loDiff=loDiff, upDiff=upDiff)
        if MaskImg[i, MaxCol] == backGroundValue:
            cv2.floodFill(MaskImg, mask=None, seedPoint=(MaxCol, i), newVal=foreGroundValue, loDiff=loDiff, upDiff=upDiff)
    for j in range(MaskImg.shape[1]):
        if MaskImg[0, j] == backGroundValue:
            cv2.floodFill(MaskImg, mask=None, seedPoint=(j, 0), newVal=foreGroundValue, loDiff=loDiff, upDiff=upDiff)
        if MaskImg[MaxRow, j] == backGroundValue:
            cv2.floodFill(MaskImg, mask=None, seedPoint=(j, MaxRow), newVal=foreGroundValue, loDiff=loDiff, upDiff=upDiff)
    DstImg[MaskImg == backGroundValue] = foreGroundValue
    return DstImg

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

def findTopNAreaContours(contours, num=1):
    """
    Find the top-n area among contours.

    Parameters
    ----------
    contours : list
        List of ndarray, comes from OpenCV.
    num : int, optional
        Number indicating the number of top-n, default is 1.

    Returns
    -------
    SortedIdx : ndarray
        The index of top-n area contours.
    """
    if contours:
        ContoursArea = map(cv2.contourArea, contours)
        SortedIdx = np.argsort(np.array(ContoursArea))[::-1]
        return SortedIdx[:num]
    __raiseError('The length of contours is 0')

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
        # np.sum(img, 0)
        # np.sum(img, 1)
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

def enhanceWithLaplacian(gray_img):
    assert 2 == gray_img.ndim

    lap_img = cv2.Laplacian(gray_img, cv2.CV_8UC1)
    enhance_img = cv2.subtract(gray_img, lap_img)
    return enhance_img

def enhanceWithLaplacian2(SrcImg):
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
    import ImageProcessTool as IPT
    # ================= gamma transform ================ #
    # SrcImg = cv2.imread('../Datas/Input/Gamma.png')
    # DstImg = gammaTransform(SrcImg, 0.5)
    # GrayImg = cv2.cvtColor(SrcImg, cv2.COLOR_BGR2GRAY)
    # DstGrayImg = gammaTransform(GrayImg, 0.5)
    # cv2.imshow('Src', SrcImg)
    # cv2.imshow('GrayImg', GrayImg)
    # cv2.imshow('DstImg', DstImg)
    # cv2.imshow('DstGrayImg', DstGrayImg)
    # cv2.waitKey()

    # ================= find max contours ================ #
    # import ImageProcessTool as IPT
    # SrcImg = cv2.imread('../Datas/Input/Gamma.png', 0)
    # Contours, Hierarchy = cv2.findContours(SrcImg.copy(), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    # ContourIdx = IPT.findTopNAreaContours(Contours, 1)
    # ShowImg = SrcImg.copy()
    # for i in ContourIdx:
    #     cv2.drawContours(ShowImg, Contours, i, color=150)
    #     cv2.imshow('Show', ShowImg)
    #     cv2.waitKey()

    # ================= drawing ================ #
    # import ImageProcessTool as IPT
    # SrcImg = cv2.imread('../Datas/Input/Gamma.png')
    # IPT.drawPoints(SrcImg, np.array([20, 20]),(0, 0, 255))
    # cv2.imshow('Src', SrcImg)
    # IPT.drawPoints(SrcImg, np.array([0, 0]),(0, 255, 255), offset=np.array([20, 20]))
    # cv2.imshow('Src', SrcImg)
    # IPT.drawLine(SrcImg, [30, 30], [100, 100], (0, 0, 255), offset=[10, 0])
    # cv2.imshow('Src', SrcImg)
    # IPT.drawLine(SrcImg, [30, 30], [100, 100], (0, 255, 255), offset=[0, 0], shift=1)
    # cv2.imshow('Src', SrcImg)
    # IPT.drawLine(SrcImg, [30>>1, 30>>1], [100>>1, 100>>1], (255, 255, 255), offset=[0, 0], shift=0)
    # cv2.imshow('Src', SrcImg)
    # cv2.waitKey()

    # Img = np.zeros((10, 10), np.uint8)
    # Roi_xyxy = [2, 2, 5, 5]
    # Roi_xywh = [2, 2, 3, 3]
    # CvtRoi_xywh = cvtRoi(Roi_xyxy, flag=ROI_CVT_XYXY2XYWH)
    # CvtRoi_xyxy = cvtRoi(Roi_xywh, flag=ROI_CVT_XYWH2XYXY)
    # print 'xyxy: ', Roi_xyxy, 'convert to xywh->', CvtRoi_xywh, 'convert back->', cvtRoi(CvtRoi_xywh, ROI_CVT_XYWH2XYXY)
    # print 'xywh: ', Roi_xywh, 'convert to xyxy->', CvtRoi_xyxy, 'convert back->', cvtRoi(CvtRoi_xyxy, ROI_CVT_XYXY2XYWH)
    #
    # _, RoiImg_xyxy = getRoiImg(Img, Roi_xyxy, roiType=ROI_TYPE_XYXY)
    # print 'RoiImg_xyxy shape: ', RoiImg_xyxy.shape
    # _, RoiImg_xywh = getRoiImg(Img, Roi_xywh, roiType=ROI_TYPE_XYWH)
    # print 'RoiImg_xywh shape: ', RoiImg_xywh.shape
    #
    # Point = [0, 0]
    # print Point, 'in roi_xyxy?', inRoi(Point, Roi_xyxy, ROI_TYPE_XYXY), '\t/\t in roi_xywh?', inRoi(Point, Roi_xywh, ROI_TYPE_XYWH)
    # Point = [2, 2]
    # print Point, 'in roi_xyxy?', inRoi(Point, Roi_xyxy, ROI_TYPE_XYXY), '\t/\t in roi_xywh?', inRoi(Point, Roi_xywh, ROI_TYPE_XYWH)
    # Point = [3, 3]
    # print Point, 'in roi_xyxy?', inRoi(Point, Roi_xyxy, ROI_TYPE_XYXY), '\t/\t in roi_xywh?', inRoi(Point, Roi_xywh, ROI_TYPE_XYWH)
    # Point = [4, 4]
    # print Point, 'in roi_xyxy?', inRoi(Point, Roi_xyxy, ROI_TYPE_XYXY), '\t/\t in roi_xywh?', inRoi(Point, Roi_xywh, ROI_TYPE_XYWH)
    # Point = [5, 5]
    # print Point, 'in roi_xyxy?', inRoi(Point, Roi_xyxy, ROI_TYPE_XYXY), '\t/\t in roi_xywh?', inRoi(Point, Roi_xywh, ROI_TYPE_XYWH)
    #
    # cv2.namedWindow('Roi2', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('Roi1', cv2.WINDOW_NORMAL)
    # Img1 = np.zeros((100, 100), np.uint8)
    # Img2 = np.zeros((100, 100), np.uint8)
    # RotatedRoi1 = [[10, 10],
    #               [50, 20],
    #               [70, 70],
    #               [20, 30]]
    # RotatedRoi2 = [[10, 50, 70, 20],
    #               [10, 20, 70, 30]]
    # print 'RotatedRoi:', RotatedRoi1
    # Point_2x1 = np.array([10, 20]).reshape(2, 1)
    # drawPoints(Img1, Point_2x1, 255)
    # print Point_2x1.ravel(), 'in RotatedRoi?', inRoi(Point_2x1, RotatedRoi1, ROI_TYPE_ROTATED)
    # Point_2x1 = np.array([10, 10]).reshape(2, 1)
    # drawPoints(Img1, Point_2x1, 255)
    # print Point_2x1.ravel(), 'in RotatedRoi?', inRoi(Point_2x1, RotatedRoi1, ROI_TYPE_ROTATED)
    # Point_2x1 = np.array([70, 70]).reshape(2, 1)
    # drawPoints(Img1, Point_2x1, 255)
    # drawPoints(Img1, Point_2x1, 255, offset=(-10, 10))
    # print Point_2x1.ravel(), 'in RotatedRoi?', inRoi(Point_2x1, RotatedRoi1, ROI_TYPE_ROTATED)
    # drawRoi(Img1, RotatedRoi1, ROI_TYPE_ROTATED, color=255)
    # drawRoi(Img2, RotatedRoi2, ROI_TYPE_ROTATED, color=255, offset=(-50, -50))
    # Contours, _ = cv2.findContours(image=Img2.copy(), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    # Rect = cv2.minAreaRect(Contours[0])
    # Box = cv2.cv.BoxPoints(Rect)
    # BoxImg = np.zeros((200, 200), np.uint8)
    # drawRoi(Img2, Box, ROI_TYPE_ROTATED, color=255)
    # drawRoi(Img1, RotatedRoi1, ROI_TYPE_ROTATED, color=255, offset=(20, 20))
    # # drawRoi(Img2, RotatedRoi2, ROI_TYPE_ROTATED, color=255, offset=(20, 20))
    # cv2.imshow('Roi2', Img2)
    # cv2.imshow('Roi1', Img1)
    # cv2.waitKey()

    # =================== test find contours 1 =================== #
    # BinImg = np.zeros((50, 50), np.uint8)
    # BinImg[25:, 35:40] = 255
    # BinImg[25:, 5:10] = 255
    # BinImg[25:30, 10:40] = 255
    # cv2.namedWindow('BinImg', cv2.WINDOW_NORMAL)
    # cv2.imshow('BinImg', BinImg)
    # Contours, Hierarchy = cv2.findContours(BinImg, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    # for contour in Contours:
    #     print 'contour:\n', contour
    #     Canvas = np.zeros((50, 50), np.uint8)
    #     cv2.drawContours(Canvas, [contour], 0, color=155, thickness=1)
    #     cv2.namedWindow('Canvas', cv2.WINDOW_NORMAL)
    #     cv2.imshow('Canvas', Canvas)
    #     cv2.waitKey()

    # =================== test find contours 2 =================== #
    # BinImg = np.zeros((50, 50), np.uint8)
    # BinImg[25:, 35:40] = 255
    # BinImg[25:, 5:10] = 255
    # BinImg[25:30, 10:40] = 255
    # cv2.namedWindow('SrcBinImg', cv2.WINDOW_NORMAL)
    # cv2.imshow('SrcBinImg', BinImg)
    # BinImg[:, 0:3] = 255
    # BinImg[0:3, :] = 255
    # BinImg[:, -1:-3:-1] = 255
    # BinImg[-1:-3:-1, :] = 255
    #
    # MaskImg = np.zeros((50, 50), np.uint8)
    # MaskImg[:, 0:3] = 255
    # MaskImg[0:3, :] = 255
    # MaskImg[:, -1:-3:-1] = 255
    # MaskImg[-1:-3:-1, :] = 255
    # cv2.namedWindow('BinImg', cv2.WINDOW_NORMAL)
    # cv2.imshow('BinImg', BinImg)
    # cv2.namedWindow('MaskImg', cv2.WINDOW_NORMAL)
    # cv2.imshow('MaskImg', MaskImg)
    #
    # Contours, Hierarchy = cv2.findContours(MaskImg, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    # print Contours
    # LastMaskContour = Contours[-1]
    # for contour in Contours:
    #     print 'contour:\n', contour
    #     Canvas = np.zeros((50, 50), np.uint8)
    #     cv2.drawContours(Canvas, [contour], 0, color=155, thickness=1)
    #     cv2.namedWindow('MaskCanvas', cv2.WINDOW_NORMAL)
    #     cv2.imshow('MaskCanvas', Canvas)
    #     cv2.waitKey()
    # # MaskImg = np.zeros((50, 50), np.uint8)
    # # cv2.drawContours(MaskImg, [LastMaskContour], 0, color=155, thickness=-1)
    # Contours, Hierarchy = cv2.findContours(BinImg, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    # # print Contours
    # # LastBinContour = Contours[-1]
    # # BinMaskImg = np.zeros((50, 50), np.uint8)
    # # cv2.drawContours(BinMaskImg, [LastBinContour], 0, color=155, thickness=-1)
    # #
    # # TargetImg = cv2.bitwise_xor(MaskImg, BinMaskImg)
    # # cv2.namedWindow('TargetImg', cv2.WINDOW_NORMAL)
    # # cv2.imshow('TargetImg', TargetImg)
    # # cv2.waitKey()
    # for contour in Contours:
    #     print 'contour:\n', contour
    #     Canvas = np.zeros((50, 50), np.uint8)
    #     cv2.drawContours(Canvas, [contour], 0, color=155, thickness=1)
    #     cv2.namedWindow('Canvas', cv2.WINDOW_NORMAL)
    #     cv2.imshow('Canvas', Canvas)
    #     cv2.waitKey()
    #
    # # im_in = cv2.imread("nickel.jpg", cv2.IMREAD_GRAYSCALE);

    # ============= im fill =============== #
    SrcImg = cv2.imread('../Datas/Input/holes.tif', 0)
    _, BinImg = cv2.threshold(SrcImg, 100, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("BinImg", BinImg)
    AfterFilled = IPT.fillHoles(BinImg, backGroundValue=255, foreGroundValue=0)
    cv2.imshow('Filled', AfterFilled)
    cv2.waitKey()
    # im_in = np.zeros((50, 50), np.uint8)
    # im_in[25:, 35:40] = 255
    # im_in[25:, 5:10] = 255
    # im_in[25:30, 10:40] = 255
    #
    # # Threshold.
    # # Set values equal to or above 220 to 0.
    # # Set values below 220 to 255.
    #
    # th, im_th = cv2.threshold(im_in, 220, 255, cv2.THRESH_BINARY);
    #
    # # Copy the thresholded image.
    # im_floodfill = im_th.copy()
    #
    # # Mask used to flood filling.
    # # Notice the size needs to be 2 pixels than the image.
    # h, w = im_th.shape[:2]
    # mask = np.zeros((h+2, w+2), np.uint8)
    #
    # # Floodfill from point (0, 0)
    # cv2.floodFill(im_floodfill, mask, (0,0), 255);
    #
    # # Invert floodfilled image
    # im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    #
    # # Combine the two images to get the foreground.
    # im_out = im_th | im_floodfill_inv
    #
    # # Display images.
    # cv2.namedWindow('Thresholded Image', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('Floodfilled Image', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('Inverted Floodfilled Image', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('Foreground', cv2.WINDOW_NORMAL)
    # cv2.imshow("Thresholded Image", im_th)
    # cv2.imshow("Floodfilled Image", im_floodfill)
    # cv2.imshow("Inverted Floodfilled Image", im_floodfill_inv)
    # cv2.imshow("Foreground", im_out)
    # cv2.waitKey(0)

    # Img = (np.random.random((100, 100)) * 255).astype(np.uint8)
    # roi_xywh = [10, 10, 20, 20]
    # roi_xyxy = [10, 10, 30, 30]
    # print 'Roi_xywh:      ', roi_xywh
    # roi_xywh2xyxy = cvtRoi(roi=roi_xywh, flag=ROI_CVT_XYWH2XYXY)
    # roi_xyxy2xywh = cvtRoi(roi=roi_xyxy, flag=ROI_CVT_XYXY2XYWH)
    #
    # print 'roi_xywh2xyxy: ', roi_xywh2xyxy
    # print 'roi_xyxy2xywh: ', roi_xyxy2xywh
    #
    # _, RoiImg_xywh = getRoiImg(Img, roi_xywh, roiType=ROI_TYPE_XYWH)
    # print 'RoiImg xywh:', RoiImg_xywh.shape
    #
    # _, RoiImg_xyxy = getRoiImg(Img, roi_xyxy, roiType=ROI_TYPE_XYXY)
    # print 'RoiImg xyxy:', RoiImg_xyxy.shape
    # print np.allclose(RoiImg_xywh, RoiImg_xyxy)


    # SrcImg = cv2.imread('../Data/girl.jpg')
    # SrcImg = cv2.imread('../Data/Cam14.bmp')
    # resizeImg1 = cv2.resize(SrcImg, (SrcImg.shape[1]/5, SrcImg.shape[0]/5))
    # drawRoi(img=resizeImg1, roi=roi_xywh, roiType=ROI_TYPE_XYWH, color=(0,0,255))
    # cv2.imshow('roi_xywh', resizeImg1)
    # resizeImg2 = cv2.resize(SrcImg, (SrcImg.shape[1]/5, SrcImg.shape[0]/5))
    # drawRoi(img=resizeImg2, roi=roi_xywh2xyxy, roiType=ROI_TYPE_XYXY, color=(0,0,255))
    # cv2.imshow('roi_xywh2xyxy', resizeImg2)
    #
    # RotateImg = rotateImg(src=SrcImg, angle_deg=30)
    # cv2.namedWindow("RotateImg", cv2.WINDOW_NORMAL)
    # cv2.imshow("RotateImg", RotateImg)
    #
    # cv2.waitKey()

