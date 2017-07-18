#ifndef URCONTROLSTAMP_HPP
#define URCONTROLSTAMP_HPP

#include "urcontrola.hpp"
#include <opencv2/opencv.hpp>
#include "deskA.hpp"


class URControlStamp : public URControlA
{
protected:

    static cv::Vec3d HeadToBase(cv::Vec3d headCoordinate, cv::Vec6d transferHeadToBase)
    {
        cv::Matx33d rm;
        cv::Rodrigues(cv::Vec3d(transferHeadToBase[3], transferHeadToBase[4], transferHeadToBase[5]), rm);
        cv::Vec3d baseCoordinate=cv::Vec3d(transferHeadToBase[0], transferHeadToBase[1], transferHeadToBase[2]) + rm*headCoordinate;
        return baseCoordinate;
    }


    static cv::Vec4d GetPlane(cv::Vec3d point[3])
    {
        cv::Vec3d normal=(point[2]-point[0]).cross(point[1]-point[0]);
        double d=-normal.ddot(point[0]);
        return cv::Vec4d(normal[0],normal[1],normal[2],d);
    }


    static cv::Matx23d GetHomography2D(cv::Vec2d ptFrom[3], cv::Vec2d ptTo[3])
    {
        cv::Vec6d B(ptTo[0][0], ptTo[0][1], ptTo[1][0], ptTo[1][1], ptTo[2][0], ptTo[2][1]);
        double aptr[36]={
            ptFrom[0][0], ptFrom[0][1], 1., 0., 0., 0.,
            0., 0., 0., ptFrom[0][0], ptFrom[0][1], 1.,
            ptFrom[1][0], ptFrom[1][1], 1., 0., 0., 0.,
            0., 0., 0., ptFrom[1][0], ptFrom[1][1], 1.,
            ptFrom[2][0], ptFrom[2][1], 1., 0., 0., 0.,
            0., 0., 0., ptFrom[2][0], ptFrom[2][1], 1.
        };
        cv::Matx66d A(aptr);

        cv::Vec6d X=A.inv()*B;

        cv::Matx23d H(X[0],X[1],X[2],X[3],X[4],X[5]);

        return H;
    }


    static cv::Vec2d PerTF(cv::Vec2d p, cv::Matx33d H)
    {
        cv::Vec3d po=H*cv::Vec3d(p[0],p[1],1);
        return cv::Vec2d(po[0]/po[2],po[1]/po[2]);
    }


    static cv::Vec3d PixelCoordinateToBaseCoordinate(cv::Vec2d pixelCoordinate, cv::Matx33d H, cv::Vec4d plane)
    {
        //    cv::Vec2d baseXY=H*cv::Vec3d(pixelCoordinate[0],pixelCoordinate[1],1);
        cv::Vec2d baseXY=PerTF(pixelCoordinate, H);
        double baseZ=(plane[3]+plane[0]*baseXY[0]+plane[1]*baseXY[1])/(-plane[2]);
        return cv::Vec3d(baseXY[0],baseXY[1],baseZ);
    }


    static cv::Vec6d ComputeTransform(const cv::Vec3d ptFrom[4], const cv::Vec3d ptTo[4])
    {
        cv::Vec3d ctFrom = (ptFrom[0] + ptFrom[1] + ptFrom[2] + ptFrom[3]) / 4;
        cv::Vec3d ctTo = (ptTo[0] + ptTo[1] + ptTo[2] + ptTo[3]) / 4;
        cv::Matx33d H = (ptFrom[0] - ctFrom)*((ptTo[0] - ctTo).t());
        H += (ptFrom[1] - ctFrom)*((ptTo[1] - ctTo).t());
        H += (ptFrom[2] - ctFrom)*((ptTo[2] - ctTo).t());
        H += (ptFrom[3] - ctFrom)*((ptTo[3] - ctTo).t());
        cv::Mat U, S, Vt;
        cv::SVDecomp(H, S, U, Vt);
        cv::Mat R0 = (U*Vt).t();
        cv::Matx33d R((double*)R0.ptr());
        cv::Vec3d rxyz;
        cv::Rodrigues(R, rxyz);
        cv::Vec3d xyz = ctTo - R*ctFrom;
        return cv::Vec6d(xyz[0], xyz[1], xyz[2], rxyz[0], rxyz[1], rxyz[2]);
    }

    static cv::Vec3d LineEquation(cv::Vec2d point1, cv::Vec2d point2)
    {
        return cv::Vec3d(point1[0],point1[1],1).cross(cv::Vec3d(point2[0],point2[1],1));
    }

    static cv::Vec2d RectangleCornerToCenter(cv::Vec2d pixelCorner[4])
    {
        cv::Vec3d diagonal[2]={
            LineEquation(pixelCorner[0], pixelCorner[2]),
            LineEquation(pixelCorner[1], pixelCorner[3])
        };

        cv::Vec3d intersection=diagonal[0].cross(diagonal[1]);
        return cv::Vec2d(intersection[0]/intersection[2], intersection[1]/intersection[2]);
    }


    static void SetHeadCoordinateUnitAxis(cv::Vec3d OXYZ[4], double handLength)
    {
        double invsqrt2=1/sqrt(2.0);

        OXYZ[0]=cv::Vec3d(0,0,0);
        OXYZ[1]=cv::Vec3d(1,0,0);
        OXYZ[2]=cv::Vec3d(0,1,0);
        OXYZ[3]=cv::Vec3d(0,0,1);

        OXYZ[1]=cv::Vec3d(invsqrt2,-invsqrt2,0);
        OXYZ[2]=cv::Vec3d(invsqrt2,invsqrt2,0);


        OXYZ[0][2]+=handLength;
        OXYZ[1][2]+=handLength;
        OXYZ[2][2]+=handLength;
        OXYZ[3][2]+=handLength;
    }

    static void RectangleCornerPixelCoordinateToCenterBaseCoordinateUnitAxis(cv::Vec3d OXYZ[4], cv::Vec2d pixelCorner[4], cv::Matx33d H, cv::Vec4d plane)
    {
        cv::Vec2d centerPixelCoordinate=RectangleCornerToCenter(pixelCorner);

        cv::Vec3d centerBaseCoordiante=PixelCoordinateToBaseCoordinate(centerPixelCoordinate, H, plane);

        cv::Vec3d cornerBaseCoordiante[4]={
            PixelCoordinateToBaseCoordinate(pixelCorner[0], H, plane),
            PixelCoordinateToBaseCoordinate(pixelCorner[1], H, plane),
            PixelCoordinateToBaseCoordinate(pixelCorner[2], H, plane),
            PixelCoordinateToBaseCoordinate(pixelCorner[3], H, plane)
        };

        cv::Vec3d edgeVector[4]={
            cornerBaseCoordiante[0]-cornerBaseCoordiante[1],
            cornerBaseCoordiante[1]-cornerBaseCoordiante[2],
            cornerBaseCoordiante[2]-cornerBaseCoordiante[3],
            cornerBaseCoordiante[3]-cornerBaseCoordiante[0]
        };

        double edgeLength[4]={
            cv::norm(edgeVector[0]),
            cv::norm(edgeVector[1]),
            cv::norm(edgeVector[2]),
            cv::norm(edgeVector[3])
        };

        size_t longest=std::max_element(edgeLength, edgeLength+4)-edgeLength;

        cv::Vec3d xUnitAxis=edgeVector[longest]/edgeLength[longest];

        cv::Vec3d zUnitAxis(plane[0],plane[1],plane[2]);
        double zLength=cv::norm(zUnitAxis);

        if(zUnitAxis.ddot(cv::Vec3d(0,0,1))>0)
            zUnitAxis/=-zLength;
        else
            zUnitAxis/=zLength;

        cv::Vec3d yUnitAxis=zUnitAxis.cross(xUnitAxis);

        OXYZ[0]=centerBaseCoordiante;
        OXYZ[1]=centerBaseCoordiante+xUnitAxis;
        OXYZ[2]=centerBaseCoordiante+yUnitAxis;
        OXYZ[3]=centerBaseCoordiante+zUnitAxis;

        //    OXYZ[0]=cornerBaseCoordiante[0];
        //    OXYZ[1]=cornerBaseCoordiante[0]+xUnitAxis;
        //    OXYZ[2]=cornerBaseCoordiante[0]+yUnitAxis;
        //    OXYZ[3]=cornerBaseCoordiante[0]+zUnitAxis;
    }


    static cv::Vec6d RectangleCornerPixelCoordinateToHeadPos(double handLength, cv::Vec2d pixelCorner[4], cv::Matx33d H, cv::Vec4d plane)
    {
        cv::Vec3d OXYZBase[4];
        RectangleCornerPixelCoordinateToCenterBaseCoordinateUnitAxis(OXYZBase, pixelCorner, H, plane);
        cv::Vec3d OXYZHead[4];
        SetHeadCoordinateUnitAxis(OXYZHead, handLength);
        return ComputeTransform(OXYZHead, OXYZBase);
    }


    static bool IsInsidePolygon(cv::Vec2d point, cv::Vec2d rectangleCorner[4])
    {
        cv::Vec2d rectangleCenter=RectangleCornerToCenter(rectangleCorner);
        cv::Vec3d rectangleEdge[4]={
            LineEquation(rectangleCorner[0], rectangleCorner[1]),
            LineEquation(rectangleCorner[1], rectangleCorner[2]),
            LineEquation(rectangleCorner[2], rectangleCorner[3]),
            LineEquation(rectangleCorner[3], rectangleCorner[0])
        };
        if(rectangleEdge[0].ddot(cv::Vec3d(rectangleCenter[0],rectangleCenter[1],1))<0)
            rectangleEdge[0]=-rectangleEdge[0];
        if(rectangleEdge[1].ddot(cv::Vec3d(rectangleCenter[0],rectangleCenter[1],1))<0)
            rectangleEdge[1]=-rectangleEdge[1];
        if(rectangleEdge[2].ddot(cv::Vec3d(rectangleCenter[0],rectangleCenter[1],1))<0)
            rectangleEdge[2]=-rectangleEdge[2];
        if(rectangleEdge[3].ddot(cv::Vec3d(rectangleCenter[0],rectangleCenter[1],1))<0)
            rectangleEdge[3]=-rectangleEdge[3];

        if( rectangleEdge[0].ddot(cv::Vec3d(point[0],point[1],1))>0 &&
                rectangleEdge[1].ddot(cv::Vec3d(point[0],point[1],1))>0 &&
                rectangleEdge[2].ddot(cv::Vec3d(point[0],point[1],1))>0 &&
                rectangleEdge[3].ddot(cv::Vec3d(point[0],point[1],1))>0 )
            return true;
        return false;
    }

    static cv::Mat PolygonMask(cv::Size maskSize, cv::Vec2d rectangleCorner[4])
    {
        cv::Mat mask(maskSize,CV_8UC1);

        for(int y=0;y<mask.rows;y++)
        {
            for(int x=0;x<mask.cols;x++)
            {
                if(IsInsidePolygon(cv::Vec2d(x,y),rectangleCorner))
                    mask.at<uchar>(y,x)=255;
                else
                    mask.at<uchar>(y,x)=0;
            }
        }
        return mask;
    }


    static bool Locate(cv::Vec2d paperCorner[4], const cv::Mat &frame, const cv::Mat &mask, double thres)
    {
        cv::Mat paper;
        cv::threshold(frame,paper,thres,255,cv::THRESH_BINARY);
        cv::bitwise_and(paper, mask, paper);
        std::vector<cv::Point> bgCandidate(1,cv::Point(0,0));
        cv::Mat bw;
        ObjectMask::FloodFromPoints(bw, paper, bgCandidate);
        //    cv::bitwise_not(bw, bw);
        cv::namedWindow("bw", cv::WINDOW_NORMAL);
        cv::imshow("bw",bw);
        cv::waitKey(1);
        cv::Point2f corner[4];
        if(!DeskA::FindDesk(bw, corner))
            return false;
        for(size_t i=0;i<4;i++)
        {
            paperCorner[i][0]=corner[i].x;
            paperCorner[i][1]=corner[i].y;
        }
        return true;
    }

public:
    cv::Vec4d plane;
    cv::Matx33d H;
    cv::Mat mask;

    URControlStamp(std::string ipPC="192.168.1.14", std::string ipUR="192.168.1.13", unsigned short portPC=1235, unsigned short portUR=30003, unsigned short modbusPort=502)
        : URControlA(ipPC, ipUR, portPC, portUR, modbusPort)
    {
        double handLength=0.21;

        //    cv::Vec3d referenceBase[4]={
        //        HeadToBase(cv::Vec3d(0,0,handLength), cv::Vec6d(0.512809,0.25732,0.210275,-2.77142,0.129088,-0.518529)),
        //        HeadToBase(cv::Vec3d(0,0,handLength), cv::Vec6d(0.128074,0.435648,0.166315,2.21754,-1.41892,0.89441)),
        //        HeadToBase(cv::Vec3d(0,0,handLength), cv::Vec6d(0.134422,-0.17449,0.20477,-1.11165,2.63452,-0.476487)),
        //        HeadToBase(cv::Vec3d(0,0,handLength), cv::Vec6d(0.510393,-0.231586,0.210497,-2.66326,0.952798,-0.559089))
        //    };

        //        cv::Vec2d referencePixel1[4]={
        //            cv::Vec2d(90,72),
        //            cv::Vec2d(146,342),
        //            cv::Vec2d(506,341),
        //            cv::Vec2d(558,69)
        //        };



        cv::Vec3d referenceBase[4]={
            HeadToBase(cv::Vec3d(0,0,handLength), cv::Vec6d(0.512474,-0.187912,0.226423,-2.98454,0.697973,-0.399517)),
            HeadToBase(cv::Vec3d(0,0,handLength), cv::Vec6d(0.214651,-0.197416,0.225197,3.13498,0.0731075,0.0959954)),
            HeadToBase(cv::Vec3d(0,0,handLength), cv::Vec6d(0.228239,0.296933,0.225976,2.36381,-2.06457,0.0251457)),
            HeadToBase(cv::Vec3d(0,0,handLength), cv::Vec6d(0.496013,0.254633,0.21989,-2.88894,-0.0586657,-0.468388))
        };



        cv::Vec2d referencePixel[4]={
            cv::Vec2d(807,148)/1.5,
            cv::Vec2d(744,492)/1.5,
            cv::Vec2d(234,485)/1.5,
            cv::Vec2d(170,148)/1.5
        };


        std::cout<<referenceBase[0]<<"\n"
                                  <<referenceBase[1]<<"\n"
                                 <<referenceBase[2]<<"\n"
                                <<referenceBase[3]<<"\n"
                               <<std::flush;


        plane=GetPlane(referenceBase);

        cv::Vec2d referenceBaseXY[4]={
            cv::Vec2d(referenceBase[0][0],referenceBase[0][1]),
            cv::Vec2d(referenceBase[1][0],referenceBase[1][1]),
            cv::Vec2d(referenceBase[2][0],referenceBase[2][1]),
            cv::Vec2d(referenceBase[3][0],referenceBase[3][1])
        };

        cv::Point2f referencePixelp[4]={
            cv::Point2f(referencePixel[0][0], referencePixel[0][1]),
            cv::Point2f(referencePixel[1][0], referencePixel[1][1]),
            cv::Point2f(referencePixel[2][0], referencePixel[2][1]),
            cv::Point2f(referencePixel[3][0], referencePixel[3][1])
        };

        cv::Point2f referenceBaseXYp[4]={
            cv::Point2f(referenceBaseXY[0][0], referenceBaseXY[0][1]),
            cv::Point2f(referenceBaseXY[1][0], referenceBaseXY[1][1]),
            cv::Point2f(referenceBaseXY[2][0], referenceBaseXY[2][1]),
            cv::Point2f(referenceBaseXY[3][0], referenceBaseXY[3][1])
        };

        cv::Mat HH=cv::getPerspectiveTransform(referencePixelp, referenceBaseXYp);

        H=cv::Matx33d(HH);

        //    cv::Matx23d H=GetHomography2D(referencePixel, referenceBaseXY);

        std::cout<<"plane="<<plane<<"\n"<<std::flush;

        //    std::cout<<"H="<<H<<"\n"<<std::flush;

        std::cout<<plane.ddot(cv::Vec4d(referenceBase[0][0],referenceBase[0][1],referenceBase[0][2],1))<<"\n";
        std::cout<<plane.ddot(cv::Vec4d(referenceBase[1][0],referenceBase[1][1],referenceBase[1][2],1))<<"\n";
        std::cout<<plane.ddot(cv::Vec4d(referenceBase[2][0],referenceBase[2][1],referenceBase[2][2],1))<<"\n";
        std::cout<<plane.ddot(cv::Vec4d(referenceBase[3][0],referenceBase[3][1],referenceBase[3][2],1))<<"\n";



        //    std::cout<<cv::Vec2d(referenceBase[0][0],referenceBase[0][1])-H*cv::Vec3d(referencePixel[0][0], referencePixel[0][1], 1)<<"\n";
        //    std::cout<<cv::Vec2d(referenceBase[1][0],referenceBase[1][1])-H*cv::Vec3d(referencePixel[1][0], referencePixel[1][1], 1)<<"\n";
        //    std::cout<<cv::Vec2d(referenceBase[2][0],referenceBase[2][1])-H*cv::Vec3d(referencePixel[2][0], referencePixel[2][1], 1)<<"\n";
        //    std::cout<<cv::Vec2d(referenceBase[3][0],referenceBase[3][1])-H*cv::Vec3d(referencePixel[3][0], referencePixel[3][1], 1)<<"\n";

        std::cout<<referenceBaseXY[0]-PerTF(referencePixel[0],H)<<"\n";
        std::cout<<referenceBaseXY[1]-PerTF(referencePixel[1],H)<<"\n";
        std::cout<<referenceBaseXY[2]-PerTF(referencePixel[2],H)<<"\n";
        std::cout<<referenceBaseXY[3]-PerTF(referencePixel[3],H)<<"\n";

        mask=PolygonMask(cv::Size(640,480),referencePixel);
//        cv::imshow("mask",mask);
//        cv::waitKey(1);

    }


    cv::Vec6d PointToPaper(const cv::Mat gray, double grayThreshold=128, double printLength=0.235)
    {

        cv::Vec2d paperPixel[4];
        if(!Locate(paperPixel, gray, mask, grayThreshold))
        {
            std::cout<<"fail to lacate paper\n"<<std::flush;
            return cv::Vec6d(0,0,0,0,0,0);
        }

        std::cout<<paperPixel[0]<<"\n"
                               <<paperPixel[1]<<"\n"
                              <<paperPixel[2]<<"\n"
                             <<paperPixel[3]<<"\n"
                            <<std::flush;

        cv::Vec6d pos=RectangleCornerPixelCoordinateToHeadPos(printLength, paperPixel, H, plane);
        return pos;
    }

    void Touch()
    {
        cv::VideoCapture c(1);

        cv::Mat frame;
        c.read(frame);

        std::cout<<frame.size()<<"\n";

        c.release();

        cv::Mat gray;
        cv::cvtColor(frame,gray,CV_BGR2GRAY);
        cv::imshow("gray", gray);
        cv::waitKey(1);

        cv::Vec6d pos=PointToPaper(gray);
        SetPose(pos[0],pos[1],pos[2]+0.04,pos[3],pos[4],pos[5], .8);

        SetPose(pos[0],pos[1],pos[2]-0.01,pos[3],pos[4],pos[5], .4);

        SetPose(pos[0],pos[1],pos[2]+0.04,pos[3],pos[4],pos[5], .4);

        SetPose(-0.0489175,0.367352,0.340857,1.05044,-2.94398,-0.0184143, .8);
    }

    void ForceTouch()
    {
        cv::VideoCapture c(0);

        cv::Mat frame;
        for(int i=0;i<10;i++)
            c.read(frame);

        std::cout<<frame.size()<<"\n";

        c.release();

        if(frame.empty())
        {
            std::cout<<"fail to get frame\n"<<std::flush;
            return;
        }

        cv::Mat gray;
        cv::cvtColor(frame,gray,CV_BGR2GRAY);
        cv::namedWindow("gray", cv::WINDOW_NORMAL);
        cv::imshow("gray", gray);
        cv::waitKey(1);

        cv::Vec6d pos=PointToPaper(gray);

        if(pos.ddot(pos)>1)
        {

            SetPose(pos[0],pos[1],pos[2]+0.05,pos[3],pos[4],pos[5], .2);

            cv::waitKey(300);

            LoadAndPlay("fff");

            //        cv::waitKey(500);

            SetPose(pos[0],pos[1],pos[2]+0.05,pos[3],pos[4],pos[5], .1);
        }

        SetPose(-0.0489175,0.367352,0.340857,1.05044,-2.94398,-0.0184143, .2);
    }

};

#endif // URCONTROLSTAMP_HPP
