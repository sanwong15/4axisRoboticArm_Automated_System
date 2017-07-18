#ifndef OBJECTMASK_HPP
#define OBJECTMASK_HPP

#include <opencv2/opencv.hpp>
#include <vector>

class ObjectMask
{
public:

    static void EdgePoints(std::vector<cv::Point> &edgePt, cv::Size sz)
    {
        edgePt.assign(2*sz.height,cv::Point());

        for(int r=0;r<sz.height;r++)
        {
            edgePt[2*r].x=0;
            edgePt[2*r].y=r;
            edgePt[2*r+1].x=sz.width-1;
            edgePt[2*r+1].y=r;
        }
    }

    static void FloodFromPoints(cv::Mat &out, const cv::Mat &in, const std::vector<cv::Point> &candidates,
                                bool inverse=true, int flags=4 + (255 << 8) + CV_FLOODFILL_MASK_ONLY,
                                cv::Scalar loDiff=cv::Scalar::all(6), cv::Scalar upDiff=cv::Scalar::all(6))
    {
        cv::Mat mask(in.rows+2,in.cols+2,CV_8UC1,cv::Scalar(0));
        cv::Mat mask1(in.rows+2,in.cols+2,CV_8UC1,cv::Scalar(0));

        for(size_t r=0;r<candidates.size();r++)
        {
            if(mask.at<uchar>(candidates[r].y+1,candidates[r].x+1)==0)
            {
                mask1.setTo(0);
                int area=cv::floodFill(in,mask1,candidates[r],cv::Scalar(255),NULL,loDiff,upDiff,flags);
                cv::bitwise_or(mask,mask1,mask);
            }
        }

        if(inverse)
            cv::bitwise_not(mask(cv::Rect(1,1,in.cols,in.rows)),out);
        else
            mask(cv::Rect(1,1,in.cols,in.rows)).copyTo(out);
        //cv::imshow("res",out);
    }


    static void Foreground(const cv::Mat &frame, cv::Mat &mask,
                           cv::Scalar loDiff=cv::Scalar::all(6), cv::Scalar upDiff=cv::Scalar::all(6))
    {
        std::vector<cv::Point> bgCandidates;
        EdgePoints(bgCandidates,frame.size());

        int connectivity = 4;
        int newMaskVal = 255;
        int flags = connectivity + (newMaskVal << 8) +
                //		CV_FLOODFILL_FIXED_RANGE +
                CV_FLOODFILL_MASK_ONLY;

        FloodFromPoints(mask,frame,bgCandidates,true,flags,loDiff,upDiff);
        cv::imshow("mask",mask);
    }


    static void FillHoleInside(cv::Mat &out, const cv::Mat &in)
    {

        std::vector<cv::Point> outCandidates;
        EdgePoints(outCandidates,in.size());

        int connectivity = 4;
        int newMaskVal = 255;
        int flags = connectivity + (newMaskVal << 8) +
                CV_FLOODFILL_FIXED_RANGE +
                CV_FLOODFILL_MASK_ONLY;

        FloodFromPoints(out,in,outCandidates,true,flags,cv::Scalar(),cv::Scalar());
    }


    static void LargestComponentContour(std::vector<cv::Point> &contour, const cv::Mat &mask)
    {
        std::vector<std::vector<cv::Point> > contours;
        std::vector<cv::Vec4i> hierarchy;

        cv::findContours( mask, contours, hierarchy, CV_RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

		if (contours.empty())
		{
			contour.clear();
			return;
		}

        std::vector<double> areas(contours.size(),0);

        for(size_t i=0;i<contours.size();i++)
            areas[i]=fabs(cv::contourArea(contours[i]));

        size_t maxIndex=std::max_element(areas.begin(),areas.end())-areas.begin();

        contour.assign(contours[maxIndex].begin(),contours[maxIndex].end());

    }


	static void NLargestComponentContour(std::vector< std::vector<cv::Point> > &contour, const cv::Mat &mask, int n)
	{
		std::vector<std::vector<cv::Point> > contours;
		std::vector<cv::Vec4i> hierarchy;

		cv::findContours(mask, contours, hierarchy, CV_RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

		std::vector<double> areas(contours.size(), 0);

		for (size_t i = 0; i<contours.size(); i++)
			areas[i] = fabs(cv::contourArea(contours[i]));

		contour.clear();

		for (int i = 0; i < n; i++)
		{
			if (areas.empty())
				return;
			size_t maxIndex = std::max_element(areas.begin(), areas.end()) - areas.begin();
			
			contour.push_back(std::vector<cv::Point>());
			contour.back().assign(contours[maxIndex].begin(), contours[maxIndex].end());

			areas.erase(areas.begin() + maxIndex);
			contours.erase(contours.begin() + maxIndex);
		}

	}




    static bool EraseSmallComponent(cv::Mat &out, const cv::Mat &in, double minArea)
    {
        std::vector<std::vector<cv::Point> > contours;
        std::vector<cv::Vec4i> hierarchy;


        cv::findContours( in, contours, hierarchy, CV_RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
        out.create(in.size(), CV_8UC1);
        out.setTo(0);

        bool found=false;
        // iterate through all the top-level contours,
        // draw each connected component with its own random color
        int idx = 0;
        for( ; idx >= 0 && !hierarchy.empty(); idx = hierarchy[idx][0] )
        {
            double area = fabs(cv::contourArea(contours[idx]));
            if(area>minArea)
            {
                cv::Scalar color(255);
                cv::drawContours( out, contours, idx, color, CV_FILLED, 8, hierarchy );
                found=true;
            }
        }

        cv::imshow("out",out);

        return found;
    }

};
#endif // OBJECTMASK_HPP
