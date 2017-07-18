#pragma once

#include "desk.hpp"
#include "objectmask.hpp"
#include "rdp.hpp"

class DeskA : public Desk
{
public:
	std::vector<cv::Point> deskCandidate;
	std::vector<cv::Point> bgCandidate;

	DeskA()
		: Desk()
	{
		deskCandidate.assign(1, cv::Point(110, 390));
		bgCandidate.assign(2, cv::Point());
		bgCandidate[0] = cv::Point(1, 1);
		bgCandidate[1] = cv::Point(630, 1);
	}
	

	static cv::Point2f SegmentIntersection(cv::Point v1a, cv::Point v1b, cv::Point v2a, cv::Point v2b)
	{
		cv::Vec3d l[2];

		RDP<int>::LinePara(l[0][0], l[0][1], l[0][2], v1a.x, v1a.y, v1b.x, v1b.y);
		RDP<int>::LinePara(l[1][0], l[1][1], l[1][2], v2a.x, v2a.y, v2b.x, v2b.y);

		cv::Vec3d ih = l[0].cross(l[1]);

		return cv::Point2f(ih[0] / ih[2], ih[1] / ih[2]);
	}

	static void EdgeToVertex(cv::Point2f vertex[4], const std::vector<cv::Point> &edge, const size_t segmentIndex[4], bool isClose = false)
	{
		size_t lastIndex = segmentIndex[3] + 1;
		if (lastIndex == edge.size() && isClose)
			lastIndex = 0;

		if (segmentIndex[0] == lastIndex)
		{
			vertex[0] = edge[segmentIndex[0]];
		}
		else
		{
			vertex[0] = SegmentIntersection(edge[segmentIndex[0]], edge[segmentIndex[0] + 1], edge[segmentIndex[3]], edge[lastIndex]);
		}

		for (int i = 1;i < 4;i++)
		{
			if (segmentIndex[i - 1] + 1 == segmentIndex[i])
			{
				vertex[i] = edge[segmentIndex[i]];
			}
			else
			{
				vertex[i] = SegmentIntersection(edge[segmentIndex[i - 1]], edge[segmentIndex[i - 1] + 1], edge[segmentIndex[i]], edge[segmentIndex[i] + 1]);
			}
		}
	}

	void DeskMask(const cv::Mat &frame, cv::Mat &mask)
	{
		cv::Mat bw;
		ObjectMask::FloodFromPoints(bw, frame, deskCandidate);

		//cv::imshow("bw", bw);

		int connectivity = 4;
		int newMaskVal = 255;
		int flags = connectivity + (newMaskVal << 8) +
			CV_FLOODFILL_FIXED_RANGE +
			CV_FLOODFILL_MASK_ONLY;

		ObjectMask::FloodFromPoints(mask, bw, bgCandidate, true, flags, cv::Scalar(), cv::Scalar());

		cv::imshow("deskmask", mask);
	}

	void PlateMask(const cv::Mat &frame, cv::Mat &mask)
	{
		cv::Mat bw;
		ObjectMask::FloodFromPoints(bw, frame, deskCandidate);

#if 0
		cv::Point plateCandidate = cv::Point(320, 240);

		int connectivity = 4;
		int newMaskVal = 255;
		int flags = connectivity + (newMaskVal << 8) +
			CV_FLOODFILL_FIXED_RANGE +
			CV_FLOODFILL_MASK_ONLY;
		ObjectMask::FloodFromPoints(mask, bw, std::vector<cv::Point>(1, plateCandidate), false, flags, cv::Scalar(), cv::Scalar());
#else

		int connectivity = 4;
		int newMaskVal = 255;
		int flags = connectivity + (newMaskVal << 8) +
			CV_FLOODFILL_FIXED_RANGE +
			CV_FLOODFILL_MASK_ONLY;

		ObjectMask::FloodFromPoints(mask, bw, bgCandidate, true, flags, cv::Scalar(), cv::Scalar());

		cv::bitwise_and(mask, bw, mask);

#endif

		cv::imshow("platemask", mask);
	}


        static bool FindDesk(const cv::Mat &bw, cv::Point2f desk[4])
	{
		std::vector<cv::Point> contour;

		ObjectMask::LargestComponentContour(contour, bw);

		if (contour.size() < 10)
                        return false;

		std::vector<size_t> segindex;

		cv::Mat ttp;
		cv::Mat(contour).copyTo(ttp);

		std::cout << "findContours!\n";
		std::cerr << "";

		RDP<int>::Approx((int*)(ttp.data), contour.size(), segindex, 4);

		std::vector<cv::Point> cnt(segindex.size(), cv::Point());
		for (size_t i = 0;i < segindex.size();i++)
			cnt[i] = contour[segindex[i]];

		std::cout << "cnt!\n";
		std::cerr << "";

		size_t segStartIdx[4] = { 0,1,2,3 };

		EdgeToVertex(desk, cnt, segStartIdx);

		std::cout << "desk=" << desk[0] << '\t' << desk[1] << '\t' << desk[2] << '\t' << desk[3] << '\t' << "\n";
		//std::cout << "contours=" << contours.size() << "\n";
		std::cout << "cnt=" << cnt.size() << "\n";
		std::cerr << "";

                return true;

	}


	static bool FindPlate(const cv::Mat &bw, cv::Point2f plate[4])
	{
		std::vector<cv::Point> contour;

		ObjectMask::LargestComponentContour(contour, bw);

		if (contour.size()<5)
			return false;

		std::vector<size_t> segindex;

		cv::Mat ttp;
		cv::Mat(contour).copyTo(ttp);

		RDP<int>::ApproxClose((int*)(ttp.data), contour.size(), segindex, 4);

		std::vector<cv::Point> cnt(segindex.size(), cv::Point());
		for (size_t i = 0;i < segindex.size();i++)
		{
			cnt[i] = contour[segindex[i]];
		}

		for (size_t i = 0;i < 4;i++)
		{
			plate[i] = cv::Point2f(cnt[i].x, cnt[i].y);
		}

		return true;		
	}


	void FindCenter(float &x, float &y, const cv::Mat &frame, float width = 0.5, float height = 0.53)
	{

		cv::Point2f desk[4];
		cv::Point2f plate[4];

		cv::Mat deskm;
		DeskMask(frame, deskm);

		FindDesk(deskm, desk);

		cv::arrowedLine(frame, desk[0], desk[1], cv::Scalar(0, 255, 0), 1);
		cv::arrowedLine(frame, desk[0], desk[3], cv::Scalar(255, 0, 0), 1);


		cv::Mat platem;
		PlateMask(frame, platem);

		FindPlate(platem, plate);

		cv::arrowedLine(frame, plate[0], plate[1], cv::Scalar(0, 255, 0), 1);
		cv::arrowedLine(frame, plate[0], plate[3], cv::Scalar(255, 0, 0), 1);

		cv::Point2f deskw[4];
		deskw[3].x = 0;
		deskw[3].y = 0;

		deskw[2].x = 0;
		deskw[2].y = height;

		deskw[1].x = width;
		deskw[1].y = height;

		deskw[0].x = width;
		deskw[0].y = 0;


		cv::Mat mp = cv::getPerspectiveTransform(desk, deskw);

		std::vector<cv::Vec2f> platew(4, cv::Vec2f());
		cv::perspectiveTransform(std::vector<cv::Vec2f>(plate, plate + 4), platew, mp);



		x = y = 0;
		for (int i = 0;i < 4;i++)
		{
			x += platew[i][0];
			y += platew[i][1];
		}

		x /= 4;
		y /= 4;

		std::cout << "platew=" << cv::Mat(platew) << '\n';

		std::cout << "mp=" << mp << '\n';
		std::cerr << "";


		cv::imshow("frame", frame);
	}

	void FindCenterA(float &x, float &y, const cv::Mat &frame)
	{
		std::vector<cv::Vec2f> platew(4, cv::Vec2f());
		Locate(platew, frame);

		x = y = 0;
		for (int i = 0;i < 4;i++)
		{
			x += platew[i][0];
			y += platew[i][1];
		}

		x /= 4;
		y /= 4;

		std::cout << "center=" << cv::Point2f(x, y) << '\n';
	}


	void Locate(std::vector<cv::Vec2f> &platew, const cv::Mat &frame)
	{
		cv::Point2f plate[4];

		cv::Mat platem;
		PlateMask(frame, platem);

		if (!FindPlate(platem, plate))
		{
			cv::imshow("frame", frame);
			cv::waitKey(1);
			platew.clear();
			return;
		}

		cv::arrowedLine(frame, plate[0], plate[1], cv::Scalar(0, 0, 255), 1);
		cv::arrowedLine(frame, plate[1], plate[2], cv::Scalar(0, 255, 0), 1);
		cv::arrowedLine(frame, plate[2], plate[3], cv::Scalar(255, 0, 0), 1);

		platew.assign(4, cv::Vec2f());
		cv::perspectiveTransform(std::vector<cv::Vec2f>(plate, plate + 4), platew, H);

		std::cout << "platew=" << cv::Mat(platew) << '\n';

		cv::imshow("frame", frame);

	}
};
