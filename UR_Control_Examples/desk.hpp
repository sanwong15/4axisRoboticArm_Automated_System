#pragma once

#include <opencv2/opencv.hpp>
class Desk
{
public:
	cv::Mat H;

	Desk()
	{		
		//Load(H, "C:\\Users\\jhanbin\\Documents\\Visual Studio 2015\\Projects\\cutter\\camera\\H2.yml");
		Load(H, "H2.yml");
	}

	static void ChessBoardPos(std::vector<cv::Point2f> &pos, int nx = 6, int ny = 9, float dx = 0.35, float dy = -0.15, float l = 0.04115)
	{
		pos.assign(nx*ny, cv::Point2f());
		for (int i = 0;i < ny;i++)
		{
			for (int j = 0;j < nx;j++)
			{
				int idx = j + i*nx;

				pos[idx].x = dx + j*l;
				pos[idx].y = dy + (ny - 1 - i)*l;
			}
		}
	}

	static bool ComputeH(cv::Mat &H, const cv::Mat &frame,
		cv::Size patternsize = cv::Size(6, 9), cv::Point2f br = cv::Point2f(0.35, -0.15), float l = 0.04115)
	{
		cv::Mat gray;
		cv::cvtColor(frame, gray, CV_BGR2GRAY);

		std::vector<cv::Point2f> corners; //this will be filled by the detected corners
										  //CALIB_CB_FAST_CHECK saves a lot of time on images
										  //that do not contain any chessboard corners
		bool patternfound = cv::findChessboardCorners(gray, patternsize, corners,
			cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE
			//    + cv::CALIB_CB_FAST_CHECK
		);

		if (!patternfound)
			return false;


		cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
			cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));

		std::vector<cv::Point2f> cornersW;

		ChessBoardPos(cornersW, patternsize.width, patternsize.height, br.x, br.y, l);

		H = cv::findHomography(corners, cornersW);

		std::cout << "H=" << H << '\n';

		cv::Mat canvas;
		frame.copyTo(canvas);
		cv::drawChessboardCorners(canvas, patternsize, cv::Mat(corners), patternfound);
		cv::imshow("FF", canvas);

		return !H.empty();
	}

	static bool Save(const cv::Mat &m, std::string fp)
	{
		cv::FileStorage fs(fp, CV_STORAGE_WRITE);
		if (fs.isOpened())
		{
			fs << "m" << m;
			fs.release();
			return true;
		}
		std::cout << "Error: can not save result\n";
		return false;
	}

	static bool Load(cv::Mat &m, std::string fp)
	{
		cv::FileStorage fs(fp, CV_STORAGE_READ);
		if (fs.isOpened())
		{
			fs["m"] >> m;
			fs.release();
			return true;
		}
		std::cout << "Failed to open file " << fp << '\n';
		return false;
	};

	static void Calib(int index = 0)
	{
		cv::VideoCapture c(index);
		if (c.isOpened())
		{
			c.set(CV_CAP_PROP_FRAME_WIDTH, 640);
			c.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
			c.set(CV_CAP_PROP_SETTINGS, 1);
			cv::Mat frame;

			if (c.read(frame))
			{

				cv::imwrite("cb.png", frame);
				cv::Mat H;
				if (ComputeH(H, frame))
				{
					Save(H, "H.yml");
					cv::waitKey();
				}
			}
		}
	}

	cv::Vec2f Recover(cv::Vec2f p)
	{
		cv::Mat pm = (cv::Mat_<double>(3, 1) << p[0], p[1], 1);
		cv::Mat om = H*pm;
		cv::Vec2f op(om.at<double>(0) / om.at<double>(2), om.at<double>(1) / om.at<double>(2));
		return op;
	}
	
};
