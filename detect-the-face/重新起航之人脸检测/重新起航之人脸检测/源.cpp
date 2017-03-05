#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;
CascadeClassifier dface;
void DetectandDraw(Mat frame);
int main()
{
	VideoCapture cap(0);
	Mat frame;
	if (!dface.load("haarcascade_frontalface_alt2.xml")){ cout << "error load xml!"; return -1; }
	if (cap.isOpened())
	{
		while (1)
		{
			cap >> frame;
			if (!frame.empty())
			{
				DetectandDraw(frame);

			}
			waitKey(10);
		}
	}
	return 0;
}
void DetectandDraw(Mat frame)
{
	Mat frame_gray;
	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	vector<Rect>faces;
	equalizeHist(frame_gray, frame_gray);
	dface.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
	for (size_t i = 0; i < faces.size(); i++)
	{
		Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
		ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 2, 8, 0);
	}
	imshow("ÈËÁ³¼ì²â", frame);
}