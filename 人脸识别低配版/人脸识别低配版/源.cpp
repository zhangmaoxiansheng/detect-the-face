//图片大小为295*412
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/contrib/contrib.hpp>

#include <iostream>
#include <stdio.h>
using namespace std;
using namespace cv;
Ptr<FaceRecognizer>face = createLBPHFaceRecognizer();
CascadeClassifier dface;
void DetectandDraw(Mat frame);
int main()
{
	Mat img1 = imread("D:\\img.jpg", 0);
	Mat img2 = imread("D:\\img23.jpg", 0);
	vector<Mat>images;
	vector<int>labels;
	images.push_back(img1);
	labels.push_back(1);
	images.push_back(img2);
	labels.push_back(0);
	face->train(images, labels);
	VideoCapture cap(0);
	Mat frame;
	if (!dface.load("haarcascade_frontalface_alt.xml")){ cout << "error load xml!"; return -1; }
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
		Mat ROI = frame_gray(faces[i]);
		Mat test;
		resize(ROI, test, Size(295, 412));
		int result = face->predict(test);
		if (result == 1){
			putText(frame, "zhangmaoxiansheng", Point(faces[i].x, faces[i].y), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 0, 255), 2);
		}
		else putText(frame, "not the cat", Point(faces[i].x, faces[i].y), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 0, 255), 2);
	}
	imshow("人脸检测", frame);
}