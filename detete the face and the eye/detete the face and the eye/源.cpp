//��̬�棬�������۾����
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;
CascadeClassifier dface;
CascadeClassifier deye;
void DetectandDraw(Mat frame);
int main()
{
	Mat frame;
	if (!dface.load("haarcascade_frontalface_alt.xml")){ cout << "error load xml!"; return -1; }
	if (!deye.load("haarcascade_eye_tree_eyeglasses.xml")){ cout << "error load xml!"; return -1; }
		
	frame = imread("d:\\img.jpg");  //�������ʶ�Ȼ����ǵͣ�����ѵ���ļ������⣬����÷��ű�׼����
			if (!frame.empty())
			{
				DetectandDraw(frame);

			}
			waitKey(10);
		
	
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
		Mat faceROI = frame_gray(faces[i]);
		vector<Rect>eyes;
		deye.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
		for (size_t j = 0; j < eyes.size(); j++)
		{
			Point eye_center(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
			int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);   //����뾶
			circle(frame, eye_center, radius, Scalar(255, 0, 0), 3, 8, 0);
		}


	}
	imshow("�������", frame);
	waitKey(0);
}