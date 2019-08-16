#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

Mat imgOriginal;
Mat imgHLS;
Mat imgThresholded;
int frameCounter = 1;
//these two vectors needed for output of findContours
vector< vector<Point> > contours;
vector<Vec4i> hierarchy;

int lowH = 0;
int highH = 180;

int lowS = 0;
int highS = 255;

int lowV = 0;
int highV = 255;

int lowL = 0;
int highL = 255;



void createTrackbars() {
	namedWindow("Control", WINDOW_NORMAL); //create a window called "Control"

	//Create trackbars in "Control" window
	cvCreateTrackbar("LowH", "Control", &lowH, 180); //Hue (0 - 180)
	cvCreateTrackbar("HighH", "Control", &highH, 180);

	cvCreateTrackbar("LowS", "Control", &lowS, 255); //Saturation (0 - 255)
	cvCreateTrackbar("HighS", "Control", &highS, 255);

	cvCreateTrackbar("LowV", "Control", &lowV, 255); //Value (0 - 255)
	cvCreateTrackbar("HighV", "Control", &highV, 255);

	cvCreateTrackbar("LowL", "Control", &lowL, 255); //Lightness (0 - 255)
	cvCreateTrackbar("HighL", "Control", &highL, 255);
}

void morphOpen(Mat &thresh) {
	//morphological opening (remove small objects from the foreground)
	erode(thresh, thresh, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
	dilate(thresh, thresh, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
}

void morphClose(Mat &thresh) {
	//morphological closing (fill small holes in the foreground)
	dilate(thresh, thresh, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
	erode(thresh, thresh, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
}

void detectHLSthresholds() {
	cvtColor(imgOriginal, imgHLS, COLOR_BGR2HLS); //Convert the captured frame from BGR to HLS
	inRange(imgHLS, Scalar(lowH, lowL, lowS), Scalar(highH, highL, highS), imgThresholded); //Threshold the image
	//inRange(imgHLS, Scalar(87, 230, 255), Scalar(94, 255, 255), imgThresholded); //Threshold for party_-2_l_l.mp4 green
	//inRange(imgHLS, Scalar(71, 169, 255), Scalar(98, 255, 255), imgThresholded); //Threshold for auto-darker3.mp4 blue
	//morphological operations
	morphClose(imgThresholded);
	morphOpen(imgThresholded);
	//find contours of filtered image using openCV findContours function
	findContours(imgThresholded, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	drawContours(imgOriginal, contours, -1, Scalar(255, 0, 255), 2, 8, hierarchy);

	////Calculate the moments of the thresholded image
	//Moments oMoments = moments(imgThresholded);
	//double m01 = oMoments.m01;
	//double m10 = oMoments.m10;
	//double area = oMoments.m00;
	//// if the area <= X, I consider that there are no object in the image and it's because of the noise, the area is not zero 
	//if (area > 50)
	//{
	//	//calculate the position of the ball
	//	int posX = m10 / area;
	//	int posY = m01 / area;
	//}

	//inRange(imgHLS, Scalar(13, 130, 255), Scalar(52, 240, 255), imgThresholded); //Threshold for party_-2_l_l.mp4 red
	////inRange(imgHLS, Scalar(9, 150, 255), Scalar(49, 255, 255), imgThresholded); //Threshold for auto-darker3.mp4 red
	////morphological operations
	//morphClose(imgThresholded);
	//morphOpen(imgThresholded);
	////find contours of filtered image using openCV findContours function
	//findContours(imgThresholded, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	//drawContours(imgOriginal, contours, -1, Scalar(255, 0, 0), 2, 8, hierarchy);

	imshow("Thresholded Image", imgThresholded); //show the thresholded image
}

void trackCamshift() {

}

int main()
{
	//VideoCapture cap(0); //capture the video from web cam
	VideoCapture cap("party_-2_l_l.mp4"); //video file
	//VideoCapture cap("auto-darker3.mp4");

	if (!cap.isOpened())  // if not success, exit program
	{
		cout << "Cannot open the web cam / file" << endl;
		return -1;
	}

	cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 360);

	createTrackbars();

	while (true)
	{
		bool frameRead = cap.read(imgOriginal); // read a new frame from video
		if (!frameRead) //if fail, break loop
		{
			cout << "Cannot read a frame from video stream" << endl;
			break;
		}
		frameCounter++;
		if (frameCounter == cap.get(CV_CAP_PROP_FRAME_COUNT)) //if end of video file reached, start from first frame
		{
			frameCounter = 1;
			cap.set(CV_CAP_PROP_POS_FRAMES, 0);
			cout << "Video loop" << endl;
		}

		resize(imgOriginal, imgOriginal, Size(640, 360), 0, 0, INTER_CUBIC); //resize to 640 by 360

		//detectHLSthresholds(); //show regions of specified HLS values

		trackCamshift();

		imshow("Original", imgOriginal); //show the original image

		//if (waitKey(0) == 32) //frame by frame with 'space'
		//{
		//	cout << "space key is pressed by user" << endl;
		//}
		if (waitKey(1) == 27) //wait for 'esc' key (27) press => break loop
		{
			cout << "esc key is pressed by user" << endl;
			break;
		}
	}

	return 0;
}