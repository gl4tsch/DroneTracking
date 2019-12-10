#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <iostream>
#include <cmath>
#include <string>
#include "test.h"

using namespace std;
using namespace cv;

struct point_sorter { // less for points
	bool operator ()(const Point2d& a, const Point2d& b)
	{
		return (a.x < b.x);
	}
};

//ground truth
string pathToImgSequence("video_1.003_darkblue_new/");
int startFrame = 168; //0.006_darkblue_new: 192 || 1.003_darkblue: 168
vector<double> allFrameTimeStamps;
vector<vector<double>> allTruePoses;
vector<vector<double>> allSLERPedPoses;

int imgSizeX = 1280;
int imgSizeY = 1024;
float resScale = 1;
int imgResizeX;
int imgResizeY;
Mat image;
Mat imgDraw;
Mat imgHLS, imgHSV, imgYCrCb;
Mat imgThresholded;
Mat imgThreshC1, imgThreshC2;
int frameCounter = startFrame;
double t;
//these two vectors needed for output of findContours
vector< vector<Point> > contours, contours2;
vector<Vec4i> hierarchy, hierarchy2;

int lowH = 0;
int highH = 180;
int lowS = 0;
int highS = 255;
int lowV = 0;
int highV = 255;
int lowL = 0;
int highL = 255;

//camshift
bool selectObject = false;
Rect selection;
Point origin;
int trackObject1 = 0;
int trackObject2 = 0;
int lmin1 = 220;//220;
int lmax1 = 255;
int lmin2 = 220;//220;
int lmax2 = 255;
Rect trackWindow, trackWindow2;
int hsize = 16;
float hranges[] = { 0,180 };
const float* phranges = hranges;
Mat frame, hls, hue, hue2, mask, mask2, hist, hist2, histimg = Mat::zeros(200, 320, CV_8UC3), histimg2 = Mat::zeros(200, 320, CV_8UC3), backproj, backproj2, backprojImage, backprojImage2, cannyOut, cannyOut2;
Mat bothbackproj;
bool paused = false;
bool backprojMode = true;
bool showHist = true;
bool recordMode = false;
RotatedRect trackBox, trackBox2;
int thresh = 0;

//blob detect
Ptr<SimpleBlobDetector> detector;
vector<KeyPoint> keypoints;
int minGrey = 220; //intensity threshold for grey scale image. 240 for handy footage, 200 for 1.003 blue
int maxGrey = 255;

// oldBlobPoints -> newBlobPoints
vector<int> oldToNewBlobMatching;
// LED3D -> newBlobPoints
vector<int> LEDtoNewBlobMatching;
// LED3D -> oldBlobPoints
vector<int> LEDtoOldBlobMatching;

const int bufferSize = 5;
Point2d pointBuffer[bufferSize];

Mat imgGrey, imgGreyOld, imgBinary, imgBinaryOld, bgr;

//PnP
vector <Point2d> imagePoints2D;
vector <Point3d> modelPoints3D;
Mat cameraMatrix, distortCoeffs;
//rotation and translation output
Mat rotVec, transVec; //Rotation in axis-angle form

//Ransac
int iterationCount = 100;
float reprojectionError = 8.0f;
int minInliers = 5;
vector <int> inliersA;

//Optical Flow
vector<uchar> status;
vector<float> err;
TermCriteria term = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 10, 0.03);
vector<Point2f> newBlobPoints, oldBlobPoints, predictedBlobPoints;

//Kalman Filter
KalmanFilter KF;
int nStates = 18;		// the number of states
int nMeasurements = 6;	// the number of measured states
int nInputs = 0;		// the number of action control
double dt = 0.016;		// time between measurements (1/FPS) //0.125
// Instantiate estimated translation and rotation
cv::Mat translation_estimated(3, 1, CV_64F);
cv::Mat rotation_estimated(3, 3, CV_64F);
Mat measurements;

bool trackingLost = true;


class quat {
	public:
		double x, y, z, w;

		quat(void) {
			x = 0;
			y = 0;
			z = 0;
			w = 0;
		}

		quat(double xx, double yy, double zz, double ww) {
			x = xx;
			y = yy;
			z = zz;
			w = ww;
		}
};

void createTrackbars() {
	namedWindow("Control", WINDOW_NORMAL); //create a window called "Control"

	//Create trackbars in "Control" window
	createTrackbar("LowH", "Control", &lowH, 180); //Hue (0 - 180)
	createTrackbar("HighH", "Control", &highH, 180);

	createTrackbar("LowS", "Control", &lowS, 255); //Saturation (0 - 255)
	createTrackbar("HighS", "Control", &highS, 255);

	createTrackbar("LowV", "Control", &lowV, 255); //Value (0 - 255)
	createTrackbar("HighV", "Control", &highV, 255);

	createTrackbar("LowL", "Control", &lowL, 255); //Lightness (0 - 255)
	createTrackbar("HighL", "Control", &highL, 255);
}

void morphOpen(Mat &thresh) {
	//morphological opening (remove small objects from the foreground)
	erode(thresh, thresh, getStructuringElement(MORPH_RECT, Size(3, 3)));
	dilate(thresh, thresh, getStructuringElement(MORPH_RECT, Size(3, 3)));
}

void morphClose(Mat &thresh) {
	//morphological closing (fill small holes in the foreground)
	dilate(thresh, thresh, getStructuringElement(MORPH_RECT, Size(3, 3)));
	erode(thresh, thresh, getStructuringElement(MORPH_RECT, Size(3, 3)));
}

float euclideanDist(Point2f& p, Point2f& q) {
	Point diff = p - q;
	return sqrt(diff.x*diff.x + diff.y*diff.y);
}

quat slerp(quat qa, quat qb, double t) {
	//https://www.euclideanspace.com/maths/algebra/realNormedAlgebra/quaternions/slerp/index.htm
	// quaternion to return
	quat qm;
	// Calculate angle between them.
	double cosHalfTheta = qa.w * qb.w + qa.x * qb.x + qa.y * qb.y + qa.z * qb.z;
	// if qa=qb or qa=-qb then theta = 0 and we can return qa
	if (abs(cosHalfTheta) >= 1.0) {
		qm.w = qa.w; qm.x = qa.x; qm.y = qa.y; qm.z = qa.z;
		return qm;
	}
	// Calculate temporary values.
	double halfTheta = acos(cosHalfTheta);
	double sinHalfTheta = sqrt(1.0 - cosHalfTheta * cosHalfTheta);
	// if theta = 180 degrees then result is not fully defined
	// we could rotate around any axis normal to qa or qb
	if (fabs(sinHalfTheta) < 0.001) { // fabs is floating point absolute
		qm.w = (qa.w * 0.5 + qb.w * 0.5);
		qm.x = (qa.x * 0.5 + qb.x * 0.5);
		qm.y = (qa.y * 0.5 + qb.y * 0.5);
		qm.z = (qa.z * 0.5 + qb.z * 0.5);
		return qm;
	}
	double ratioA = sin((1 - t) * halfTheta) / sinHalfTheta;
	double ratioB = sin(t * halfTheta) / sinHalfTheta;
	//calculate Quaternion.
	qm.w = (qa.w * ratioA + qb.w * ratioB);
	qm.x = (qa.x * ratioA + qb.x * ratioB);
	qm.y = (qa.y * ratioA + qb.y * ratioB);
	qm.z = (qa.z * ratioA + qb.z * ratioB);
	return qm;
}


// Converts a given Euler angles to Rotation Matrix
// Convention used is Y-Z-X Tait-Bryan angles
// Reference:
// https://www.euclideanspace.com/maths/geometry/rotations/conversions/eulerToMatrix/index.htm
cv::Mat euler2rot(const cv::Mat & euler)
{
	cv::Mat rotationMatrix(3, 3, CV_64F);

	double bank = euler.at<double>(0);
	double attitude = euler.at<double>(1);
	double heading = euler.at<double>(2);

	// Assuming the angles are in radians.
	double ch = cos(heading);
	double sh = sin(heading);
	double ca = cos(attitude);
	double sa = sin(attitude);
	double cb = cos(bank);
	double sb = sin(bank);

	double m00, m01, m02, m10, m11, m12, m20, m21, m22;

	m00 = ch * ca;
	m01 = sh * sb - ch * sa*cb;
	m02 = ch * sa*sb + sh * cb;
	m10 = sa;
	m11 = ca * cb;
	m12 = -ca * sb;
	m20 = -sh * ca;
	m21 = sh * sa*cb + ch * sb;
	m22 = -sh * sa*sb + ch * cb;

	rotationMatrix.at<double>(0, 0) = m00;
	rotationMatrix.at<double>(0, 1) = m01;
	rotationMatrix.at<double>(0, 2) = m02;
	rotationMatrix.at<double>(1, 0) = m10;
	rotationMatrix.at<double>(1, 1) = m11;
	rotationMatrix.at<double>(1, 2) = m12;
	rotationMatrix.at<double>(2, 0) = m20;
	rotationMatrix.at<double>(2, 1) = m21;
	rotationMatrix.at<double>(2, 2) = m22;

	return rotationMatrix;
}

// Converts a given Rotation Matrix to Euler angles
// Convention used is Y-Z-X Tait-Bryan angles
// Reference code implementation:
// https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToEuler/index.htm
cv::Mat rot2euler(const cv::Mat & rotationMatrix)
{
	cv::Mat euler(3, 1, CV_64F);

	double m00 = rotationMatrix.at<double>(0, 0);
	double m02 = rotationMatrix.at<double>(0, 2);
	double m10 = rotationMatrix.at<double>(1, 0);
	double m11 = rotationMatrix.at<double>(1, 1);
	double m12 = rotationMatrix.at<double>(1, 2);
	double m20 = rotationMatrix.at<double>(2, 0);
	double m22 = rotationMatrix.at<double>(2, 2);

	double bank, attitude, heading;

	// Assuming the angles are in radians.
	if (m10 > 0.998) { // singularity at north pole
		bank = 0;
		attitude = CV_PI / 2;
		heading = atan2(m02, m22);
	}
	else if (m10 < -0.998) { // singularity at south pole
		bank = 0;
		attitude = -CV_PI / 2;
		heading = atan2(m02, m22);
	}
	else
	{
		bank = atan2(-m12, m11);
		attitude = asin(m10);
		heading = atan2(-m20, m00);
	}

	euler.at<double>(0) = bank;
	euler.at<double>(1) = attitude;
	euler.at<double>(2) = heading;

	return euler;
}

void fillMeasurements(cv::Mat &measurements, const vector<double> &transVec, const Mat &rotation_measured)
{
	// Set measurement to predict
	measurements.at<double>(0) = transVec[0]; // x
	measurements.at<double>(1) = transVec[1]; // y
	measurements.at<double>(2) = transVec[2]; // z
	measurements.at<double>(3) = rotation_measured.at<double>(0);      // roll
	measurements.at<double>(4) = rotation_measured.at<double>(1);      // pitch
	measurements.at<double>(5) = rotation_measured.at<double>(2);      // yaw
}


int matchPointToPoints(Point2f point, vector<Point2f> points, float maxDistThresh) {
	float closest = numeric_limits<float>::infinity();
	int index = -1;
	for (int i = 0; i < points.size(); i++) {
		float dist = euclideanDist(point, points[i]);
		if (dist < closest) {
			closest = dist;
			if (dist < maxDistThresh) {
				index = i;
			}
		}
	}
	return index;
}

void detectHLSthresholds() {
	cvtColor(image, imgHLS, COLOR_RGB2HLS); //Convert the captured frame from BGR to HLS
	inRange(imgHLS, Scalar(lowH, lowL, lowS), Scalar(highH, highL, highS), imgThresholded); //Threshold the image
	//inRange(imgHLS, Scalar(87, 230, 255), Scalar(94, 255, 255), imgThresholded); //Threshold for party_-2_l_l.mp4 green
	//inRange(imgHLS, Scalar(71, 169, 255), Scalar(98, 255, 255), imgThresholded); //Threshold for auto-darker3.mp4 blue

	//morphological operations
	//morphClose(imgThresholded);
	//morphOpen(imgThresholded);

	//find contours of filtered image using openCV findContours function
	//findContours(imgThresholded, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
	//drawContours(image, contours, -1, Scalar(255, 0, 255), 2, 8, hierarchy);

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

void detectHSVthresholds() {
	cvtColor(image, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HLS
	inRange(imgHSV, Scalar(lowH, lowS, lowV), Scalar(highH, highS, highV), imgThresholded); //Threshold the image
	imshow("Thresholded Image", imgThresholded); //show the thresholded image
}

void detectYCrCbthresholds() {
	cvtColor(image, imgHSV, COLOR_BGR2YCrCb); //Convert the captured frame from BGR to HLS
	inRange(imgHSV, Scalar(lowL, lowS, lowV), Scalar(highL, highS, highV), imgThresholded); //Threshold the image
	imshow("Thresholded Image", imgThresholded); //show the thresholded image
}

void detectLabthresholds() {
	cvtColor(image, imgHSV, COLOR_BGR2Lab); //Convert the captured frame from BGR to HLS
	inRange(imgHSV, Scalar(lowL, lowS, lowV), Scalar(highL, highS, highV), imgThresholded); //Threshold the image
	imshow("Thresholded Image", imgThresholded); //show the thresholded image
}


// User draws box around object to track. This triggers CAMShift to start tracking
static void onMouse(int event, int x, int y, int, void*)
{
	if (selectObject)
	{
		selection.x = MIN(x, origin.x);
		selection.y = MIN(y, origin.y);
		selection.width = std::abs(x - origin.x);
		selection.height = std::abs(y - origin.y);
		selection &= Rect(0, 0, image.cols, image.rows);
	}
	switch (event)
	{
	case EVENT_LBUTTONDOWN:
		origin = Point(x, y);
		selection = Rect(x, y, 0, 0);
		selectObject = true;
		break;
	case EVENT_LBUTTONUP:
		selectObject = false;
		if (selection.width > 0 && selection.height > 0)
		{
			if (waitKey(0) == '1') {
				trackObject1 = -1;   // Set up CAMShift properties in main() loop
			}
			else {
				trackObject2 = -1;
			}
		}
		break;
	}
}

void drawRotatedRect(RotatedRect rr, Mat img, Scalar color)
{
	Point2f vertices[4];
	rr.points(vertices);
	for (int i = 0; i < 4; i++)
		line(img, vertices[i], vertices[(i + 1) % 4], color, 2);
}

Rect cutRectToImgBounds(Rect r, int imgWidth, int imgHeight)
{
	Rect tempR = r;
	if (r.x + r.width > imgWidth) {
		tempR = Rect(tempR.tl(), Size(imgWidth - tempR.x, tempR.height));
	}
	if (r.y + r.height > imgHeight) {
		tempR = Rect(tempR.tl(), Size(tempR.width, imgHeight - tempR.y));
	}
	if (r.x < 0) {
		tempR.x = 0;
	}
	if (r.y < 0) {
		tempR.y = 0;
	}

	return tempR;
}

Point2d calculateMedian() {
	Point2d point = Point2d(0, 0);
	for (int i = 0; i < bufferSize; i++) {
		point += pointBuffer[i];
	}
	return point / bufferSize;
}

void fitBandContours(Rect roi, Rect roi2)
{
	bothbackproj = backprojImage + backprojImage2;

	Canny(backproj, cannyOut, 200, 200 * 2);
	findContours(cannyOut, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE); //, roi.tl());
	Canny(backproj2, cannyOut2, 200, 200 * 2);
	findContours(cannyOut2, contours2, hierarchy2, RETR_CCOMP, CHAIN_APPROX_SIMPLE); //, roi2.tl());

	vector<Point> contourPoints;
	for (vector<Point> v : contours) {
		for (Point p : v) {
			contourPoints.push_back(p);
			circle(bothbackproj, p, 0, Scalar(0,0,255),2);
		}
	}
	for (vector<Point> v : contours2) {
		for (Point p : v) {
			contourPoints.push_back(p);
			circle(bothbackproj, p, 0, Scalar(0, 0, 255), 2);
		}
	}
	if (contourPoints.size() > 4) {
		RotatedRect rr = fitEllipse(contourPoints);
		ellipse(imgDraw, rr, Scalar(0, 255, 0), 3, LINE_AA);
		/*pointBuffer[frameCounter % bufferSize] = rr.center;
		if (frameCounter > bufferSize) {
			Point2d p = calculateMedian();
			line(image, p, p, Scalar(0, 0, 255), 10);
		}*/

	}
	//drawContours(bothbackproj, contours, -1, Scalar(255, 0, 255), 1, 8, hierarchy);
	//drawContours(bothbackproj, contours2, -1, Scalar(255, 0, 255), 1, 8, hierarchy2);
	imshow("Backprojection", backprojImage);
	imshow("Backprojection2", backprojImage2);
	imshow("Both Backprojections", bothbackproj);
}

void fitBandBlob()
{
	if (!backproj.empty()) {
		Mat bpCombi = backproj + backproj2;
		imshow("test", bpCombi);
		detector->detect(bpCombi, keypoints);
		// Draw detected blobs as red circles.
		// DrawMatchesFlags::DRAW_RICH_KEYPOINTS flag ensures the size of the circle corresponds to the size of blob
		drawKeypoints(backprojImage, keypoints, backprojImage, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		imshow("Backprojection", backprojImage);
		KeyPoint::convert(keypoints, newBlobPoints);
		if (newBlobPoints.size() > 4) {
			RotatedRect rr = fitEllipse(newBlobPoints);
			ellipse(image, rr, Scalar(0, 255, 0), 3, LINE_AA);
		}
	}
}

//RotatedRect trackCamshift(int trackObject, Mat& hsv, int vmin, int vmax, int smin, Rect selection, Mat& mask, Mat& hue) {
//	if (!paused) {
//		cvtColor(image, hsv, COLOR_BGR2HSV);
//		if (trackObject)
//		{
//			inRange(hsv, Scalar(0, smin, MIN(vmin, vmax)),
//				Scalar(180, 256, MAX(vmin, vmax)), mask);
//			int ch[] = { 0, 0 };
//			hue.create(hsv.size(), hsv.depth());
//			mixChannels(&hsv, 1, &hue, 1, ch, 1);
//			if (trackObject < 0)
//			{
//				// Object has been selected by user, set up CAMShift search properties once
//				Mat roi(hue, selection), maskroi(mask, selection);
//				calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);
//				normalize(hist, hist, 0, 255, NORM_MINMAX);
//				trackWindow = selection;
//				trackObject1 = 1; // Don't set up again, unless user selects new ROI
//				histimg = Scalar::all(0);
//				int binW = histimg.cols / hsize;
//				Mat buf(1, hsize, CV_8UC3);
//				for (int i = 0; i < hsize; i++)
//					buf.at<Vec3b>(i) = Vec3b(saturate_cast<uchar>(i*180. / hsize), 255, 255);
//				cvtColor(buf, buf, COLOR_HSV2BGR);
//				for (int i = 0; i < hsize; i++)
//				{
//					int val = saturate_cast<int>(hist.at<float>(i)*histimg.rows / 255);
//					rectangle(histimg, Point(i*binW, histimg.rows),
//						Point((i + 1)*binW, histimg.rows - val),
//						Scalar(buf.at<Vec3b>(i)), -1, 8);
//				}
//			}
//		}
//	}
//}

void getBlobsByColor(vector<Point2f>& blobPositions, Mat& imgThreshC1, Mat& imgThreshC2, vector<int>& blobsC1, vector<int>& blobsC2, int radius) {
	//Mat bothThresh = Mat(imgThreshC1.rows, imgThreshC1.cols, CV_8U);
	Mat bothThresh = imgThreshC1 + imgThreshC2;
	for (int i = 0; i < blobPositions.size(); i++) {
		for (int j = 1; j <= radius; j++) {
			if (bothThresh.at<uchar>((int)blobPositions[i].y, (int)blobPositions[i].x + j) > 0) {
				if (imgThreshC1.at<uchar>((int)blobPositions[i].y, (int)blobPositions[i].x + j) > 0) {
					blobsC1.push_back(i);
					break;
				}
				else {
					blobsC2.push_back(i);
					break;
				}
			}
			else if (bothThresh.at<uchar>((int)blobPositions[i].y, (int)blobPositions[i].x - j) > 0) {
				if (imgThreshC1.at<uchar>((int)blobPositions[i].y, (int)blobPositions[i].x - j) > 0) {
					blobsC1.push_back(i);
					break;
				}
				else {
					blobsC2.push_back(i);
					break;
				}
			}
			else if (bothThresh.at<uchar>((int)blobPositions[i].y + j, (int)blobPositions[i].x) > 0) {
				if (imgThreshC1.at<uchar>((int)blobPositions[i].y + j, (int)blobPositions[i].x) > 0) {
					blobsC1.push_back(i);
					break;
				}
				else {
					blobsC2.push_back(i);
					break;
				}
			}
			else if (bothThresh.at<uchar>((int)blobPositions[i].y - j, (int)blobPositions[i].x) > 0) {
				if (imgThreshC1.at<uchar>((int)blobPositions[i].y - j, (int)blobPositions[i].x) > 0) {
					blobsC1.push_back(i);
					break;
				}
				else {
					blobsC2.push_back(i);
					break;
				}
			}
		}
	}
}

void trackCamshift() {
	if (!paused)
	{
		cvtColor(image, hls, COLOR_BGR2HLS);
		if (trackObject1)
		{
			inRange(hls, Scalar(0, lmin1, 250),
				Scalar(180, lmax1, 255), mask);
			int ch[] = { 0, 0 };
			hue.create(hls.size(), hls.depth());
			mixChannels(&hls, 1, &hue, 1, ch, 1);
			if (trackObject1 < 0)
			{
				// Object has been selected by user, set up CAMShift search properties once
				Mat roi(hue, selection), maskroi(mask, selection);
				calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);
				normalize(hist, hist, 0, 255, NORM_MINMAX);
				trackWindow = selection;
				trackObject1 = 1; // Don't set up again, unless user selects new ROI
				histimg = Scalar::all(0);
				int binW = histimg.cols / hsize;
				Mat buf(1, hsize, CV_8UC3);
				for (int i = 0; i < hsize; i++)
					buf.at<Vec3b>(i) = Vec3b(saturate_cast<uchar>(i*180. / hsize), 255, 255);
				cvtColor(buf, buf, COLOR_HSV2BGR);
				for (int i = 0; i < hsize; i++)
				{
					int val = saturate_cast<int>(hist.at<float>(i)*histimg.rows / 255);
					rectangle(histimg, Point(i*binW, histimg.rows),
						Point((i + 1)*binW, histimg.rows - val),
						Scalar(buf.at<Vec3b>(i)), -1, 8);
				}
			}
			// Perform CAMShift
			calcBackProject(&hue, 1, 0, hist, backproj, &phranges);
			backproj &= mask;
			trackBox = CamShift(backproj, trackWindow,
				TermCriteria(TermCriteria::EPS | TermCriteria::COUNT, 10, 1));
			if (trackWindow.area() <= 1)
			{
				int cols = backproj.cols, rows = backproj.rows, r = (MIN(cols, rows) + 5) / 6;
				trackWindow = Rect(trackWindow.x - r, trackWindow.y - r,
					trackWindow.x + r, trackWindow.y + r) &
					Rect(0, 0, cols, rows);
			}
			if (backprojMode) {
				//backprojection image
				image.copyTo(backprojImage);
				cvtColor(backproj, backprojImage, COLOR_GRAY2BGR);

				if (trackBox.size.height > 0 && trackBox.size.width > 0) {
					//fitBand(cutRectToImgBounds(trackBox.boundingRect(), imgResizeX, imgResizeY));
					drawRotatedRect(trackBox, imgDraw, Scalar(0,255,255));
					//rectangle(backprojImage, trackBox.boundingRect(), Scalar(0, 0, 255));
					//ellipse(imgDraw, trackBox, Scalar(255, 0, 0), 2, LINE_AA);
				}
				imshow("Backprojection", backprojImage);
			}
		}
		if (trackObject2)
		{
			inRange(hls, Scalar(0, lmin2, 250),
				Scalar(180, lmax2, 255), mask2);
			int ch[] = { 0, 0 };
			hue2.create(hls.size(), hls.depth());
			mixChannels(&hls, 1, &hue2, 1, ch, 1);
			if (trackObject2 < 0)
			{
				// Object has been selected by user, set up CAMShift search properties once
				Mat roi(hue2, selection), maskroi(mask2, selection);
				calcHist(&roi, 1, 0, maskroi, hist2, 1, &hsize, &phranges);
				normalize(hist2, hist2, 0, 255, NORM_MINMAX);
				trackWindow2 = selection;
				trackObject2 = 1; // Don't set up again, unless user selects new ROI
				histimg2 = Scalar::all(0);
				int binW = histimg2.cols / hsize;
				Mat buf(1, hsize, CV_8UC3);
				for (int i = 0; i < hsize; i++)
					buf.at<Vec3b>(i) = Vec3b(saturate_cast<uchar>(i*180. / hsize), 255, 255);
				cvtColor(buf, buf, COLOR_HSV2BGR);
				for (int i = 0; i < hsize; i++)
				{
					int val = saturate_cast<int>(hist2.at<float>(i)*histimg2.rows / 255);
					rectangle(histimg2, Point(i*binW, histimg2.rows),
						Point((i + 1)*binW, histimg2.rows - val),
						Scalar(buf.at<Vec3b>(i)), -1, 8);
				}
			}
			// Perform CAMShift
			calcBackProject(&hue2, 1, 0, hist2, backproj2, &phranges);
			backproj2 &= mask2;
			trackBox2 = CamShift(backproj2, trackWindow2,
				TermCriteria(TermCriteria::EPS | TermCriteria::COUNT, 10, 1));
			if (trackWindow2.area() <= 1)
			{
				int cols = backproj2.cols, rows = backproj2.rows, r = (MIN(cols, rows) + 5) / 6;
				trackWindow2 = Rect(trackWindow2.x - r, trackWindow2.y - r,
					trackWindow2.x + r, trackWindow2.y + r) &
					Rect(0, 0, cols, rows);
			}
			if (backprojMode) {
				//backprojection image
				image.copyTo(backprojImage2);
				cvtColor(backproj2, backprojImage2, COLOR_GRAY2BGR);

				if (trackBox2.size.height > 0 && trackBox2.size.width > 0) {
					//fitBand(cutRectToImgBounds(trackBox2.boundingRect(), imgResizeX, imgResizeY));
					//drawRotatedRect(trackBox2, imgDraw, Scalar(0,255,255));
					//rectangle(backprojImage2, trackBox2.boundingRect(), Scalar(0, 0, 255));
					ellipse(imgDraw, trackBox2, Scalar(255, 0, 0), 2, LINE_AA);
				}
				imshow("Backprojection2", backprojImage2);
			}
		}
		if (trackObject1 && trackObject2) {
			//fitBand(cutRectToImgBounds(trackBox.boundingRect(), imgResizeX, imgResizeY), cutRectToImgBounds(trackBox2.boundingRect(), imgResizeX, imgResizeY));
		}
	}
	else if (trackObject1 < 0 || trackObject2 < 0)
		paused = false;
	if (selectObject && selection.width > 0 && selection.height > 0)
	{
		Mat roi(image, selection);
		bitwise_not(roi, roi);
	}
	imshow("Histogram", histimg);
	imshow("Histogram2", histimg2);
}

void LEDdetect()
{
	//compute mask based on camshift track rotated rect
	Mat blobMask = Mat::zeros(imgResizeY, imgResizeX, CV_8UC3);
	Point2f vertices2f[4];
	Point vertices[4];
	trackBox.points(vertices2f);
	for (int i = 0; i < 4; i++){
		vertices[i] = vertices2f[i];
	}
	fillConvexPoly(blobMask, vertices, 4, Scalar(255,255,255));
	Mat maskedImg;
	image.copyTo(maskedImg, blobMask);
	imshow("maskedImg", maskedImg);

	Mat maskedGrey, maskedBinary;
	cvtColor(maskedImg, maskedGrey, cv::COLOR_BGR2GRAY);
	threshold(maskedGrey, maskedBinary, thresh, 255, THRESH_BINARY);
	erode(maskedBinary, maskedBinary, getStructuringElement(MORPH_RECT, Size(3, 3)));
	dilate(maskedBinary, maskedBinary, getStructuringElement(MORPH_RECT, Size(3, 3)));
	imshow("binary", maskedBinary);

	////threshold masked image
	//cvtColor(maskedImg, imgHLS, COLOR_BGR2HLS); //Convert the captured frame from BGR to HLS
	//inRange(imgHLS, Scalar(lowH, lowL, lowS), Scalar(highH, highL, highS), imgThresholded); //Threshold the image
	//imshow("Thresholded image", imgThresholded);

	////canny edge
	//Canny(imgThresholded, cannyOut, 200, 200 * 2);
	//findContours(cannyOut, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

	////fit ellipse with detected edge points
	//vector<Point> contourPoints;
	//for (vector<Point> v : contours) {
	//	for (Point p : v) {
	//		contourPoints.push_back(p);
	//	}
	//}
	//if (contourPoints.size() > 4) {
	//	RotatedRect rr = fitEllipse(contourPoints);
	//	ellipse(image, rr, Scalar(0, 255, 0), 3, LINE_AA);
	//}
}

void greyLEDdetect(){
	cvtColor(image, imgGrey, cv::COLOR_BGR2GRAY);
	inRange(imgGrey, minGrey, maxGrey, imgBinary); //Thresholde image
	//morphOpen(imgBinary);
	//morphClose(imgBinary);
	detector->detect(imgBinary, keypoints);
	drawKeypoints(imgDraw, keypoints, imgDraw, Scalar(255, 0, 0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	KeyPoint::convert(keypoints, newBlobPoints);
	
	imshow("Draw", imgDraw);
	waitKey(1);

	//todo: remove
	/*RotatedRect rr = fitEllipse(newBlobPoints);
	ellipse(imgDraw, rr, Scalar(0, 255, 0), 3, LINE_AA);*/
}

void PnPapproxInit() {
	//bool solvePnP(InputArray objectPoints, InputArray imagePoints, InputArray cameraMatrix, InputArray distCoeffs, OutputArray rvec, OutputArray tvec, bool useExtrinsicGuess = false, int flags = SOLVEPNP_ITERATIVE);
	
	//approx internal camera parameters
	double focalLength = imgSizeX;
	Point2d center = Point2d(imgSizeX / 2, imgSizeY / 2);
	cameraMatrix = (Mat_<double>(3, 3) << focalLength, 0, center.x, 0, focalLength, center.y, 0, 0, 1);
	distortCoeffs = Mat::zeros(4, 1, DataType<double>::type); // Assuming no lens distortion
	cout << "Camera Matrix " << endl << cameraMatrix << endl;
}

void PnPinit() {
	//internal camera parameters
	double focalLengthX = 1056.547145;
	double focalLengthY = 1056.333864;
	Point2d center = Point2d(638.890253, imgSizeY - 1 - 499.522829);
	cameraMatrix = (Mat_<double>(3, 3) << focalLengthX, 0, center.x, 0, focalLengthY, center.y, 0, 0, 1);
	distortCoeffs = Mat::zeros(4, 1, DataType<double>::type); // image already undistorted
	cout << "Camera Matrix " << endl << cameraMatrix << endl;
}

void reDetectMatch() {
	//sort blobs by x coordinate descending
	sort(newBlobPoints.begin(), newBlobPoints.end(), point_sorter());

	cvtColor(image, imgHLS, COLOR_RGB2HLS); //Convert the captured frame from BGR to HLS with hue switch because of red edge case
	//inRange(imgHLS, Scalar(0, 0, 0), Scalar(180, 255, 255), imgThreshC1);
	inRange(imgHLS, Scalar(15, 115, 255), Scalar(35, 255, 255), imgThreshC1); //Threshold for blue in darkblue 1.003
	inRange(imgHLS, Scalar(85, 115, 255), Scalar(180, 255, 255), imgThreshC2); //Threshold for red in darkblue 1.003
	vector<int> blobsC1, blobsC2;
	getBlobsByColor(newBlobPoints, imgThreshC1, imgThreshC2, blobsC1, blobsC2, 5);

	if (!blobsC1.empty()) {
		for (int i = 0; i < min((int)blobsC1.size(), 4); i++) { //left color from left to 3
			LEDtoNewBlobMatching[4 - blobsC1.size() + i] = blobsC1[i];
		}
	}
	if (!blobsC2.empty()) {
		for (int i = 0; i < min((int)blobsC2.size(), 6); i++) { //right color from 4 to right
			LEDtoNewBlobMatching[4 + i] = blobsC2[i];
		}
	}
}

void PnPtest(int frame) {
	vector<double> rvecTest, tvecTest;
	vector<Point2d> outputTestPoints;
	vector<Point3d> relevantLEDs; //those 3DLEDs that have a known 2D blob position
	vector<Point2d> correspondingBlobs;//the corresponding blob positions

	if (trackingLost) {
		reDetectMatch();
	}
	for (int i = 0; i < LEDtoNewBlobMatching.size(); i++) {
		if (LEDtoNewBlobMatching[i] >= 0) { //push all 3d led points that have a match
			relevantLEDs.push_back(modelPoints3D[i]);
			correspondingBlobs.push_back(newBlobPoints[LEDtoNewBlobMatching[i]]);
		}
	}

	if (relevantLEDs.size() >= 4) {
		solvePnP(relevantLEDs, correspondingBlobs, cameraMatrix, distortCoeffs, rvecTest, tvecTest, false, SOLVEPNP_ITERATIVE);

		//resulting rvec and tvec transform from the model coordinate system to the camera, so invert? turns out no
		Mat rotmatTest;
		Mat eulermatTest(3, 1, CV_64F);
		Rodrigues(rvecTest, rotmatTest);

		eulermatTest = rot2euler(rotmatTest);

		////https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToEuler/index.htm
		//double roll = atan2(-rotmatTest.at<double>(2, 0), rotmatTest.at<double>(0, 0));
		//double pitch = atan2(-rotmatTest.at<double>(1, 2), rotmatTest.at<double>(1, 1));
		//double yaw = asin(rotmatTest.at<double>(1, 0));



		/*cout << "rot: " << rvecTest[0] << " " << rvecTest[1] << " " << rvecTest[2] << endl
			<< "trans: " << tvecTest[0] << " " << tvecTest[1] << " " << tvecTest[2] << endl;*/

			//cout << "rotEul: " << "heading=" << heading << " bank=" << bank << " attitude=" << attitude << endl;

			//rotmatTest = rotmatTest.inv(DECOMP_SVD);
			//Rodrigues(rotmatTest, rvecTest);

			//cout << "rotInv: " << rvecTest[0] << " " << rvecTest[1] << " " << rvecTest[2] << endl;

		projectPoints(modelPoints3D, rvecTest, tvecTest, cameraMatrix, distortCoeffs, outputTestPoints);

		int reprojectioncounter = 0;
		for (int i = 0; i < outputTestPoints.size(); i++) {
			circle(imgDraw, outputTestPoints[i], 1, Scalar(0, 255, 0), 2); // green: reprojected LEDs
			putText(imgDraw, to_string(i), outputTestPoints[i], FONT_HERSHEY_SCRIPT_SIMPLEX, 1, Scalar(0, 255, 0));
			//for all reprojected LEDs check if a blob is detected there
			int index = matchPointToPoints(outputTestPoints[i], newBlobPoints, 5);
			LEDtoOldBlobMatching[i] = index;
			if (index >= 0) {
				reprojectioncounter++;
			}
		}
		imshow("Draw", imgDraw);
		waitKey(1);
		cout << "";
		if (reprojectioncounter < 4) {
			//bad pose. reinitiate detection phase
			//relevantLEDs = get 4 random LEDs from modelPoints3D
			//solveP3P with all possible matches in newBlobPoints
			//reproject each one and count LED to blob matches
			trackingLost = true;
		}
		else {
			trackingLost = false;
			fillMeasurements(measurements, tvecTest, eulermatTest);
		}
	}
	else {
		trackingLost = true;
	}
}

vector<double> readAllTS(string pathToFile) {
	double ts;
	string frameName;
	vector<double> timeStamps;
	ifstream tsFile(pathToFile);
	while (tsFile >> ts >> frameName) {
		timeStamps.push_back(ts);
	}
	tsFile.close();
	return timeStamps;
}

vector<Point3d> fillModelPoints() {
	Point3d LED10 = Point3d(2.67097532115543046e-01, -3.47028848784865090e-02, 1.71293486959551572e-01);
	Point3d LED9 = Point3d(2.47425499670097748e-01, -3.36907769164135340e-02, 1.99522292397339263e-01);
	Point3d LED8 = Point3d(2.21684687650311907e-01, -3.22243632083136639e-02, 2.21157774715360605e-01);
	Point3d LED7 = Point3d(1.91487232768038446e-01, -3.06480855540219206e-02, 2.37121236105386574e-01);
	Point3d LED6 = Point3d(1.58621091131353253e-01, -2.94846784427151495e-02, 2.44794772578041803e-01);
	Point3d LED5 = Point3d(1.25148501293946501e-01, -2.64061662524335758e-02, 2.44441906540085907e-01);
	Point3d LED4 = Point3d(9.22603348005930773e-02, -2.45705276530471320e-02, 2.35824422950973500e-01);
	Point3d LED3 = Point3d(6.15248984566603985e-02, -2.44769579196429457e-02, 2.19972846145236156e-01);
	Point3d LED2 = Point3d(3.71471618613731930e-02, -2.49878525424118911e-02, 1.97927458344047347e-01);
	Point3d LED1 = Point3d(1.74547127144131856e-02, -2.65532292111988720e-02, 1.70035917308056728e-01);
	return vector<Point3d>{LED1, LED2, LED3, LED4, LED5, LED6, LED7, LED8, LED9, LED10};
}

vector<vector<double>> readAllPoses(string pathToFile) {
	double ts, rx, ry, rz, rw, tx, ty, tz, one;
	vector<vector<double>> poses;
	string line;
	ifstream poseFile(pathToFile);
	getline(poseFile,line);
	getline(poseFile, line);//throw away the first 2 strangely formated lines
	while (poseFile >> ts >> rx >> ry >> rz >> rw >> tx >> ty >> tz >> one) {
		vector<double> pose;
		pose.push_back(ts);
		pose.push_back(rx);
		pose.push_back(ry);
		pose.push_back(rz);
		pose.push_back(rw);
		pose.push_back(tx);
		pose.push_back(ty);
		pose.push_back(tz);
		poses.push_back(pose);
	}
	poseFile.close();
	return poses;
}

vector<vector<double>> interpolateAllTruePosesAtFrameTS() {
	double frameTS;
	vector<vector<double>> posesToTimeStamps;
	vector<double> poseAtTS;
	int bookMark = 0;
	for (int i = 0; i < allFrameTimeStamps.size(); i++) {
		frameTS = allFrameTimeStamps[i];
		
		for (int ii = bookMark; ii < allTruePoses.size(); ii++) { //loop through all prerecorded poses until timestamp is bigger than current frame i
			double poseTS = allTruePoses[ii][0];
			if (poseTS > frameTS) {
				vector<double> poseLeft = allTruePoses[max(ii - 1, 0)];
				vector<double> poseRight = allTruePoses[ii];

				//calc pose at frameTS
				vector<double>poseAtTS;
				poseAtTS.push_back(frameTS);
				double relation = (frameTS - poseLeft[0]) / (poseTS - poseLeft[0]);

				//slerp quaternion
				quat qa = quat(poseLeft[1], poseLeft[2], poseLeft[3], poseLeft[4]);
				quat qb = quat(poseRight[1], poseRight[2], poseRight[3], poseRight[4]);
				quat qm = slerp(qa, qb, relation); //<--

				//lerp position
				Point3d pa = Point3d(poseLeft[5], poseLeft[6], poseLeft[7]);
				Point3d pb = Point3d(poseRight[5], poseRight[6], poseRight[7]);
				Point3d pm = relation * pb + (1 - relation) * pa; //<--

				poseAtTS.push_back(qm.x);
				poseAtTS.push_back(qm.y);
				poseAtTS.push_back(qm.z);
				poseAtTS.push_back(qm.w);
				poseAtTS.push_back(pm.x);
				poseAtTS.push_back(pm.y);
				poseAtTS.push_back(pm.z);
				posesToTimeStamps.push_back(poseAtTS); // fill return vector
				bookMark = ii;
				break;
			}
		}
	}
	return posesToTimeStamps;
}

void drawTruePose(int frame) {
	vector<Point3d> truePoints;
	vector<Point2d> trueImgPoints;
	Point3d p = Point3d(0, 0, 0);
	truePoints = modelPoints3D;
	truePoints.push_back(p);

	vector<double> rotvec, transvec;
	double qx = allSLERPedPoses[frame][1];
	double qy = allSLERPedPoses[frame][2];
	double qz = allSLERPedPoses[frame][3];
	double qw = allSLERPedPoses[frame][4];

	//axis angle
	//quaternion to axisangle, see https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToAngle/index.htm
	/*double angle = 2 * acos(qw);
	double xAxis = qx / sqrt(1 - qw * qw);
	double yAxis = qy / sqrt(1 - qw * qw);
	double zAxis = qz / sqrt(1 - qw * qw);
	
	rotvec.push_back(xAxis);
	rotvec.push_back(yAxis);
	rotvec.push_back(zAxis);
	rotvec.push_back(angle);*/

	//rot matrix
	//quaternion to rotation matrix, see https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm
	double sqw = qw*qw;
    double sqx = qx*qx;
    double sqy = qy*qy;
    double sqz = qz*qz;

    // invs (inverse square length) is only required if quaternion is not already normalised
	double invs = 1 / (sqx + sqy + sqz + sqw);
    double m00 = ( sqx - sqy - sqz + sqw)*invs ; // since sqw + sqx + sqy + sqz =1/invs*invs
	double m11 = (-sqx + sqy - sqz + sqw)*invs ;
	double m22 = (-sqx - sqy + sqz + sqw)*invs ;
    
    double tmp1 = qx*qy;
    double tmp2 = qz*qw;
	double m10 = 2.0 * (tmp1 + tmp2)*invs ;
	double m01 = 2.0 * (tmp1 - tmp2)*invs ;
    
    tmp1 = qx*qz;
    tmp2 = qy*qw;
	double m20 = 2.0 * (tmp1 - tmp2)*invs ;
	double m02 = 2.0 * (tmp1 + tmp2)*invs ;
    tmp1 = qy*qz;
    tmp2 = qx*qw;
	double m21 = 2.0 * (tmp1 + tmp2)*invs ;
	double m12 = 2.0 * (tmp1 - tmp2)*invs ;
	
	Mat rotmat = (Mat_<double>(3, 3) << m00, m01, m02, m10, m11, m12, m20, m21, m22);
	//Mat rotmat = (Mat_<double>(3, 3) << m00, m10, m20, m01, m11, m21, m02, m12, m22);

	float rotAngle = CV_PI; //180 deg
	Mat rotCorrection = (Mat_<double>(3, 3) << 1, 0, 0, 0, cos(rotAngle), -sin(rotAngle), 0, sin(rotAngle), cos(rotAngle));

	rotmat = rotCorrection * rotmat;

	Rodrigues(rotmat, rotvec);

	transvec.push_back(allSLERPedPoses[frame][5]);//x
	transvec.push_back(-allSLERPedPoses[frame][6]);//y
	transvec.push_back(-allSLERPedPoses[frame][7]);//z

	projectPoints(truePoints, rotvec, transvec, cameraMatrix, distortCoeffs, trueImgPoints);
	/*cout << "true rot: " << rotvec[0] << " " << rotvec[1] << " " << rotvec[2] << endl
		<< "true trans: " << transvec[0] << " " << transvec[1] << " " << transvec[2] << endl;*/

	double heading = atan2(-rotmat.at<double>(2, 0), rotmat.at<double>(0, 0));
	double bank = atan2(-rotmat.at<double>(1, 2), rotmat.at<double>(1, 1));
	double attitude = asin(rotmat.at<double>(1, 0));

	//cout << "true rotEul: " << "heading=" << heading << " bank=" << bank << " attitude=" << attitude << endl;

	for (Point2d p : trueImgPoints) {
		circle(image, p, 1, Scalar(255, 0, 255), 2);
	}
}

void trackBlobsOFlow(float l1thresh) { //used to match new blobs to old blobs
	if (!oldBlobPoints.empty()) { //if there is information from a previous frame

	//dense optical flow:
		//Mat flow(imgGreyOld.size(), CV_32FC2);
		//calcOpticalFlowFarneback(imgGreyOld, imgGrey, flow, 0.5, 3, 15, 3, 5, 1.1, 0);
		//// visualization
		//Mat flow_parts[2];
		//split(flow, flow_parts);
		//Mat magnitude, angle, magn_norm;
		//cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
		//normalize(magnitude, magn_norm, 0.0f, 1.0f, NORM_MINMAX);
		//angle *= ((1.f / 360.f) * (180.f / 255.f));
		////build hsv image
		//Mat _hsv[3], hsv, hsv8;
		//_hsv[0] = angle;
		//_hsv[1] = Mat::ones(angle.size(), CV_32F);
		//_hsv[2] = magn_norm;
		//merge(_hsv, 3, hsv);
		//hsv.convertTo(hsv8, CV_8U, 255.0);
		//cvtColor(hsv8, bgr, COLOR_HSV2BGR);
		//imshow("frame2", bgr);


	//sparse optical flow:
		calcOpticalFlowPyrLK(imgGreyOld, imgGrey, oldBlobPoints, predictedBlobPoints, status, err, Size(10, 10), 4, term);
		ostringstream oss;
		if (!err.empty()) {
			copy(err.begin(), err.end() - 1, ostream_iterator<float>(oss, ","));
			oss << err.back();
		}
		cout << "errors:" << oss.str() << endl;

		oldToNewBlobMatching = vector<int>(oldBlobPoints.size());
		for (int i = 0; i < oldBlobPoints.size(); i++) {
			//circle(image, oldBlobPoints[i], 2, Scalar(255, 0, 0), 2);// blue: old blobs
			putText(image, to_string(i), oldBlobPoints[i], FONT_HERSHEY_SCRIPT_SIMPLEX, 1, Scalar(255, 0, 0),1,8,true);
		}
		for (int i = 0; i < predictedBlobPoints.size(); i++) {
			circle(image, predictedBlobPoints[i], 4, Scalar(0, 255, 0), 2);//predicted blobs
			line(image, oldBlobPoints[i], predictedBlobPoints[i], Scalar(0, 255, 0), 2);
			if (err[i] > l1thresh) {
				oldToNewBlobMatching[i] = -1;
			}
			else {
				oldToNewBlobMatching[i] = matchPointToPoints(predictedBlobPoints[i], newBlobPoints, 10);
			}
			putText(image, to_string(oldToNewBlobMatching[i]), predictedBlobPoints[i], FONT_HERSHEY_SCRIPT_SIMPLEX, 1, Scalar(0, 255, 255));
			//putText(image, to_string((int)err[i]), predictedBlobPoints[i], FONT_HERSHEY_SCRIPT_SIMPLEX, 1, Scalar(0, 255, 255));
		}
		imshow("Original", image);
		waitKey(1);
		for (int i = 0; i < LEDtoNewBlobMatching.size(); i++) {
			int index = LEDtoOldBlobMatching[i];
			if (index >= 0) {
				LEDtoNewBlobMatching[i] = oldToNewBlobMatching[LEDtoOldBlobMatching[i]];
				if (LEDtoNewBlobMatching[i] >= 0) {
					//putText(image, to_string(i), newBlobPoints[LEDtoNewBlobMatching[i]], FONT_HERSHEY_SCRIPT_SIMPLEX, 1, Scalar(0, 255, 255));
				}
			}
			else {
				LEDtoNewBlobMatching[i] = -1;
			}
		}

		imshow("Original", image);
		waitKey(1);
	}
}

void initKalmanFilter(cv::KalmanFilter &KF, int nStates, int nMeasurements, int nInputs, double dt)
{
	//https://docs.opencv.org/master/dc/d2c/tutorial_real_time_pose.html
	KF.init(nStates, nMeasurements, nInputs, CV_64F);                 // init Kalman Filter
	cv::setIdentity(KF.processNoiseCov, cv::Scalar::all(1e-5));       // set process noise
	cv::setIdentity(KF.measurementNoiseCov, cv::Scalar::all(1e-4));   // set measurement noise
	cv::setIdentity(KF.errorCovPost, cv::Scalar::all(1));             // error covariance
				   /* DYNAMIC MODEL */
	//  [1 0 0 dt  0  0 dt2   0   0 0 0 0  0  0  0   0   0   0]
	//  [0 1 0  0 dt  0   0 dt2   0 0 0 0  0  0  0   0   0   0]
	//  [0 0 1  0  0 dt   0   0 dt2 0 0 0  0  0  0   0   0   0]
	//  [0 0 0  1  0  0  dt   0   0 0 0 0  0  0  0   0   0   0]
	//  [0 0 0  0  1  0   0  dt   0 0 0 0  0  0  0   0   0   0]
	//  [0 0 0  0  0  1   0   0  dt 0 0 0  0  0  0   0   0   0]
	//  [0 0 0  0  0  0   1   0   0 0 0 0  0  0  0   0   0   0]
	//  [0 0 0  0  0  0   0   1   0 0 0 0  0  0  0   0   0   0]
	//  [0 0 0  0  0  0   0   0   1 0 0 0  0  0  0   0   0   0]
	//  [0 0 0  0  0  0   0   0   0 1 0 0 dt  0  0 dt2   0   0]
	//  [0 0 0  0  0  0   0   0   0 0 1 0  0 dt  0   0 dt2   0]
	//  [0 0 0  0  0  0   0   0   0 0 0 1  0  0 dt   0   0 dt2]
	//  [0 0 0  0  0  0   0   0   0 0 0 0  1  0  0  dt   0   0]
	//  [0 0 0  0  0  0   0   0   0 0 0 0  0  1  0   0  dt   0]
	//  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  1   0   0  dt]
	//  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   1   0   0]
	//  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   0   1   0]
	//  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   0   0   1]
	// position
	KF.transitionMatrix.at<double>(0, 3) = dt;
	KF.transitionMatrix.at<double>(1, 4) = dt;
	KF.transitionMatrix.at<double>(2, 5) = dt;
	KF.transitionMatrix.at<double>(3, 6) = dt;
	KF.transitionMatrix.at<double>(4, 7) = dt;
	KF.transitionMatrix.at<double>(5, 8) = dt;
	KF.transitionMatrix.at<double>(0, 6) = 0.5*pow(dt, 2);
	KF.transitionMatrix.at<double>(1, 7) = 0.5*pow(dt, 2);
	KF.transitionMatrix.at<double>(2, 8) = 0.5*pow(dt, 2);
	// orientation
	KF.transitionMatrix.at<double>(9, 12) = dt;
	KF.transitionMatrix.at<double>(10, 13) = dt;
	KF.transitionMatrix.at<double>(11, 14) = dt;
	KF.transitionMatrix.at<double>(12, 15) = dt;
	KF.transitionMatrix.at<double>(13, 16) = dt;
	KF.transitionMatrix.at<double>(14, 17) = dt;
	KF.transitionMatrix.at<double>(9, 15) = 0.5*pow(dt, 2);
	KF.transitionMatrix.at<double>(10, 16) = 0.5*pow(dt, 2);
	KF.transitionMatrix.at<double>(11, 17) = 0.5*pow(dt, 2);
	/* MEASUREMENT MODEL */
//  [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
//  [0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
//  [0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
//  [0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0]
//  [0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0]
//  [0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0]
	KF.measurementMatrix.at<double>(0, 0) = 1;  // x
	KF.measurementMatrix.at<double>(1, 1) = 1;  // y
	KF.measurementMatrix.at<double>(2, 2) = 1;  // z
	KF.measurementMatrix.at<double>(3, 9) = 1;  // roll
	KF.measurementMatrix.at<double>(4, 10) = 1; // pitch
	KF.measurementMatrix.at<double>(5, 11) = 1; // yaw
}

void updateKalmanFilter(cv::KalmanFilter &KF, cv::Mat &measurement, cv::Mat &translation_estimated, cv::Mat &rotation_estimated)
{
	// First predict, to update the internal statePre variable
	cv::Mat prediction = KF.predict();
	// The "correct" phase that is going to use the predicted value and our measurement
	cv::Mat estimated = KF.correct(measurement);
	// Estimated translation
	translation_estimated.at<double>(0) = estimated.at<double>(0);
	translation_estimated.at<double>(1) = estimated.at<double>(1);
	translation_estimated.at<double>(2) = estimated.at<double>(2);
	// Estimated euler angles
	cv::Mat eulers_estimated(3, 1, CV_64F);
	eulers_estimated.at<double>(0) = estimated.at<double>(9);
	eulers_estimated.at<double>(1) = estimated.at<double>(10);
	eulers_estimated.at<double>(2) = estimated.at<double>(11);
	// Convert estimated quaternion to rotation matrix
	rotation_estimated = euler2rot(eulers_estimated);

}

void drawKalmanPoints() {
	vector<Point2d> outputTestPoints;
	vector<double> rotvecKF;
	Rodrigues(rotation_estimated, rotvecKF);
	projectPoints(modelPoints3D, rotvecKF, translation_estimated, cameraMatrix, distortCoeffs, outputTestPoints);
	for (Point2d p : outputTestPoints)
	{
		circle(imgDraw, p, 1, Scalar(255, 0, 255), 2); // kalman predicted LEDs
	}
}


int main()
{
	//VideoCapture cap(0); //capture the video from web cam
	//VideoCapture cap("party_-2_l_l.mp4"); //video file
	//VideoCapture cap("auto-darker3.mp4");
	//VideoCapture cap("night-normal.mp4");
	//VideoCapture cap("VID_20190920_155534.mp4");
	VideoCapture cap(pathToImgSequence + "img00000.png");

	if (!cap.isOpened())  // if not success, exit program
	{
		cout << "Cannot open the web cam / file" << endl;
		return -1;
	}

	imgSizeX = cap.get(CAP_PROP_FRAME_WIDTH);
	imgSizeY = cap.get(CAP_PROP_FRAME_HEIGHT);

	imgResizeX = imgSizeX * resScale; //640;
	imgResizeY = imgSizeY * resScale;//360; || 512

	cap.set(CAP_PROP_POS_FRAMES, startFrame); //offset start frame

	/*cap.set(CAP_PROP_FRAME_WIDTH, imgSizeX);
	cap.set(CAP_PROP_FRAME_HEIGHT, imgSizeY);*/

	//fill ground truth from files
	allFrameTimeStamps = readAllTS(pathToImgSequence + "ts.txt");
	allTruePoses = readAllPoses(pathToImgSequence + "Camera2Targret.txt");
	allSLERPedPoses = interpolateAllTruePosesAtFrameTS();
	modelPoints3D = fillModelPoints();
	LEDtoNewBlobMatching = vector<int>(modelPoints3D.size());
	LEDtoOldBlobMatching = LEDtoNewBlobMatching;

	createTrackbars();

	//camshift temp
	namedWindow("Original", 1);
	setMouseCallback("Original", onMouse, 0);
	namedWindow("Backprojection", 1);
	namedWindow("Backprojection2", 1);
	createTrackbar("Lmin1", "Backprojection", &lmin1, 255, 0);
	createTrackbar("Lmax1", "Backprojection", &lmax1, 255, 0);
	/*createTrackbar("Lmin2", "Backprojection2", &lmin2, 255, 0);
	createTrackbar("Lmax2", "Backprojection2", &lmax2, 255, 0);*/


	//blob
	// Setup SimpleBlobDetector parameters.
	SimpleBlobDetector::Params params;
	
	// Change thresholds
	params.minThreshold = minGrey; //240 for handy footage
	params.maxThreshold = maxGrey;

	// Filter by Area.
	params.filterByArea = false; //true for handy
	params.minArea = 3;

	// Filter by Circularity
	params.filterByCircularity = false;
	params.minCircularity = 0.1;

	// Filter by Convexity
	params.filterByConvexity = false;
	params.minConvexity = 0.87;

	// Filter by Inertia
	params.filterByInertia = false; //true for handy
	params.minInertiaRatio = 0.01;

	// Blob merge distance
	params.minDistBetweenBlobs = 0; //3 for handy

	// Blob intensity
	params.filterByColor = false;
	params.blobColor = 255;

	// Set up detector with params
	detector = SimpleBlobDetector::create(params);
	// SimpleBlobDetector::create creates a smart pointer. 
	// So you need to use arrow ( ->) instead of dot ( . )

	//PnPapproxInit();
	PnPinit();
	
	initKalmanFilter(KF, nStates, nMeasurements, nInputs, dt);    // init function
	measurements = Mat(nMeasurements, 1, CV_64FC1); measurements.setTo(Scalar(0));

	paused = false;
	recordMode = false;
	bool frameRead = cap.read(frame); // read a new frame from video (frame if resize, image if not)

	for(;;)
	{
		if (!paused) {
			t = (double)getTickCount(); //start timer
			bool frameRead = cap.read(frame); // read a new frame from video (frame if resize, image if not)
			if (!frameRead) //if fail, break loop
			{
				cout << "Cannot read a frame from video stream" << endl;
				break;
			}
			frameCounter++;
			if (frameCounter == cap.get(CAP_PROP_FRAME_COUNT)-1) //if end of video file reached, start from first frame
			{
				frameCounter = startFrame;
				cap.set(CAP_PROP_POS_FRAMES, startFrame);
				cout << "Video loop" << endl;
			}
		}
		resize(frame, image, Size(imgResizeX, imgResizeY), 0, 0, INTER_CUBIC);
		image.copyTo(imgDraw);
		imshow("Original", image);
		waitKey(1);

		//detectHLSthresholds(); //show regions of specified HLS values
		//detectHSVthresholds();
		//detectYCrCbthresholds();
		//detectLabthresholds();
		//trackCamshift();
		//LEDdetect();
		//fitBandBlob();
		if (!backproj.empty() && !backproj2.empty()) {
			//fitBandContours(cutRectToImgBounds(trackBox.boundingRect(), imgResizeX, imgResizeY), cutRectToImgBounds(trackBox2.boundingRect(), imgResizeX, imgResizeY));
		}

		greyLEDdetect();
		if (!trackingLost) {
			trackBlobsOFlow(20);
		}
		PnPtest(frameCounter);

		if (!trackingLost) {
			updateKalmanFilter(KF, measurements, translation_estimated, rotation_estimated);
			drawKalmanPoints();
		}

		////solvePnP(modelPoints3D, imagePoints2D, cameraMatrix, distortCoeffs, rotVec, transVec, false, iterationCount, reprojectionError, minInliers, inliersA, SOLVEPNP_IPPE);
		
		//drawTruePose(frameCounter);

		//update previous frame and points for optical flow
		imgGreyOld = imgGrey.clone();
		imgBinaryOld = imgBinary.clone();
		oldBlobPoints = newBlobPoints;
		
		//imshow("imgGrey", imgGrey);
		imshow("Original", image); //show the original image
		namedWindow("Draw", 1);
		namedWindow("Binary", 1);
		createTrackbar("minGrey", "Binary", &minGrey, 255, 0);
		createTrackbar("maxGrey", "Binary", &maxGrey, 255, 0);
		imshow("Draw", imgDraw);
		imshow("Binary", imgBinary);


		if (recordMode) {
			imwrite("recorder/original" + to_string(frameCounter) + ".png", image);
			imwrite("recorder/draw" + to_string(frameCounter) + ".png", imgDraw);
			//imwrite("recorder/gray" + to_string(frameCounter) + ".png", imgGrey);
			imwrite("recorder/bin" + to_string(frameCounter) + ".png", imgBinary);
			//imwrite("recorder/backproj1" + to_string(frameCounter) + ".png", backprojImage);
			//imwrite("recorder/backproj2" + to_string(frameCounter) + ".png", backprojImage2);
			//imwrite("recorder/HLSthresholded" + to_string(frameCounter) + ".png", imgThresholded);
			//imwrite("recorder/optflow" + to_string(frameCounter) + ".png", bgr);
		}

		if (waitKey(0) == 32) //frame by frame with 'space'
		{
			cout << "space key is pressed by user" << endl;
		}

		char c = (char)waitKey(10);
		switch (c)
		{
		case 'b':
			backprojMode = !backprojMode;
			break;
		case 'c':
			trackObject1 = 0;
			histimg = Scalar::all(0);
			break;
		case 'h':
			showHist = !showHist;
			if (!showHist)
				destroyWindow("Histogram");
			else
				namedWindow("Histogram", 1);
			break;
		case 'p':
			paused = !paused;
			break;
		case 'q':
			if (paused) {

			}
			break;
		case 'r':
			recordMode = !recordMode;
			break;
		case 's':
			imwrite("recorder/original" + to_string(frameCounter) + ".png", image);
			imwrite("recorder/draw" + to_string(frameCounter) + ".png", imgDraw);
			//imwrite("recorder/gray" + to_string(frameCounter) + ".png", imgGrey);
			imwrite("recorder/bin" + to_string(frameCounter) + ".png", imgBinary);
			//imwrite("recorder/backproj1" + to_string(frameCounter) + ".png", backprojImage);
			//imwrite("recorder/backproj2" + to_string(frameCounter) + ".png", backprojImage2);
			//imwrite("recorder/HLSthresholded" + to_string(frameCounter) + ".png", imgThresholded);
			//imwrite("recorder/optflow" + to_string(frameCounter) + ".png", bgr);
			break;
		default:
			;//cout << c;
		}

		if (!paused) {
			float fps = getTickFrequency() / ((double)getTickCount() - t);
			cout << "FPS: " << fps << endl;
		}
	}

	return 0;
}