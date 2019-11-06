#include <opencv2/opencv.hpp>
#include <iostream>
#include "test.h"

using namespace std;
using namespace cv;

int imgSizeX = 640;
int imgSizeY = 360;
Mat image;
Mat imgHLS;
Mat imgThresholded;
int frameCounter = 0;
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
int lmin1 = 220;
int lmax1 = 255;
int lmin2 = 220;
int lmax2 = 255;
Rect trackWindow, trackWindow2;
int hsize = 16;
float hranges[] = { 0,180 };
const float* phranges = hranges;
Mat frame, hls, hue, hue2, mask, mask2, hist, hist2, histimg = Mat::zeros(200, 320, CV_8UC3), histimg2 = Mat::zeros(200, 320, CV_8UC3), backproj, backproj2, backprojImage, backprojImage2, cannyOut, cannyOut2;
bool paused = false;
bool backprojMode = true;
bool showHist = true;
RotatedRect trackBox, trackBox2;
int thresh = 0;

//blob detect
Ptr<SimpleBlobDetector> detector;
vector<KeyPoint> keypoints;

const int bufferSize = 5;
Point2d pointBuffer[bufferSize];

Mat imgGrey, imgGreyOld, imgBinary;

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
vector<Point2f> blobPoints, oldBlobPoints, blobPredictions;


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
	erode(thresh, thresh, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
	dilate(thresh, thresh, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
}

void morphClose(Mat &thresh) {
	//morphological closing (fill small holes in the foreground)
	dilate(thresh, thresh, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
	erode(thresh, thresh, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
}

void detectHLSthresholds() {
	cvtColor(image, imgHLS, COLOR_BGR2HLS); //Convert the captured frame from BGR to HLS
	inRange(imgHLS, Scalar(lowH, lowL, lowS), Scalar(highH, highL, highS), imgThresholded); //Threshold the image
	//inRange(imgHLS, Scalar(87, 230, 255), Scalar(94, 255, 255), imgThresholded); //Threshold for party_-2_l_l.mp4 green
	//inRange(imgHLS, Scalar(71, 169, 255), Scalar(98, 255, 255), imgThresholded); //Threshold for auto-darker3.mp4 blue
	//morphological operations
	morphClose(imgThresholded);
	morphOpen(imgThresholded);
	//find contours of filtered image using openCV findContours function
	findContours(imgThresholded, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
	drawContours(image, contours, -1, Scalar(255, 0, 255), 2, 8, hierarchy);

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

void drawRotatedRect(RotatedRect rr, Mat img)
{
	Point2f vertices[4];
	rr.points(vertices);
	for (int i = 0; i < 4; i++)
		line(img, vertices[i], vertices[(i + 1) % 4], Scalar(0, 255, 0));
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
	Canny(backproj, cannyOut, 200, 200 * 2);
	findContours(cannyOut, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE); //, roi.tl());
	Canny(backproj2, cannyOut2, 200, 200 * 2);
	findContours(cannyOut2, contours2, hierarchy2, RETR_CCOMP, CHAIN_APPROX_SIMPLE); //, roi2.tl());

	vector<Point> contourPoints;
	for (vector<Point> v : contours) {
		for (Point p : v) {
			contourPoints.push_back(p);
		}
	}
	for (vector<Point> v : contours2) {
		for (Point p : v) {
			contourPoints.push_back(p);
		}
	}
	if (contourPoints.size() > 4) {
		RotatedRect rr = fitEllipse(contourPoints);
		ellipse(image, rr, Scalar(0, 255, 0), 3, LINE_AA);
		/*pointBuffer[frameCounter % bufferSize] = rr.center;
		if (frameCounter > bufferSize) {
			Point2d p = calculateMedian();
			line(image, p, p, Scalar(0, 0, 255), 10);
		}*/

	}
	drawContours(backprojImage, contours, -1, Scalar(255, 0, 255), 1, 8, hierarchy);
	drawContours(backprojImage2, contours2, -1, Scalar(255, 0, 255), 1, 8, hierarchy2);
	imshow("Backprojection", backprojImage);
	imshow("Backprojection2", backprojImage2);
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
		KeyPoint::convert(keypoints, blobPoints);
		if (blobPoints.size() > 4) {
			RotatedRect rr = fitEllipse(blobPoints);
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
					//fitBand(cutRectToImgBounds(trackBox.boundingRect(), imgSizeX, imgSizeY));
					drawRotatedRect(trackBox, backprojImage);
					rectangle(backprojImage, trackBox.boundingRect(), Scalar(0, 0, 255));
					//ellipse(image, trackBox, Scalar(0, 0, 255), 3, LINE_AA);
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
					//fitBand(cutRectToImgBounds(trackBox2.boundingRect(), imgSizeX, imgSizeY));
					drawRotatedRect(trackBox2, backprojImage2);
					rectangle(backprojImage2, trackBox2.boundingRect(), Scalar(0, 0, 255));
					//ellipse(image, trackBox2, Scalar(0, 255, 0), 3, LINE_AA);
				}
				imshow("Backprojection2", backprojImage2);
			}
		}
		if (trackObject1 && trackObject2) {
			//fitBand(cutRectToImgBounds(trackBox.boundingRect(), imgSizeX, imgSizeY), cutRectToImgBounds(trackBox2.boundingRect(), imgSizeX, imgSizeY));
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
	Mat blobMask = Mat::zeros(imgSizeY, imgSizeX, CV_8UC3);
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
	threshold(imgGrey, imgBinary, 240, 255, THRESH_BINARY);
	detector->detect(imgGrey, keypoints);
	drawKeypoints(imgGrey, keypoints, imgGrey, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	KeyPoint::convert(keypoints, blobPoints);
	imshow("binary", imgBinary);

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

void match2Dto3Dpoints() {
	//sort modelPoints3D and imagePoints2D to match
	//cv::projectPoints (InputArray objectPoints, InputArray rvec, InputArray tvec, InputArray cameraMatrix, InputArray distCoeffs, OutputArray imagePoints, OutputArray jacobian=noArray(), double aspectRatio=0)
	//optical flow
}

void initKalmanFilter(cv::KalmanFilter &KF, int nStates, int nMeasurements, int nInputs, double dt)
{
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

void kalman() {
	//https://docs.opencv.org/master/dc/d2c/tutorial_real_time_pose.html
	KalmanFilter KF;
	int nStates = 18;            // the number of states
	int nMeasurements = 6;       // the number of measured states
	int nInputs = 0;             // the number of action control
	double dt = 0.125;           // time between measurements (1/FPS)
	initKalmanFilter(KF, nStates, nMeasurements, nInputs, dt);    // init function

}


int main()
{
	//VideoCapture cap(0); //capture the video from web cam
	//VideoCapture cap("party_-2_l_l.mp4"); //video file
	//VideoCapture cap("auto-darker3.mp4");
	//VideoCapture cap("night-normal.mp4");
	VideoCapture cap("VID_20190920_155534.mp4");

	if (!cap.isOpened())  // if not success, exit program
	{
		cout << "Cannot open the web cam / file" << endl;
		return -1;
	}

	cap.set(CAP_PROP_FRAME_WIDTH, imgSizeX);
	cap.set(CAP_PROP_FRAME_HEIGHT, imgSizeY);

	//createTrackbars();

	//camshift temp
	namedWindow("Original", 1);
	setMouseCallback("Original", onMouse, 0);
	namedWindow("Backprojection", 1);
	namedWindow("Backprojection2", 1);
	createTrackbar("Lmin1", "Backprojection", &lmin1, 255, 0);
	createTrackbar("Lmax1", "Backprojection", &lmax1, 255, 0);
	createTrackbar("Lmin2", "Backprojection2", &lmin2, 255, 0);
	createTrackbar("Lmax2", "Backprojection2", &lmax2, 255, 0);


	//blob
	// Setup SimpleBlobDetector parameters.
	SimpleBlobDetector::Params params;
	
	// Change thresholds
	params.minThreshold = 240;
	params.maxThreshold = 255;

	// Filter by Area.
	params.filterByArea = true;
	params.minArea = 3;

	// Filter by Circularity
	params.filterByCircularity = false;
	params.minCircularity = 0.1;

	// Filter by Convexity
	params.filterByConvexity = false;
	params.minConvexity = 0.87;

	// Filter by Inertia
	params.filterByInertia = true;
	params.minInertiaRatio = 0.01;

	// Blob merge distance
	params.minDistBetweenBlobs = 3;

	// Blob intensity
	params.filterByColor = false;
	params.blobColor = 255;

	// Set up detector with params
	detector = SimpleBlobDetector::create(params);
	// SimpleBlobDetector::create creates a smart pointer. 
	// So you need to use arrow ( ->) instead of dot ( . )

	PnPapproxInit();

	for(;;)
	{
		if (!paused) {
			t = (double)getTickCount(); //start timer
			bool frameRead = cap.read(frame); // read a new frame from video
			if (!frameRead) //if fail, break loop
			{
				cout << "Cannot read a frame from video stream" << endl;
				break;
			}
			frameCounter++;
			if (frameCounter == cap.get(CAP_PROP_FRAME_COUNT)-1) //if end of video file reached, start from first frame
			{
				frameCounter = 0;
				cap.set(CAP_PROP_POS_FRAMES, 0);
				cout << "Video loop" << endl;
			}
		}
		resize(frame, image, Size(imgSizeX, imgSizeY), 0, 0, INTER_CUBIC); //resize to 640 by 360

		//detectHLSthresholds(); //show regions of specified HLS values
		//trackCamshift();
		//LEDdetect();
		//fitBandBlob();
		/*if (!backproj.empty() && !backproj2.empty()) {
			fitBandContours(cutRectToImgBounds(trackBox.boundingRect(), imgSizeX, imgSizeY), cutRectToImgBounds(trackBox2.boundingRect(), imgSizeX, imgSizeY));
		}*/

		greyLEDdetect();

		if (frameCounter > 1) { //if not first frame do optical flow
			calcOpticalFlowPyrLK(imgGreyOld, imgGrey, oldBlobPoints, blobPredictions, status, err, Size(15,15), 2, term);
			for (Point2f op : oldBlobPoints) {
				circle(imgGrey, op, 3, Scalar(255,0,0));//blue
			}
			for (Point2f pp : blobPredictions) {
				circle(imgGrey, pp, 3, Scalar(0, 255, 0));//green
			}
		}

		//solvePnPRansac(modelPoints3D, imagePoints2D, cameraMatrix, distortCoeffs, rotVec, transVec, false, iterationCount, reprojectionError, minInliers, inliersA, SOLVEPNP_IPPE);
		
		imshow("imgGrey", imgGrey);
		imshow("Original", image); //show the original image

		//if (waitKey(0) == 32) //frame by frame with 'space'
		//{
		//	cout << "space key is pressed by user" << endl;
		//}

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
		default:
			;
		}

		//update previous frame and points for optical flow
		imgGreyOld = imgGrey.clone();
		oldBlobPoints = blobPoints;

		if (!paused) {
			float fps = getTickFrequency() / ((double)getTickCount() - t);
			cout << "FPS: " << fps << endl;
		}
	}

	return 0;
}