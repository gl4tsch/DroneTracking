#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int imgSizeX = 640;
int imgSizeY = 360;
Mat image;
Mat imgHLS;
Mat imgThresholded;
int frameCounter = 1;
double t;
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

//camshift
bool selectObject = false;
Rect selection;
Point origin;
int trackObject = 0;
int vmin = 250, vmax = 256, smin = 0;
Rect trackWindow;
int hsize = 16;
float hranges[] = { 0,180 };
const float* phranges = hranges;
Mat frame, hsv, hue, mask, hist, histimg = Mat::zeros(200, 320, CV_8UC3), backproj, backprojImage, cannyOut;
bool paused = false;
bool backprojMode = false;
bool showHist = true;



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
	cvtColor(image, imgHLS, COLOR_BGR2HLS); //Convert the captured frame from BGR to HLS
	inRange(imgHLS, Scalar(lowH, lowL, lowS), Scalar(highH, highL, highS), imgThresholded); //Threshold the image
	//inRange(imgHLS, Scalar(87, 230, 255), Scalar(94, 255, 255), imgThresholded); //Threshold for party_-2_l_l.mp4 green
	//inRange(imgHLS, Scalar(71, 169, 255), Scalar(98, 255, 255), imgThresholded); //Threshold for auto-darker3.mp4 blue
	//morphological operations
	morphClose(imgThresholded);
	morphOpen(imgThresholded);
	//find contours of filtered image using openCV findContours function
	findContours(imgThresholded, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
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
			trackObject = -1;   // Set up CAMShift properties in main() loop
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

Rect cutRectImgBounds(Rect r, int imgWidth, int imgHeight)
{
	if (r.x + r.width > imgWidth) {
		return Rect(r.tl(), Size(imgWidth - r.x, r.height));
	}
	else {
		return r;
	}
}

void fitBand(Rect roi)
{
	Canny(backprojImage(roi), cannyOut, 200, 200 * 2);
	findContours(cannyOut, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, roi.tl());
	vector<Point> contourPoints;
	for (vector<Point> v : contours) {
		for (Point p : v) {
			contourPoints.push_back(p);
		}
	}
	if (contourPoints.size() > 3) {
		RotatedRect rr = fitEllipse(contourPoints);
		ellipse(image, rr, Scalar(0, 255, 0), 3, LINE_AA);
	}
	drawContours(backprojImage, contours, -1, Scalar(255, 0, 255), 1, 8, hierarchy);
}

void trackCamshift() {
	if (!paused)
	{
		cvtColor(image, hsv, COLOR_BGR2HSV);
		if (trackObject)
		{
			int _vmin = vmin, _vmax = vmax;
			inRange(hsv, Scalar(0, smin, MIN(_vmin, _vmax)),
				Scalar(180, 256, MAX(_vmin, _vmax)), mask);
			int ch[] = { 0, 0 };
			hue.create(hsv.size(), hsv.depth());
			mixChannels(&hsv, 1, &hue, 1, ch, 1);
			if (trackObject < 0)
			{
				// Object has been selected by user, set up CAMShift search properties once
				Mat roi(hue, selection), maskroi(mask, selection);
				calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);
				normalize(hist, hist, 0, 255, NORM_MINMAX);
				trackWindow = selection;
				trackObject = 1; // Don't set up again, unless user selects new ROI
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
			RotatedRect trackBox = CamShift(backproj, trackWindow,
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
					fitBand(cutRectImgBounds(trackBox.boundingRect(), imgSizeX, imgSizeY));
					drawRotatedRect(trackBox, backprojImage);
					rectangle(backprojImage, trackBox.boundingRect(), Scalar(0, 0, 255));
					//ellipse(backprojImage, trackBox, Scalar(0, 0, 255), 3, LINE_AA);
				}
				imshow("Backprojection", backprojImage);
			}
		}
	}
	else if (trackObject < 0)
		paused = false;
	if (selectObject && selection.width > 0 && selection.height > 0)
	{
		Mat roi(image, selection);
		bitwise_not(roi, roi);
	}
	imshow("Histogram", histimg);
}


int main()
{
	//VideoCapture cap(0); //capture the video from web cam
	VideoCapture cap("party_-2_l_l.mp4"); //video file
	//VideoCapture cap("auto-darker3.mp4");
	//VideoCapture cap("night-normal.mp4");

	if (!cap.isOpened())  // if not success, exit program
	{
		cout << "Cannot open the web cam / file" << endl;
		return -1;
	}

	cap.set(CV_CAP_PROP_FRAME_WIDTH, imgSizeX);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, imgSizeY);

	//createTrackbars();

	//camshift temp
	namedWindow("Original", 1);
	setMouseCallback("Original", onMouse, 0);
	namedWindow("Backprojection", 1);
	createTrackbar("Vmin", "Original", &vmin, 256, 0);
	createTrackbar("Vmax", "Original", &vmax, 256, 0);
	createTrackbar("Smin", "Original", &smin, 256, 0);

	for(;;)
	{
		if (!paused) {
			t = (double)getTickCount(); //measure time
			bool frameRead = cap.read(frame); // read a new frame from video
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
		}
		resize(frame, image, Size(imgSizeX, imgSizeY), 0, 0, INTER_CUBIC); //resize to 640 by 360

		//detectHLSthresholds(); //show regions of specified HLS values
		trackCamshift();


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
			trackObject = 0;
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
		if (!paused) {
			t = ((double)getTickCount() - t) / getTickFrequency();
			cout << "Times passed in seconds: " << t << endl;
		}
	}

	return 0;
}