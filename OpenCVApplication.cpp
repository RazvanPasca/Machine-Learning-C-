// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include "OpenCVApplication.h"
#include "time.h"
#include <random>
#include <stdio.h>

using namespace std;

void testOpenImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image", src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName) == 0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName, "bmp");
	while (fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(), src);
		if (waitKey() == 27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				uchar neg = 255 - val;
				dst.at<uchar>(i, j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int)src.step; // no dword alignment is done !!!
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i, j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i, j) = (r + g + b) / 3;
			}
		}

		imshow("input image", src);
		imshow("gray image", dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele d eculoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				int hi = i * width * 3 + j * 3;
				int gi = i * width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1, dst2;
		//without interpolation
		resizeImg(src, dst1, 320, false);
		//with interpolation
		resizeImg(src, dst2, 320, true);
		imshow("input image", src);
		imshow("resized image (without interpolation)", dst1);
		imshow("resized image (with interpolation)", dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src, dst, gauss;
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int)k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss, dst, pL, pH, 3);
		imshow("input image", src);
		imshow("canny", dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}

	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		Canny(grayFrame, edges, 40, 100, 3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = cvWaitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n");
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];

	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;

		imshow(WIN_SRC, frame);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115) { //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess)
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
			x, y,
			(int)(*src).at<Vec3b>(y, x)[2],
			(int)(*src).at<Vec3b>(y, x)[1],
			(int)(*src).at<Vec3b>(y, x)[0]);
	}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

/* Histogram display function - display a histogram using bars (simlilar to L3 / PI)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i < hist_cols; i++)
		if (hist[i] > max_hist)
			max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

void leastSquares() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		FILE* f = fopen(fname, "r");
		int n;
		float x, y;
		fscanf(f, "%d", &n);
		float* points_x = (float*)malloc(n * sizeof(float));
		float* points_y = (float*)malloc(n * sizeof(float));

		float minx = 50000.0f;
		float miny = minx;

		float maxx = -minx;
		float maxy = -miny;


		for (int i = 0; i < n; i++) {
			fscanf(f, "%f%f", &x, &y);

			points_x[i] = x;
			points_y[i] = y;

			minx = min(minx, x);
			miny = min(miny, y);

			maxx = max(maxx, x);
			maxy = max(maxy, y);

		}

		Mat img(maxy - miny + 50, maxx - minx + 50, CV_8UC3, Scalar(255, 255, 255));

		float x_sum = 0;
		float y_sum = 0;
		float x_sq_sum = 0;
		float y_sq_sum = 0;
		float xy_sum = 0;
		float ysq_xsq_diff_sum = 0;

		for (int i = 0; i < n; i++) {

			points_x[i] -= minx;;
			points_y[i] -= miny;

			x_sum += points_x[i];
			y_sum += points_y[i];
			xy_sum += points_x[i] * points_y[i];
			x_sq_sum += pow(points_x[i], 2);
			y_sq_sum += pow(points_y[i], 2);
			ysq_xsq_diff_sum += (points_y[i] * points_y[i] - points_x[i] * points_x[i]);

			img.at<Vec3b>((int)points_y[i], (int)points_x[i])[0] = 0;
			img.at<Vec3b>((int)points_y[i], (int)points_x[i])[1] = 0;
			img.at<Vec3b>((int)points_y[i], (int)points_x[i])[2] = 0;
		}

		float theta1, theta0;
		float beta;
		float ro;

		theta1 = (n * xy_sum - x_sum * y_sum) / (n*x_sq_sum - pow(x_sum, 2));
		theta0 = 1.0f / n * (y_sum - theta1 * x_sum);

		beta = -1.0f / 2 * (atan2(2 * xy_sum - 2.0f * (x_sum*y_sum) / n, ysq_xsq_diff_sum + 1.0f* (x_sum*x_sum) / n - 1.0f* (y_sum*y_sum) / n));
		ro = 1.0f / n * (cos(beta) * x_sum + sin(beta)*y_sum);

		int y1 = theta0;
		int y2 = theta0 + theta1 * (maxx - minx);
		line(img, Point(0, y1), Point(maxx - minx, y2), Scalar(0, 255, 0), 3);

		int y3 = 0;
		int y4 = 0;

		int x3 = 0; int x4 = 0;

		if (fabs(beta) > PI / 4 && fabs(beta) < 3 * PI / 4) {

			y3 = ro / sin(beta);
			y4 = (ro - (maxx - minx)*cos(beta)) / sin(beta);
			line(img, Point(0, y3), Point(maxx - minx, y4), Scalar(255, 0, 0));


		}
		else {

			x3 = ro / cos(beta);
			x4 = (ro - (maxy - miny)*sin(beta)) / cos(beta);
			line(img, Point(x3, 0), Point(x4, maxy - miny), Scalar(255, 0, 0));
		}
		printf("maxx-minx: %f ; maxx: %f \n", maxx - minx, maxx);

		printf("betha and ro %f %f", beta, ro);

		imshow("title", img);
		fclose(f);

		waitKey();
	}
}

float distance(int x, int y, int a, int b, int c) {
	return fabs(a*x + b * y + c) / sqrt(a*a + b * b);
}

void ransac() {
	char fname[MAX_PATH];
	srand(time(NULL));
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);

		vector<Point2d> points;

		int height = src.rows;
		int width = src.cols;

		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
				if (src.at<uchar>(i, j) == 0)
					points.push_back(Point2d(j, i));

		int k = 1;
		float q = 0.3f;
		float dist = 10.0f;
		float p = 0.99f;
		int s = 2;

		int N = log(1 - p) / log(1 - pow(q, s));

		int max_consensus = 0;
		int a, b, c;
		int point1 = 0;
		int point2 = 0;
		int best_p1 = 0;
		int best_p2 = 0;

		while (k <= N) {

			//step 1
			int point1 = rand() % points.size();
			int point2 = rand() % points.size();
			while (point1 == point2) {
				point2 = rand() % points.size();
			}

			//step 2
			int a_cand = points.at(point1).y - points.at(point2).y;
			int b_cand = points.at(point2).x - points.at(point1).x;
			int c_cand = points.at(point1).x*points.at(point2).y - points.at(point2).x*points.at(point1).y;

			int consensus = 0;
			for (Point2d point : points) {
				if (distance(point.x, point.y, a_cand, b_cand, c_cand) <= dist)
					consensus++;
			}

			//step 3
			if (consensus > max_consensus) {
				max_consensus = consensus;
				a = a_cand;
				b = b_cand;
				c = c_cand;
				best_p1 = point1;
				best_p2 = point2;
			}

			//step 4
			if (max_consensus > q*points.size())
				break;

			//step 5
			k++;

		}

		printf("N is %d, k is %d \n", N, k);
		line(src, points.at(best_p1), points.at(best_p2), Scalar(0, 0, 0), 2);
		imshow("image", src);
		waitKey();
	}
}

struct peak {
	int theta; int ro; int votes;
	bool operator < (const peak& o) const {
		return votes > o.votes;
	}
};

void hough() {
	char fname[MAX_PATH];
	srand(time(NULL));
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int D = sqrt(src.cols*src.cols + src.rows*src.rows);
		int rows = src.rows;
		int cols = src.cols;
		int ro = 0;
		int maxVotes = 0;

		Mat Hough(360, D + 1, CV_32SC1); //matrix with int values

		for (int i = 0; i < 360; i++)
			for (int j = 0; j < D + 1; j++)
				Hough.at<int>(i, j) = 0;

		for (int i = 0; i < rows; i++)
			for (int j = 0; j < cols; j++)
				for (int theta = 0; theta < 360; theta++) {
					if (src.at<uchar>(i, j) > 0) {
						float rad = theta * PI / 180.0f;
						ro = j * cos(rad) + i * sin(rad);
						if (ro > 0 && ro < D + 1) {
							Hough.at<int>(theta, ro)++;
							maxVotes = max(Hough.at<int>(theta, ro), maxVotes);
						}
					}
				}
		Mat houghSpace;
		Hough.convertTo(houghSpace, CV_8UC1, 255.f / maxVotes);

		int windowSize = 3;
		vector<peak> peaks;
		int nr_lines = 10;

		for (int i = 0; i < 360; i++)
			for (int j = 0; j < D + 1; j++) {
				boolean isPeak = true;
				for (int h = -windowSize; h <= windowSize; h++)
					for (int w = -windowSize; w <= windowSize; w++) {
						if (i + h < 360 && i + h>0 && j + w < D + 1 && j + w>0)
							if (Hough.at<int>(i + h, j + w) > Hough.at<int>(i, j))
								isPeak = false;
					}
				if (isPeak) {
					peak currPeak = { i,j,Hough.at<int>(i, j) };
					peaks.push_back(currPeak);
				}
			}

		std::sort(peaks.begin(), peaks.end());
		printf("%d\n", peaks.size());
		Mat color_img = imread(fname, CV_LOAD_IMAGE_COLOR);
		for (int n = 0; n < nr_lines; n++) {
			peak currPeak = peaks.at(n);
			float rad = currPeak.theta * PI / 180.0f;
			int y_0 = currPeak.ro / sin(rad);
			int y_1 = (currPeak.ro - src.cols*cos(rad)) / sin(rad);
			printf("%d, %d, %d for peak %d\n", currPeak.ro, currPeak.theta, currPeak.votes, n);
			line(color_img, Point(0, y_0), Point(src.cols, y_1), Scalar(0, 255, 0), 2);
		}

		imshow("lines", color_img);
		imshow("image", houghSpace.t());
		waitKey();
	}
}

int distToNeigh(int k, int l) {
	if ((k == -1 && l == -1) || (k == -1 && l == 1))
		return 3;
	if (k == 1 && l == 1 || k == 1 && l == -1)
		return 3;
	if (k == 0 && l == 0)
		return 0;
	return 2;
}

uchar getDistance(Mat dt, int i, int j, int k, int l) {
	//printf("i reached i,j,k,l %d %d %d %d\n", i,j,k,l);
	if (dt.at<uchar>(i + k, j + l) + distToNeigh(k, l) > 255)
		return 255;
	else
		return dt.at<uchar>(i + k, j + l) + distToNeigh(k, l);
}

uchar min5(uchar a, uchar b, uchar c, uchar d, uchar e) {
	return min(a, min(b, min(c, min(d, e))));
}

void distanceTransform() {
	char fname[MAX_PATH];
	srand(time(NULL));
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);

		Mat dt = src.clone();

		int cols = src.cols;
		int rows = src.rows;

		//1st pass
		for (int i = 1; i < rows - 1; i++)
			for (int j = 1; j < cols - 1; j++)
				for (int l = -1; l <= 1; l++)
					for (int k = -1; k <= 0; k++)
						if (k != 0 || l != 1)
							dt.at<uchar>(i, j) = min(getDistance(dt, i, j, k, l), dt.at<uchar>(i, j));

		//2nd pass
		for (int i = rows - 2; i > 0; i--)
			for (int j = cols - 2; j > 0; j--)
				for (int l = 1; l > -2; l--)
					for (int k = 1; k > -1; k--)
						if (k != 0 || l != 1)
							dt.at<uchar>(i, j) = min(getDistance(dt, i, j, k, l), dt.at<uchar>(i, j));


		openFileDlg(fname);
		Mat unkObj;
		unkObj = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);

		cols = unkObj.cols;
		rows = unkObj.rows;
		int sumOfDistances = 0;
		int nrContourPoints = 0;

		for (int i = 0; i < rows; i++)
			for (int j = 0; j < cols; j++)
				if (unkObj.at<uchar>(i, j) == 0) {
					sumOfDistances += dt.at<uchar>(i, j);
					nrContourPoints++;
					dt.at<uchar>(i, j) = 0;
				}

		printf("mean of distances is %d", sumOfDistances / nrContourPoints);

		waitKey();
	}
}

void statisticalAnalysis() {
	int close = 0;
	int nr_pictures = 400;
	int width = 361;
	Mat featureMatrix(nr_pictures, width, CV_8UC1);

	while (!close)
	{
		Mat src;
		char fname[MAX_PATH];
		for (int i = 1; i <= nr_pictures; i++) {
			sprintf(fname, "stats/face%05d.bmp", i);
			src = imread(fname, 0);
			int rows = src.rows;
			int cols = src.cols;
			for (int row = 0; row < rows; row++)
				for (int col = 0; col < cols; col++)
					featureMatrix.at<uchar>(i - 1, row*rows + col) = src.at<uchar>(row, col);
		}

		vector<float> feature_means(361);
		vector<float> feature_deviations(361);

		//Calculating means
		for (int i = 0; i < nr_pictures; i++)
			for (int j = 0; j < 361; j++)
				feature_means.at(j) += featureMatrix.at<uchar>(i, j);

		for (int i = 0; i < 361; i++)
			feature_means.at(i) /= (float)400;

		//Calculating deviations
		for (int i = 0; i < nr_pictures; i++)
			for (int j = 0; j < 361; j++)
				feature_deviations.at(j) += pow(featureMatrix.at<uchar>(i, j) - feature_means.at(j), 2);

		for (int i = 0; i < 361; i++)
			feature_deviations.at(i) = sqrt(feature_deviations.at(i) / float(400));

		//Calculating covariances
		Mat covarianceMatrix(361, 361, CV_32F);
		Mat correlationMatrix(361, 361, CV_32F);
		for (int pic = 0; pic < 400; pic++)
			for (int i = 0; i < 361; i++)
				for (int j = 0; j < 361; j++)
					if (i == j)
						covarianceMatrix.at<float>(i, j) += feature_deviations.at(i)*feature_deviations.at(i);
					else
						covarianceMatrix.at<float>(i, j) += (featureMatrix.at<uchar>(pic, i) - feature_means.at(i))*(featureMatrix.at<uchar>(pic, j) - feature_means.at(j));

		for (int i = 0; i < 361; i++)
			for (int j = 0; j < 361; j++) {
				covarianceMatrix.at<float>(i, j) /= (float)400;
				correlationMatrix.at<float>(i, j) = covarianceMatrix.at<float>(i, j) / (feature_deviations.at(i)*feature_deviations.at(j));
			}

		FILE* covarFile = fopen("csv/covar.csv", "w");
		FILE* corelFile = fopen("csv/corel.csv", "w");
		for (int i = 0; i < 361; i++)
			for (int j = 0; j < 361; j++) {
				fprintf(covarFile, "%f%s", covarianceMatrix.at<float>(i, j), (j == 360) ? "\n" : ",");
				fprintf(corelFile, "%f%s", correlationMatrix.at<float>(i, j), (j == 360) ? "\n" : ",");
			}

		fclose(covarFile);
		fclose(corelFile);

		int row1 = 5;
		int col1 = 4;

		int row2 = 5;
		int col2 = 9;

		Mat correlChart(256, 256, CV_8UC1, 255);

		for (int i = 0; i < nr_pictures; i++) {
			int feature1 = featureMatrix.at<uchar>(i, row1 * 19 + col1);
			int feature2 = featureMatrix.at<uchar>(i, row2 * 19 + col2);
			correlChart.at<uchar>(255 - feature1, 255 - feature2) = 0;
		}
		printf("Correlation between f1 and f2 is %f\n", correlationMatrix.at<float>(row1 * 19 + col1, row2 * 19 + col2));
		imshow("Correlation chart", correlChart);
		waitKey();
		close = 1;
	}
}

float distance(Vec3i p1, Point p2) {
	float x_dif = p1[0] - p2.x;
	float y_dif = p1[1] - p2.y;
	return sqrt(pow(x_dif, 2) + pow(y_dif, 2)) / 1.0f;
}

void K_means(int k) {
	char fname[MAX_PATH];
	srand(time(NULL));
	while (openFileDlg(fname))
	{
		vector<Point> cluster_means;
		vector<Vec3i> image_points;
		vector<Vec3b> colors;

		Mat src;
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);

		int rows = src.rows;
		int cols = src.cols;
		Mat clustered_img(rows, cols, CV_8UC3, CV_RGB(255, 255, 255));



		for (int i = 0; i < rows; i++)
			for (int j = 0; j < cols; j++)
				if (src.at<uchar>(i, j) == 0)
					image_points.push_back(Vec3i(j, i, 0));

		default_random_engine gen_rows;
		gen_rows.seed(time(NULL));
		default_random_engine gen_cols;
		gen_cols.seed(time(NULL) + 312894);
		default_random_engine gen_colors;
		gen_colors.seed(time(NULL) + 1523);

		uniform_int_distribution<int> dist_rows(0, rows - 1);
		uniform_int_distribution<int> dist_cols(0, cols - 1);
		uniform_int_distribution<int> dist_colors(0, 255);

		for (int i = 0; i < k; i++) {
			int rand_x = dist_cols(gen_cols);
			int rand_y = dist_rows(gen_rows);
			cluster_means.push_back(Point(rand_x, rand_y));

			int r = dist_colors(gen_colors);
			int g = dist_colors(gen_colors);
			int b = dist_colors(gen_colors);
			colors.push_back(Vec3b((uchar)r, (uchar)g, (uchar)b));
		}

		/*for (Vec3i cluster : image_points) {
			printf("Our points are %d %d \n", cluster[0], cluster[1]);
		}*/

		boolean changed = true;
		int iterations = 0;
		while (changed) {
			changed = false;
			printf("reached iteration %d\n", iterations++);

			//Assignation
			for (int i = 0; i < image_points.size(); i++) {

				int min_k = 0;
				double min_dist = 5000000.0;
				int curr_cluster = -1;

				for (Point cluster : cluster_means) {
					curr_cluster++;

					float dist_to_point = distance(image_points.at(i), cluster);
					if (dist_to_point < min_dist) {
						min_dist = dist_to_point;
						min_k = curr_cluster;
					}
				}

				if (image_points.at(i)[2] != min_k)
					changed = true;
				image_points.at(i)[2] = min_k;
			}

			//Update 
			for (int i = 0; i < cluster_means.size(); i++) {

				int new_x = 0;
				int new_y = 0;
				int nr_points = 0;

				for (Vec3i image_point : image_points) {
					if (image_point[2] == i) {
						new_x += image_point[0];
						new_y += image_point[1];
						nr_points++;
					}
				}
				if (nr_points > 0) {
					cluster_means.at(i).x = new_x / nr_points;
					cluster_means.at(i).y = new_y / nr_points;
				}
			}

			Mat clustered_img(rows, cols, CV_8UC3, CV_RGB(255, 255, 255));

			//Colors
			for (Vec3i point : image_points) {
				clustered_img.at<Vec3b>(point[1], point[0]) = colors.at(point[2]);
			}

			for (Point cluster : cluster_means) {
				circle(clustered_img, Point2d(cluster), 5, Scalar(0, 0, 0));
			}

			imshow("clustered", clustered_img);
			waitKey();
		}

		for (Vec3i point : image_points) {
			clustered_img.at<Vec3b>(point[1], point[0]) = colors.at(point[2]);
		}

		for (Point cluster : cluster_means) {
			circle(clustered_img, Point2d(cluster), 5, Scalar(0, 0, 0));
		}
		imshow("clustered", clustered_img);
		waitKey();

		for (Point cluster : cluster_means)
			printf("Cluster coords are %d %d", cluster.y, cluster.x);

		Mat Voronoi(rows, cols, CV_8UC3, CV_RGB(255, 255, 255));
		for (int row = 0; row < rows; row++) {
			for (int col = 0; col < cols; col++) {

				int min_k = 0;
				float min_dist = 50000.0f;
				int curr_cluster = 0;

				for (Point cluster : cluster_means) {
					float dist_to_point = distance(Vec3i(col, row, -1), cluster);
					if (dist_to_point < min_dist) {
						min_dist = dist_to_point;
						min_k = curr_cluster;
					}
					curr_cluster++;
				}
				Voronoi.at<Vec3b>(row, col) = colors.at(min_k);
			}
		}
		imshow("Voronoi", Voronoi);
		waitKey();
	}
}



void my_PCA() {
	char fname[MAX_PATH];
	srand(time(NULL));
	while (openFileDlg(fname))
	{
		FILE* f = fopen(fname, "r");
		int n;
		int d;
		fscanf(f, "%d %d", &n, &d);
		Mat X(n, d, CV_32F);

		vector<float> means(d, 0.0);

		for (int row = 0; row < n; row++)
			for (int col = 0; col < d; col++) {
				float x;
				fscanf(f, "%f", &x);
				X.at<float>(row, col) = x;
				means.at(col) += x;
			}

		for (int i = 0; i < d; i++)
			means.at(i) /= n;

		for (int row = 0; row < n; row++)
			for (int col = 0; col < d; col++) {
				X.at<float>(row, col) -= means.at(col);
			}

		Mat C(d, d, CV_32F);
		C = X.t() * X / (n - 1);

		Mat Lambda, Q;
		eigen(C, Lambda, Q);
		Q = Q.t();

		for (int i = 0; i < d; i++)
			printf("Eigen value %d is %f\n", i, Lambda.at<float>(i));

		int k = (d == 7) ? 2 : 3;

		Mat K(d, k, CV_32F);
		for (int i = 0; i < d; i++)
			for (int j = 0; j < k; j++)
				K.at<float>(i, j) = Q.at<float>(i, j);

		Mat Xk(n, k, CV_32F);
		Xk = X * K;
		Mat X_tilda(n, d, CV_32F);

		X_tilda = Xk * K.t();
		float abs_dif = 0.0f;

		for (int row = 0; row < n; row++)
			for (int col = 0; col < d; col++)
				abs_dif += fabs(X.at<float>(row, col) - X_tilda.at<float>(row, col));
		abs_dif /= (n*d);

		printf("Mean abs difference is %f\n", abs_dif);

		float min_x = 32000.0f;
		float max_x = -32000.0f;

		float min_y = 32000.0f;
		float max_y = -32000.0f;

		float min_z = 32000.0f;
		float max_z = -32000.0f;

		for (int row = 0; row < n; row++)
			for (int col = 0; col < k; col++) {
				if (col == 0) {
					min_x = min(min_x, Xk.at<float>(row, col));
					max_x = max(max_x, Xk.at<float>(row, col));
				}
				if (col == 1) {
					min_y = min(min_y, Xk.at<float>(row, col));
					max_y = max(max_y, Xk.at<float>(row, col));
				}
				if (col == 2) {
					min_z = min(min_z, Xk.at<float>(row, col));
					max_z = max(max_z, Xk.at<float>(row, col));
				}
			}

		Mat img(max_y - min_y + 1, max_x - min_x + 1, CV_8UC1, Scalar(255));

		for (int row = 0; row < n; row++)
			if (k == 2)
				img.at<uchar>(Xk.at<float>(row, 0) - min_x, Xk.at<float>(row, 1) - min_y) = 0;

			else {

				int normalized = (Xk.at<float>(row, 2) - min_z) / (max_z - min_z) * 255;

				img.at<uchar>(Xk.at<float>(row, 0) - min_y, Xk.at<float>(row, 1) - min_x) = 255 - normalized;
			}

		imshow("Xk is", img.t());

	}
}

vector<int> computeHistogram(int bins, Mat img) {
	int rows = img.rows;
	int cols = img.cols;

	vector<int> r(bins, 0);
	vector<int> g(bins, 0);
	vector<int> b(bins, 0);
	int bin_size = 256 / bins;

	for (int row = 0; row < rows; row++)
		for (int col = 0; col < cols; col++) {
			Vec3b bgr = img.at<Vec3b>(row, col);
			b.at((int)bgr[0] / bin_size)++;
			g.at((int)bgr[1] / bin_size)++;
			r.at((int)bgr[2] / bin_size)++;
		}

	vector<int> final_histo;
	for (int i = 0; i < bins; i++)
		final_histo.push_back(b.at(i));

	for (int i = 0; i < bins; i++)
		final_histo.push_back(g.at(i));

	for (int i = 0; i < bins; i++)
		final_histo.push_back(r.at(i));

	return final_histo;
}

struct DataPoint {
	float distance; uchar class_name;
	bool operator < (const DataPoint& o) const {
		return distance < o.distance;
	}
};

int argmax(vector<int> numbers) {
	int max = -1;
	int argmax;
	for (int i = 0; i < numbers.size(); i++) {
		if (numbers.at(i) > max) {
			max = numbers.at(i);
			argmax = i;
		}
	}
	return argmax;
}


int argmax(vector<float> numbers) {
	float max = -500000.0f;
	int argmax;
	for (int i = 0; i < numbers.size(); i++) {
		if (numbers.at(i) > max) {
			max = numbers.at(i);
			argmax = i;
		}
	}
	return argmax;
}

float euclid_distance(vector<int> train_point, vector<int> target_point) {
	float sum_sq_difs = 0.0f;
	for (int i = 0; i < train_point.size(); i++)
		sum_sq_difs += pow(train_point.at(i) - target_point.at(i), 2);
	return sqrt(sum_sq_difs);
}

void KNN(int k, int bins) {
	const int nrclasses = 6;
	char classes[nrclasses][10] =
	{ "beach", "city", "desert", "forest", "landscape", "snow" };

	int nrinst = 672;
	int d = 3 * bins;

	Mat X(nrinst, d, CV_32S);
	Mat y(nrinst, 1, CV_8UC1);

	Mat C(nrclasses, nrclasses, CV_32S, Scalar(0));

	Mat src;
	char fname[MAX_PATH];
	int c = 0, fileNr = 0, rowX = 0;

	for (c = 0; c < nrclasses; c++) {
		fileNr = 0;
		while (1) {
			sprintf(fname, "KNN/train/%s/%06d.jpeg", classes[c], fileNr++);
			Mat img = imread(fname, CV_LOAD_IMAGE_COLOR);
			if (img.cols == 0) break;
			vector<int> histogram = computeHistogram(bins, img);
			for (int feature = 0; feature < d; feature++)
				X.at<int>(rowX, feature) = histogram.at(feature);
			y.at<uchar>(rowX) = c;
			rowX++;
		}
	}

	c = 0;
	for (c = 0; c < nrclasses; c++) {
		fileNr = 0;
		while (1) {
			sprintf(fname, "KNN/test/%s/%06d.jpeg", classes[c], fileNr++);
			Mat img = imread(fname, CV_LOAD_IMAGE_COLOR);
			if (img.cols == 0) break;

			vector<int> train_point_histo = computeHistogram(bins, img);
			vector<DataPoint> neighbors(X.rows);
			vector<int> neighbor_histogram(3 * bins);

			//Compute distance to all neighbors
			for (int row = 0; row < X.rows; row++) {
				for (int feature = 0; feature < bins * 3; feature++)
					neighbor_histogram.at(feature) = X.at<int>(row, feature);

				float distance = euclid_distance(train_point_histo, neighbor_histogram);
				DataPoint data_point = { distance, y.at<uchar>(row) };
				neighbors.at(row) = data_point;
			}

			//sort based on distance
			std::sort(neighbors.begin(), neighbors.end());
			vector<int> votes(nrclasses, 0);

			//compute votes
			for (int i = 0; i < k; i++)
				votes.at(neighbors[i].class_name) += 1;

			int predicted_class = argmax(votes);
			C.at<int>(c, predicted_class) += 1;
		}

	}
	int s = 0;
	int s_all = 0;;
	for (int i = 0; i < nrclasses; i++) {
		s += C.at<int>(i, i);
		for (int j = 0; j < nrclasses; j++) {
			s_all += C.at<int>(i, j);
			printf("%d  ", C.at<int>(i, j));
		}
		printf("\n");
	}
	printf("%f", (float)s / (float)s_all);
	imshow("text", X);
	waitKey();

}

vector<uchar> thresholdGreyscale(Mat src) {
	int rows = src.rows;
	int cols = src.cols;
	vector<uchar> img;

	for (int row = 0; row < rows; row++)
		for (int col = 0; col < cols; col++)
			if (src.at<uchar>(row, col) > 127)
				img.push_back(255);
			else img.push_back(0);

	return img;
}

void Naive_Bayes() {
	const int nrclasses = 10;
	char classes[nrclasses][10] =
	{ "0", "1", "2", "3", "4" , "5","6","7","8","9" };

	Mat X(0, 784, CV_32S);
	Mat Y(0, 1, CV_8UC1);

	Mat C(nrclasses, nrclasses, CV_32S, Scalar(0));

	Mat src;
	char fname[MAX_PATH];
	int c = 0, fileNr = 0, rowX = 0;

	int class_occurences[10] = { 0 };
	int nr_of_images = 0;

	Mat likelihoodMatrix(nrclasses, 784, CV_32F, Scalar(0));

	for (c = 0; c < nrclasses; c++) {
		fileNr = 0;
		while (1) {
			sprintf(fname, "Bayes/train/%s/%06d.png", classes[c], fileNr++);
			Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
			if (img.cols == 0) break;
			vector<uchar> img_tresh = thresholdGreyscale(img);
			class_occurences[c]++;
			Y.push_back(c);

			for (int pixel = 0; pixel < 784; pixel++) {
				if (img_tresh.at(pixel) == 255)
					likelihoodMatrix.at<float>(c, pixel)++;
			}
			rowX++;

		}
	}

	printf("Loaded %d images\n", rowX);
	//Normalizing likelihoodMatrix
	for (c = 0; c < nrclasses; c++)
		for (int pixel = 0; pixel < 784; pixel++) {
			likelihoodMatrix.at<float>(c, pixel) == 0 ?
				likelihoodMatrix.at<float>(c, pixel) = 1e-5 : likelihoodMatrix.at<float>(c, pixel) /= class_occurences[c];
		}


	char fname1[MAX_PATH];
	while (openFileDlg(fname1))
	{
		Mat src;
		src = imread(fname1, CV_LOAD_IMAGE_GRAYSCALE);
		vector<uchar> img_tresh = thresholdGreyscale(src);
		vector<float> probabilities;

		for (c = 0; c < nrclasses; c++) {
			float posterior = 0.0f;

			for (int pixel = 0; pixel < 784; pixel++) {
				img_tresh.at(pixel) == 0 ? posterior += log(1 - likelihoodMatrix.at<float>(c, pixel)) : posterior += log(likelihoodMatrix.at<float>(c, pixel));
			}
			posterior += log((float)class_occurences[c] / rowX);
			probabilities.push_back(posterior);
		}

		int predicted_class = argmax(probabilities);
		for (float probability : probabilities)
			printf("The values are %f \n", probability);
		printf("Predicted class is %s\n", classes[predicted_class]);
		waitKey();
	}
}

void Perceptron(float learning_rate) {

	char fname1[MAX_PATH];
	while (openFileDlg(fname1))
	{
		Mat src;
		src = imread(fname1, CV_LOAD_IMAGE_COLOR);
		Vec3b red(0, 0, 255);
		Vec3b blue(255, 0, 0);

		float E_limit = 1e-5;
		int max_iter = 10e5;

		Mat W(3, 1, CV_32F, Scalar(0.1f));
		Mat X(0, 3, CV_32F);
		Mat Y(0, 1, CV_32F);
		int rows = src.rows;
		int cols = src.cols;

		for (int row = 0; row < rows; row++) {
			for (int col = 0; col < cols; col++) {
				if (src.at<Vec3b>(row, col) == red) {
					Y.push_back(-1.0f);
					float v[3] = { 1.0f,float(row),float(col) };
					Mat d(1, 3, CV_32F, v);
					X.push_back(d);
				}
				else if (src.at<Vec3b>(row, col) == blue) {
					Y.push_back(1.0f);
					float v[3] = { 1.0f,float(row),float(col) };
					Mat d(1, 3, CV_32F, v);
					X.push_back(d);
				}
			}
		}
		float width = (float)cols;

		int nr_iters = 0;
		float last_change;
		boolean missclassified = true;
		int nr_examples = X.rows;
		while (missclassified) {
			missclassified = false;
			for (int i = 0; i < nr_examples; i++) {
				Mat z = (W.t())*(X.row(i).t());
				float y_hat = z.at<float>(0, 0);
				//if misclassified, update weights
				if (y_hat*Y.at<float>(i) <= 0)
				{
					missclassified = true;
					W.at<float>(0, 0) += learning_rate * X.at<float>(i, 0)*Y.at<float>(i);
					W.at<float>(1, 0) += learning_rate * X.at<float>(i, 1)*Y.at<float>(i);
					W.at<float>(2, 0) += learning_rate * X.at<float>(i, 2)*Y.at<float>(i);
					nr_iters++;
				}
			}
		}

		printf("Weights %f, %f, %f", W.at<float>(0, 0), W.at<float>(1, 0), W.at<float>(2, 0));
		Point p1 = Point(0, (-W.at<float>(0, 0) / W.at<float>(1, 0)));
		Point p2 = Point(width, ((-W.at<float>(0, 0) - W.at<float>(2, 0)*width) / W.at<float>(1, 0)));

		line(src, p1, p2, CV_RGB(255, 0, 255));
		imshow("Imageinea", src);
		waitKey();
	}
}

const int MAXT = 20;

struct DT {
	int feature_index;
	int threshold;
	int class_label;
	double error;
	int classify(Mat single_point) {
		if (single_point.at<double>(feature_index) < threshold)
			return class_label;
		else
			return -class_label;
	}
};

struct Classifier {
	int T;
	double alphas[MAXT];
	DT dt_h[MAXT];
	int classify(Mat single_point) {
		double label = 0.0f;
		for (int i = 0; i < T; i++) {
			label += alphas[i] * (double)(dt_h[i].classify(single_point));
		}
		return label > 0 ? 1 : -1;
	}
};

DT findWeakDT(Mat X, Mat Y, Mat W, int rows, int cols);

void AdaBoost() {
	char fname1[MAX_PATH];
	while (openFileDlg(fname1))
	{
		Mat src;
		Mat X(0, 2, CV_64F);
		Mat Y(0, 1, CV_64F);
		src = imread(fname1, CV_LOAD_IMAGE_COLOR);
		Vec3b red(0, 0, 255);
		Vec3b blue(255, 0, 0);
		vector<Vec3b> colors;

		colors.push_back(Vec3b(255, 255, 20)); //yellow
		colors.push_back(Vec3b(20, 255, 255)); //cyan

		int rows = src.rows;
		int cols = src.cols;
		int nr_feat = 2;

		for (int row = 0; row < rows; row++) {
			for (int col = 0; col < cols; col++) {
				//cout << "Reached row:" << row << " Col:" << col << endl;
				if (src.at<Vec3b>(row, col) == red) {
					Y.push_back(1.0);
					double v[2] = { (double)row,(double)col };
					Mat d(1, 2, CV_64F, v);
					X.push_back(d);
				}
				else if (src.at<Vec3b>(row, col) == blue) {
					Y.push_back(-1.0);
					double v[2] = { (double)row,(double)col };
					Mat d(1, 2, CV_64F, v);
					X.push_back(d);
				}
			}
		}

		int nr_points = X.rows;

		Mat W(nr_points, 1, CV_64F, Scalar(1.0f / nr_points));


		cout << "Nr coloane X:" << X.cols << endl;
		cout << "Nr randuri X:" << X.rows << endl;

		Classifier adaBoostClassifier;

		adaBoostClassifier.T = MAXT;

		for (int i = 0; i < MAXT; i++) {
			DT simple_DT = findWeakDT(X, Y, W, rows, cols);
			double alpha = 0.5f*log((1 - simple_DT.error) / simple_DT.error);
			double weights_sum = 0.0f;
			for (int index = 0; index < nr_points; index++) {
				int label_hat = simple_DT.classify(X.row(index));
				double sign = Y.at<double>(index)*(double)label_hat;
				W.at<double>(index) *= (double)exp((-alpha) * sign);
				weights_sum += W.at<double>(index);
			}
			for (int index = 0; index < nr_points; index++)
				W.at<double>(index) /= weights_sum;
			adaBoostClassifier.alphas[i] = alpha;
			adaBoostClassifier.dt_h[i] = simple_DT;
		}


		Mat toDraw = Mat(src);
		for (int row = 0; row < rows; row++)
			for (int col = 0; col < cols; col++) {
				double v[2] = { (double)row,(double)col };
				if (toDraw.at<Vec3b>(row, col) != red && toDraw.at<Vec3b>(row, col) != blue) {
					Mat d(1, 2, CV_64F, v);
					int label = adaBoostClassifier.classify(d.row(0));
					toDraw.at<Vec3b>(row, col) = label == 1 ? colors.at(1) : colors.at(0);
				}
			}

		imshow("decision boundary", toDraw);
		waitKey();


	}
}

DT findWeakDT(Mat X, Mat Y, Mat W, int rows, int cols) {
	DT best_DT;
	double best_error = 10000.0f;
	for (int feature_index = 0; feature_index < X.cols; feature_index++) {
		int size = feature_index == 0 ? rows : cols;
		for (int threshold = 0; threshold < size; threshold++) {
			for (int label = -1; label <= 1; label += 2) {
				double error = 0.0f;
				//Go over all points and see how well we do
				DT curr_DT = { feature_index,threshold,label,0 };
				for (int point_index = 0; point_index < X.rows; point_index++) {
					double label_hat = (double)curr_DT.classify(X.row(point_index));
					if (label_hat*Y.at<double>(point_index) < 0)
						error += W.at<double>(point_index);
				}
				curr_DT.error = error;
				if (error < best_error) {
					best_error = error;
					best_DT = curr_DT;
				}
			}
		}
	}
	return best_DT;
}


int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative - diblook style\n");
		printf(" 4 - BGR->HSV\n");
		printf(" 5 - Resize image\n");
		printf(" 6 - Canny edge detection\n");
		printf(" 7 - Edges in a video sequence\n");
		printf(" 8 - Snap frame from live video\n");
		printf(" 9 - Mouse callback demo\n");
		printf(" 10 - Least squares\n");
		printf(" 11 - RANSAC\n");
		printf(" 12 - Hough Line\n");
		printf(" 13 - Distance Transform\n");
		printf(" 14 - Statistical Analysis\n");
		printf(" 15 - K Means\n");
		printf(" 16 - My PCA\n");
		printf(" 17 - KNN\n");
		printf(" 18 - Naive Bayes Class\n");
		printf(" 19 - Perceptron \n");
		printf(" 20 - AdaBoost \n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d", &op);
		switch (op)
		{
		case 1:
			testOpenImage();
			break;
		case 2:
			testOpenImagesFld();
			break;
		case 3:
			testParcurgereSimplaDiblookStyle(); //diblook style
			break;
		case 4:
			//testColor2Gray();
			testBGR2HSV();
			break;
		case 5:
			testResize();
			break;
		case 6:
			testCanny();
			break;
		case 7:
			testVideoSequence();
			break;
		case 8:
			testSnap();
			break;
		case 9:
			testMouseClick();
			break;
		case 10:
			leastSquares();
			break;
		case 11:
			ransac();
			break;
		case 12:
			hough();
			break;
		case 13:
			distanceTransform();
			break;
		case 14:
			statisticalAnalysis();
			break;
		case 15: {
			int k = 0;
			printf("Give the number of clusters");
			scanf("%d", &k);
			K_means(k);
			break;
		}
		case 16:
			my_PCA();
			break;
		case 17: {
			int k = 0;
			int bins = 0;
			printf("Give the number of neighs\n");
			scanf("%d", &k);
			printf("Give the number of bins\n");
			scanf("%d", &bins);
			KNN(k, bins);
			break; }
		case 18: {
			Naive_Bayes();
		}
		case 19: {
			float learning_rate;
			printf("Give the learning rate \n");
			scanf("%f", &learning_rate);
			Perceptron(learning_rate);
		}
		case 20: {
			AdaBoost();
		}
		}
	} while (op != 0);
	return 0;
}