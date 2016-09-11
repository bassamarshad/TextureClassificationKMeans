#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <cmath>
#include <random>

using namespace cv;
using namespace std;

Mat generateMask2D(float allMasks[5][5], int row1, int row2);
void kmeansrgb(int NO_CLUSTERS, Mat src);
void kmeansgray(int NO_CLUSTERS, Mat src);
void kmeanstexture(int NO_CLUSTERS, Mat src);
void kmeanstextureBF(int NO_CLUSTERS, Mat src);
double leastDiff(int NO_CLUSTERS, Mat src, vector<Mat> masks);


int main(int /*argc*/, char** /*argv*/)
{
	/// Load source image and convert it to gray
	Mat src;

	// Read original image
	src = imread("Nat-4.jpg");

	// Show original image
	namedWindow("original image", WINDOW_AUTOSIZE);
	imshow("original image", src);


	// Decide Cluters number
	const int NO_CLUSTERS = 4;

	//kmeansrgb(NO_CLUSTERS, src);
	//kmeansgray(NO_CLUSTERS, src);
	kmeanstexture(NO_CLUSTERS, src);

	//kmeanstextureBF(NO_CLUSTERS, src);


	waitKey(0);
	return 0;
}


void kmeanstexture(int NO_CLUSTERS, Mat src)
{


	Mat gray;

	cvtColor(src, gray, COLOR_BGR2GRAY);
	Mat bestLabels, centers;
	Mat srcClustered(src.rows, src.cols, gray.type(), Scalar(0));


	// Create colors for the clusters should work for any number of clusters...one time lut
	// use for large number of clusters
	int logofnearestpow2 = log2(pow(ceil(log2(NO_CLUSTERS)), 2));
	int gray_increment = 255 / (int)pow(2, logofnearestpow2);
	

	vector<Scalar> colorVector;
	for (int i = 0; i < NO_CLUSTERS; i++)
	{
		for (int ir = 0; ir < 256; ir += gray_increment)
		{ 
			colorVector.push_back(Scalar(ir));
		}
	}



	// Create the features based on spatial and color coordinates:
	// xi, yi, f1,f2,f3,f4,f5
	Mat feature_img = Mat::zeros(src.cols*src.rows, 7, CV_32F);


	//-------------------------------------------------------//

	/*
	float level[5] = { 1  ,  4  , 6 ,  4  , 1 };
	float edge[5] = { -1 ,-2  , 0,   2,   1 };
	float spot[5] = { -1 ,  0 ,  2,   0 ,-1 };
	float wave[5] = { -1 ,  2,   0 ,-2 ,  1 };
	float ripple[5] = { 1 ,-4   , 6 ,-4 ,  1 };
	*/
	float allMasks[5][5] = { { 1, 4, 6, 4, 1 },      //level -- 0
	{ -1, -2, 0, 2, 1 },    //edge --  1
	{ -1, 0, 2, 0, -1 },    //spot -- 2
	{ -1, 2, 0, -2, 1 },    //wave -- 3 
	{ 1, -4, 6, -4, 1 } };  //ripple -- 4

	//Mat result;
	vector<Mat> masks;
	//First Mask number is transposed !



	masks.push_back(generateMask2D(allMasks, 0, 3));
	masks.push_back(generateMask2D(allMasks, 0, 4));
	masks.push_back(generateMask2D(allMasks, 1, 0));
	masks.push_back(generateMask2D(allMasks, 2, 3));
	masks.push_back(generateMask2D(allMasks, 3, 1));

	//cout << "M = " << endl << " " << result << endl << endl;

	//convolving with gray
	//Convolution with the kernels
	vector<Mat> convolved;
	for (int k = 0; k < 5; k++)
	{
		Mat conv;
		filter2D(gray, conv, -1, masks.at(k), Point(-1, -1), 0, BORDER_DEFAULT);
		convolved.push_back(conv);
	}

	// Rearrange feature vectors
	// each row is a feature vector as folow:
	// xi, yi, r, g, b
	for (int i = 0; i<src.cols*src.rows; i++)
	{
		feature_img.at<float>(i, 0) = (i / src.cols) / (float)src.rows;   // the / .rows to make it fractional
		feature_img.at<float>(i, 1) = (i % src.cols) / (float)src.cols;   // the / .cols to make the coordinate fractional.

		// use these two lines if you want to cluster based on color only
		//feature_img.at<float>(i, 0) = 0;
		//feature_img.at<float>(i, 1) = 0;

		feature_img.at<float>(i, 2) = convolved[0].data[i] / 255.0;
		feature_img.at<float>(i, 3) = convolved[1].data[i] / 255.0;
		feature_img.at<float>(i, 4) = convolved[2].data[i] / 255.0;
		feature_img.at<float>(i, 5) = convolved[3].data[i] / 255.0;
		feature_img.at<float>(i, 6) = convolved[4].data[i] / 255.0;
	}

	cv::kmeans(feature_img, NO_CLUSTERS, bestLabels,
		TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0),
		3, KMEANS_PP_CENTERS, centers);

	//cout << "M = " << endl << " " << gray << endl << endl;


	//Rearrange the results
	for (int i = 0; i < src.cols*src.rows; i++)
	{
		int y = i / src.cols;
		int x = i%src.cols;

		int clusterIdx = bestLabels.at<int>(i);
		//for (int c = 0; c < 3; c++)
		//{
		// use this with the large number of clusters color table in vector above...
		srcClustered.at<uchar>(y, x) = (uchar)colorVector.at(clusterIdx).val;

		// use for small number of cluster...usees the hard coded color in colorTab above.
		//srcClustered.at<Vec3b>(y, x)[c] = colorTab[clusterIdx].val[c];
		//}
	}

	// Show clustered image
	namedWindow("Texture-clustered image", WINDOW_AUTOSIZE);
	imshow("Texture-clustered image", srcClustered);


}

void kmeansrgb(int NO_CLUSTERS, Mat src)
{
	Mat bestLabels, centers;
	Mat srcClustered(src.rows, src.cols, CV_8UC3, Scalar(0, 0, 0));


	// Create colors for th clusters should work for nay number of clusters...one time lut
	// use for large number of clusters
	int logofnearestpow2 = log2(pow(ceil(log2(NO_CLUSTERS)), 2));
	int rcolor_increment = 255 / (int)pow(2, logofnearestpow2 / 3);
	int gcolor_increment = 255 / (int)pow(2, logofnearestpow2 / 3);
	int bcolor_increment = 255 / (int)pow(2, logofnearestpow2 - (2 * (logofnearestpow2 / 3)));

	//cout << "color " << logofnearestpow2 << endl;
	//cout << "color " << rcolor_increment << endl;
	//cout << "color " << gcolor_increment << endl;
	//cout << "color " << bcolor_increment << endl;

	vector<Scalar> colorVector;
	for (int i = 0; i < NO_CLUSTERS; i++)
	{
		for (int ir = 0; ir < 256; ir += rcolor_increment)
			for (int ig = 0; ig < 256; ig += gcolor_increment)
				for (int ib = 0; ib < 256; ib += bcolor_increment)
					colorVector.push_back(Scalar(ir, ig, ib));
	}

	// use for small number of clusters
	Scalar colorTab[] =
	{
		Scalar(0, 0, 255),
		Scalar(0, 255, 0),
		Scalar(255, 100, 100),
		Scalar(255, 0, 255),
		Scalar(0, 255, 255),
		Scalar(125, 250, 155),
		Scalar(25, 250, 55),
		Scalar(120, 50, 185),
		Scalar(0, 50, 155),
		Scalar(70, 90, 185)
	};

	// Create the features based on spatial and color coordinates:
	// xi, yi, r, g, b
	Mat feature_img = Mat::zeros(src.cols*src.rows, 5, CV_32F);
	// Split the image into 3 planes
	vector<Mat> bgr;
	cv::split(src, bgr);

	// Rearrange feature vectors
	// each row is a feature vector as folow:
	// xi, yi, r, g, b
	for (int i = 0; i<src.cols*src.rows; i++)
	{
		feature_img.at<float>(i, 0) = (i / src.cols) / (float)src.rows;   // the / .rows to make it fractional
		feature_img.at<float>(i, 1) = (i % src.cols) / (float)src.cols;   // the / .cols to make the coordinate fractional.

		// use these two lines if you want to cluster based on color only
		//feature_img.at<float>(i, 0) = 0;
		//feature_img.at<float>(i, 1) = 0;

		feature_img.at<float>(i, 2) = bgr[0].data[i] / 255.0;
		feature_img.at<float>(i, 3) = bgr[1].data[i] / 255.0;
		feature_img.at<float>(i, 4) = bgr[2].data[i] / 255.0;
	}

	cv::kmeans(feature_img, NO_CLUSTERS, bestLabels,
		TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0),
		3, KMEANS_PP_CENTERS, centers);

	//Rearrange the results
	for (int i = 0; i < src.cols*src.rows; i++)
	{
		int y = i / src.cols;
		int x = i%src.cols;

		int clusterIdx = bestLabels.at<int>(i);
		for (int c = 0; c < 3; c++)
		{
			// use this with the large number of clusters color table in vector above...
			srcClustered.at<Vec3b>(y, x)[c] = colorVector.at(clusterIdx).val[c];

			// use for small number of cluster...usees the hard coded color in colorTab above.
			//srcClustered.at<Vec3b>(y, x)[c] = colorTab[clusterIdx].val[c];
		}
	}

	// Show clustered image
	namedWindow("RGB-clustered image", WINDOW_AUTOSIZE);
	imshow("RGB-clustered image", srcClustered);

}



void kmeansgray(int NO_CLUSTERS, Mat src)
{

	Mat gray;

	cvtColor(src, gray, COLOR_BGR2GRAY);
	Mat bestLabels, centers;
	Mat srcClustered(src.rows, src.cols, gray.type(), Scalar(0));


	// Create colors for the clusters should work for any number of clusters...one time lut
	// use for large number of clusters
	int logofnearestpow2 = log2(pow(ceil(log2(NO_CLUSTERS)), 2));
	int gray_increment = 255 / (int)pow(2, logofnearestpow2);
	
	//cout << "color " << bcolor_increment << endl;

	vector<Scalar> colorVector;
	for (int i = 0; i < NO_CLUSTERS; i++)
	{
		for (int ir = 0; ir < 256; ir += gray_increment)
		{ 
			colorVector.push_back(Scalar(ir));
		}
	}

	

	// Create the features based on spatial and color coordinates:
	// xi, yi, grayLevl
	Mat feature_img = Mat::zeros(src.cols*src.rows, 3, CV_32F);
	
	// Rearrange feature vectors
	// each row is a feature vector as folow:
	// xi, yi, grayLevel
	for (int i = 0; i<src.cols*src.rows; i++)
	{
		feature_img.at<float>(i, 0) = (i / src.cols) / (float)src.rows;   // the / .rows to make it fractional
		feature_img.at<float>(i, 1) = (i % src.cols) / (float)src.cols;   // the / .cols to make the coordinate fractional.
		feature_img.at<float>(i, 2) = gray.data[i] / 255.0;
	
	}

	cv::kmeans(feature_img, NO_CLUSTERS, bestLabels,
		TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0),
		3, KMEANS_PP_CENTERS, centers);

	//Rearrange the results
	for (int i = 0; i < src.cols*src.rows; i++)
	{
		int y = i / src.cols;
		int x = i%src.cols;

		int clusterIdx = bestLabels.at<int>(i);
		//for (int c = 0; c < 3; c++)
		//{
		// use this with the large number of clusters color table in vector above...
		srcClustered.at<uchar>(y, x) = (uchar)colorVector.at(clusterIdx).val;

		// use for small number of cluster...usees the hard coded color in colorTab above.
		//srcClustered.at<Vec3b>(y, x)[c] = colorTab[clusterIdx].val[c];
		//}
	}

	// Show clustered image
	namedWindow("Gray-clustered image", WINDOW_AUTOSIZE);
	imshow("Gray-clustered image", srcClustered);

}


Mat generateMask2D(float allMasks[5][5], int row1, int row2)
{
	Mat mask = Mat::zeros(5, 5, CV_32F);

	for (int i = 0; i < 5; i++)
	{
		for (int j = 0; j < 5; j++)
		{
			mask.at<float>(i, j) = (float)(allMasks[row2][j] * allMasks[row1][i]);
			//mask[i][j]=(allMasks[row2][j] * allMasks[row1][i]);

		}
	}

	//Normalizing the kernel with sum of each element
	double sum1 = sum(mask)[0];

	if (sum1 != 0)
	{
		mask = mask / (float)sum1;
	}
	else if (sum1 == 0)
	{
		//cout << "\n bingo \n";
		mask = 0.0000001;
	}

	return mask;

}


void kmeanstextureBF(int NO_CLUSTERS, Mat src)
{


	//-------------------------------------------------------//

	/*
	float level[5] = { 1  ,  4  , 6 ,  4  , 1 };
	float edge[5] = { -1 ,-2  , 0,   2,   1 };
	float spot[5] = { -1 ,  0 ,  2,   0 ,-1 };
	float wave[5] = { -1 ,  2,   0 ,-2 ,  1 };
	float ripple[5] = { 1 ,-4   , 6 ,-4 ,  1 };
	*/
	float allMasks[5][5] = { { 1, 4, 6, 4, 1 },      //level -- 0
	{ -1, -2, 0, 2, 1 },    //edge --  1
	{ -1, 0, 2, 0, -1 },    //spot -- 2
	{ -1, 2, 0, -2, 1 },    //wave -- 3 
	{ 1, -4, 6, -4, 1 } };  //ripple -- 4

	//Mat result;
	vector<Mat> masks;

	for (int i = 0; i < 5; i++)
	{
		for (int j = 0; j < 5; j++)
		{
			//First Mask number is transposed !
			masks.push_back(generateMask2D(allMasks, i, j));
		}
	}

	double minval = 100;
	//int countloop = 0;
	//int arr1[10];

	String str;
	for (int i11 = 0; i11 < 25; i11++)
	{
		for (int i21 = i11 + 1; i21 < 25; i21++)
		{
			for (int i31 = i21 + 1; i31 < 25; i31++)
			{
				for (int i41 = i31 + 1; i41 < 25; i41++)
				{
					for (int i51 = i41 + 1; i51 < 25; i51++)
					{

						vector<Mat> tempVector;
						tempVector.push_back(masks.at(i11));
						tempVector.push_back(masks.at(i21));
						tempVector.push_back(masks.at(i31));
						tempVector.push_back(masks.at(i41));
						tempVector.push_back(masks.at(i51));

						double temp = leastDiff(NO_CLUSTERS, src, tempVector);
						if (temp <= minval)
						{
							minval = temp;

							str = to_string(i11) + " " + to_string(i21) + " " + to_string(i31) + " " + to_string(i41) + " " + to_string(i51);
							cout << "\n" << str;
							//cout << "\n" << i11 << " " << i12 << " " << i21 << " " << i22 << " " << i31 << " " << i32 << " " << i41 << " " << i42 << " " << i51 << " " << i52 << " \n";
						}


						//countloop++;
						//cout << "\n" << countloop;
						tempVector.clear();

					}
				}
			}
		}
	}


	cout << "\n" << str;

	//cout << "\n Total  binvals count : " << count;

	// Show clustered image
	//namedWindow("Texture-clusteredBF image", WINDOW_AUTOSIZE);
	//imshow("Texture-clusteredBF image", srcClustered);


}


double leastDiff(int NO_CLUSTERS, Mat src, vector<Mat> masks)
{
	Mat gray;

	cvtColor(src, gray, COLOR_BGR2GRAY);
	Mat bestLabels, centers;
	Mat srcClustered(src.rows, src.cols, gray.type(), Scalar(0));


	// Create colors for the clusters should work for any number of clusters...one time lut
	// use for large number of clusters
	int logofnearestpow2 = log2(pow(ceil(log2(NO_CLUSTERS)), 2));
	int gray_increment = 255 / (int)pow(2, logofnearestpow2);


	vector<Scalar> colorVector;
	for (int i = 0; i < NO_CLUSTERS; i++)
	{
		for (int ir = 0; ir < 256; ir += gray_increment)
		{
			colorVector.push_back(Scalar(ir));
		}
	}



	// Create the features based on spatial and color coordinates:
	// xi, yi,f1,f2,f3,f4,f5
	Mat feature_img = Mat::zeros(src.cols*src.rows, 7, CV_32F);


	//convolving with gray
	//Convolution with the kernels
	vector<Mat> convolved;
	for (int k = 0; k < 5; k++)
	{
		Mat conv;
		filter2D(gray, conv, -1, masks.at(k), Point(-1, -1), 0, BORDER_DEFAULT);
		convolved.push_back(conv);
	}




	// Rearrange feature vectors
	// each row is a feature vector as folow:
	// xi, yi, r, g, b
	for (int i = 0; i<src.cols*src.rows; i++)
	{
		feature_img.at<float>(i, 0) = (i / src.cols) / (float)src.rows;   // the / .rows to make it fractional
		feature_img.at<float>(i, 1) = (i % src.cols) / (float)src.cols;   // the / .cols to make the coordinate fractional.
		feature_img.at<float>(i, 2) = convolved[0].data[i] / 255.0;
		feature_img.at<float>(i, 3) = convolved[1].data[i] / 255.0;
		feature_img.at<float>(i, 4) = convolved[2].data[i] / 255.0;
		feature_img.at<float>(i, 5) = convolved[3].data[i] / 255.0;
		feature_img.at<float>(i, 6) = convolved[4].data[i] / 255.0;
	}

	cv::kmeans(feature_img, NO_CLUSTERS, bestLabels, TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0), 3, KMEANS_PP_CENTERS, centers);


	//Rearrange the results
	for (int i = 0; i < src.cols*src.rows; i++)
	{
		int y = i / src.cols;
		int x = i%src.cols;

		int clusterIdx = bestLabels.at<int>(i);
		//for (int c = 0; c < 3; c++)
		//{
		// use this with the large number of clusters color table in vector above...
		srcClustered.at<uchar>(y, x) = (uchar)colorVector.at(clusterIdx).val;

		// use for small number of cluster...usees the hard coded color in colorTab above.
		//srcClustered.at<Vec3b>(y, x)[c] = colorTab[clusterIdx].val[c];
		//}
	}

	//Histogram for bestLabels

	// Initialize parameters
	int histSize = 256;    // bin size
	float range[] = { 0, 255 };
	const float *ranges[] = { range };

	// Calculate histogram
	MatND hist;
	calcHist(&srcClustered, 1, 0, Mat(), hist, 1, &histSize, ranges, true, false);

	double total;
	total = gray.rows * gray.cols;
	//cout << "\n Total : " << total << "\n";

	double SumSquared = 0;
	for (int h = 0; h < histSize; h++)
	{
		float binVal = hist.at<float>(h);
		if (binVal != 0)
		{
			SumSquared += pow((binVal / total), 2);
			//cout << " " << binVal;
			//count += binVal;
		}
	}

	colorVector.clear();
	convolved.clear();
	return SumSquared;
}