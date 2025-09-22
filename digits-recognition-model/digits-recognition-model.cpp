#include <iostream>
#include <opencv2/opencv.hpp>

#include "dataset.hpp"

int main()
{
	/*
	std::string image_path = "test_image.jpg";

	cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);

	cv::imshow("Hello open CV", img);

	cv::waitKey(0);
	*/

	const auto digitsData = ExtractDigitsDataCsv("Train.csv");

	return 0;
}
