#include <iostream>
#include <opencv2/opencv.hpp>
#include <tensorflow/c/c_api.h>

#include "dataset.hpp"

void PrintMatrix2D(const cv::Mat& mat, const cv::Size& size) {
	const uchar* data = mat.data;

	for (size_t i = 0; i < size.width; i++) {
		for (size_t j = 0; j < size.height; j++) {
			size_t index = i * size.width + j;

			std::cout <<(int)data[index] << " ";
		}

		std::cout << std::endl;
	}
}

int main()
{
	/*
	const auto digitsData = ExtractDigitsDataCsv("Train.csv");
	const auto& digits = digitsData.first;
	const auto& images = digitsData.second;

	const auto resizedImage = cv::Mat(28 * 5, 28 * 5, CV_8U);
	std::cout << (int)digits[50] << '\n';
	cv::resize(images[50], resizedImage, cv::Size(28 * 5, 28 * 5));

	cv::imshow("Hello open CV", resizedImage);

	cv::waitKey(0);

	//PrintMatrix2D(digitsData[0], {28, 28});
	*/

	std::cout << TF_Version() << std::endl;

	return 0;
}
