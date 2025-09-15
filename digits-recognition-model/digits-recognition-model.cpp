// digits-recognition-model.cpp: definiuje punkt wejścia dla aplikacji.
//

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;

int main()
{
	std::string image_path = "test_image.jpg";

	cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);

	cv::imshow("Hello open CV", img);

	cv::waitKey(0);

	return 0;
}
