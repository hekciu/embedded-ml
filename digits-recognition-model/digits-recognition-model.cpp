#include <iostream>
#include <opencv2/opencv.hpp>
#include <tensorflow/c/c_api.h>
#include <filesystem>

#include "dataset.hpp"
#include "model-training.hpp"


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


bool DirectoryExists(const std::string& directory_name) {
	return std::filesystem::exists(directory_name);
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

	bool restore = DirectoryExists("checkpoints");

	try {

		ModelDescription model("../../../frozen_models/graph_v1.pb");

		if (restore) {
			std::cout << "Restoring weights from checkpoint" << '\n';

			model.Checkpoint("./checkpoints/checkpoint", ModelDescription::CheckpointType::Restore);
		}
		else {
			std::cout << "Initializing model weights" << '\n';

			model.Init();
		}

		float testdata[] = { 1.0, 2.0, 3.0 };

		std::cout << "initial predictions: " << std::endl;
		model.Predict(testdata, 3);

		std::cout << "Training " << std::endl;
		for (int i = 0; i < 200; i++) {
			model.RunTrainStep();
		}

		std::cout << "Updated predictions: " << std::endl;
		model.Predict(testdata, 3);
	}
	catch (const std::exception& ex) {
		std::cerr << ex.what() << std::endl;
		return 1;
	}

	return 0;
}
