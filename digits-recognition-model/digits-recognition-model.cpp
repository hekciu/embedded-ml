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


std::vector<std::vector<float>> NormalizeDigitsData(const std::vector<cv::Mat>& data) {
	std::vector<std::vector<float>> output = {};

	const size_t image_size = 28 * 28;

	for (const auto& mat : data) {
		std::vector<float> img = {};

		for (size_t i = 0; i < image_size; i++) {
			img.push_back(static_cast<float>(*(mat.data + i)) / static_cast<float>(255.0));
		}

		output.push_back(img);
	}

	return output;
}


uint8_t GetWinningDigit(const float* data) {
	uint8_t output = 0;

	for (uint8_t i = 0; i < 10; i++) {
		if (*(data + i) > *(data + output)) {
			output = i;
		}
	}

	return output;
}


int main()
{
	/*
	const auto& digits = digitsData.first;
	const auto& images = digitsData.second;

	const auto resizedImage = cv::Mat(28 * 5, 28 * 5, CV_8U);
	std::cout << (int)digits[50] << '\n';
	cv::resize(images[50], resizedImage, cv::Size(28 * 5, 28 * 5));

	cv::imshow("Hello open CV", resizedImage);

	cv::waitKey(0);

	//PrintMatrix2D(digitsData[0], {28, 28});
	*/

	const auto digitsData = ExtractDigitsDataCsv("Train.csv");

	std::cout << TF_Version() << std::endl;

	//bool restore = DirectoryExists("checkpoints");

	bool restore = false;
	int prescaleTrainData = 1; // 1 => all train data

	try {

		ModelDescription model("../../../frozen_models/graph_v1.pb");

		if (restore) {
			std::cout << "Restoring weights from checkpoint" << '\n';

			model.Checkpoint("./checkpoints/checkpoint_real", ModelDescription::CheckpointType::Restore);
		}
		else {
			std::cout << "Initializing model weights" << '\n';

			model.Init();
		}

		const auto& normalizedDigitsData = NormalizeDigitsData(digitsData.second);

		//const auto& testdata = normalizedDigitsData[50].data();

		//const auto& initialPrediction = model.Predict(testdata);

		//std::cout << "initial predictions: " << GetWinningDigit(initialPrediction.data()) << std::endl;

		std::cout << "Training " << std::endl;
		for (int i = 0; i < normalizedDigitsData.size() / prescaleTrainData; i++) {
			if (i % 1000 == 0) {
				std::cout << "AAAAAA " << i << std::endl;
			}
			model.RunTrainStep(normalizedDigitsData[i], digitsData.first[i]);
		}

		for (int i = 0; i < 50; i++) {
			const auto& testdata = normalizedDigitsData[i].data();

			std::cout << "Predictions (should be: " << (int)digitsData.first[i] <<"): " << std::endl;
			const auto& initialPrediction = model.Predict(testdata);

			std::cout << "prediction: " << (int)GetWinningDigit(initialPrediction.data()) << std::endl;
		}

		// https://danishshres.medium.com/

		model.Checkpoint("./checkpoints/checkpoint_1", ModelDescription::CheckpointType::Save);
	}
	catch (const std::exception& ex) {
		std::cerr << ex.what() << std::endl;
		return 1;
	}

	return 0;
}
