#include <vector>
#include <fstream>
#include <stdint.h>

#include <opencv2/opencv.hpp>

#include "dataset.hpp"


std::pair<std::vector<uchar>, std::vector<cv::Mat>> ExtractDigitsDataCsv(const std::string& filePath) {
	std::ifstream fin(filePath);

	auto rows = ExtractCsvCellsAsBytes(fin, ',');

	fin.close();

	std::vector<uchar> digits = {};
	std::vector<cv::Mat> images = {};

	for (auto& row : rows) {
		//cv::Mat image({ 28, 28 }, row.data());

		digits.push_back(*row.data());
		const auto image = cv::Mat(28, 28, cv::DataType<uchar>::type, row.data() + 1);
		//const auto image = cv::Mat::create();

		const uchar* data = row.data();

		images.push_back(image.clone());
	}

	return { digits, images };
}


std::vector<std::vector<std::string>> ExtractCsvCells(std::ifstream& fin, const char& delimiter) {
	std::vector<std::vector<std::string>> output = {};

	std::string line;

	while (std::getline(fin, line)) {
		std::vector<std::string> row = {};

		std::string cell = "";

		for (const char& c : line) {
			if (c == delimiter) {
				row.push_back(cell);
				cell = "";

				continue;
			}

			cell += c;
		}

		row.push_back(cell);

		output.push_back(row);
	}

	return output;
}


std::vector<std::vector<uint8_t>> ExtractCsvCellsAsBytes(std::ifstream& fin, const char& delimiter) {
	std::vector<std::vector<uint8_t>> output = {};

	std::string line;

	// skip first one
	std::getline(fin, std::string());

	while (std::getline(fin, line)) {
		std::vector<uint8_t> row = {};

		std::string cell = "";

		for (const char& c : line) {
			if (c == delimiter) {
				uint8_t value = std::stoi(cell);

				row.push_back(value);
				cell = "";
				continue;
			}

			cell += c;
		}
		output.push_back(row);
	}

	return output;
}


void PrintCsvCells(const std::vector<std::vector<uint8_t>>& cells) {
	for (const auto& row : cells) {
		for (const auto& cell : row) {
			std::cout << cell << " ";
		}

		std::cout << std::endl;
	}
}


void PrintCsvCells(const std::vector<std::vector<std::string>>& cells) {
	for (const auto& row : cells) {
		for (const auto& cell : row) {
			std::cout << cell << " ";
		}

		std::cout << std::endl;
	}
}
