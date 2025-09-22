#include <vector>
#include <fstream>
#include <stdint.h>

#include <opencv2/opencv.hpp>

#include "dataset.hpp"


std::vector<std::vector<std::string>> ExtractCsvCells(std::ifstream& fin, const char& delimiter);
void PrintCsvCells(const std::vector<std::vector<std::string>>& cells);


std::vector<cv::Mat> ExtractDigitsDataCsv(const std::string& filePath) {
	std::ifstream fin(filePath);

	const auto cells = ExtractCsvCellsAsBytes(fin, ',');

	PrintCsvCells(cells);

	fin.close();

	return {};
}


std::vector<std::vector<std::string>> ExtractCsvCells(std::ifstream& fin, const char& delimiter) {
	std::vector<std::vector<std::string>> output = {};

	std::string line;

		size_t i = 0;

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

		std::cout << i << '\n';
		i++;

		output.push_back(row);
	}

	return output;
}


std::vector<std::vector<uint8_t>> ExtractCsvCellsAsBytes(std::ifstream& fin, const char& delimiter) {
	std::vector<std::vector<uint8_t>> output = {};

	std::string line;

	size_t i = 0;

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

		uint8_t value = std::stoi(cell);
		row.push_back(value);

		std::cout << i << '\n';
		i++;

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
