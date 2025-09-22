#include <vector>
#include <string>


std::vector<std::vector<std::string>> ExtractCsvCells(std::ifstream& fin, const char& delimiter);
std::vector<std::vector<uint8_t>> ExtractCsvCellsAsBytes(std::ifstream& fin, const char& delimiter);
void PrintCsvCells(const std::vector<std::vector<std::string>>& cells);
void PrintCsvCells(const std::vector<std::vector<uint8_t>>& cells);
std::vector<cv::Mat> ExtractDigitsDataCsv(const std::string& filePath);
