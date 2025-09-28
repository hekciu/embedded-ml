#include <iostream>
#include <exception>
#include <fstream>
#include <vector>
#include "model-training.hpp"

// https://github.com/doleron/opencv-deep-learning-c-plusplus/blob/master/googlenet_classification.cpp
// https://gist.github.com/asimshankar/7c9f8a9b04323e93bb217109da8c7ad2


ModelDescription::ModelDescription(const std::string& filename) {
	status = TF_NewStatus();
	graph = TF_NewGraph();

	TF_SessionOptions* session_opts = TF_NewSessionOptions();
	session = TF_NewSession(graph, session_opts, status);
	TF_DeleteSessionOptions(session_opts);

	if (!Okay()) throw std::exception("Could not create TF session");

	TF_Buffer* graph_def = ReadFile(filename);

	if (graph_def == nullptr) throw std::exception("Could not read graph file");

	std::cout << "size of graph file in bytes: " << graph_def->length << std::endl;

	TF_ImportGraphDefOptions* graph_def_opts = TF_NewImportGraphDefOptions();
	TF_GraphImportGraphDef(graph, graph_def, graph_def_opts, status);

	TF_DeleteImportGraphDefOptions(graph_def_opts);
	TF_DeleteBuffer(graph_def);

	if (!Okay()) throw std::exception("Could not import graph def");

	input.oper = TF_GraphOperationByName(graph, "input");
	input.index = 0;

	if (input.oper == NULL) throw std::exception("input.oper is NULL");

	target.oper = TF_GraphOperationByName(graph, "target");
	target.index = 0;

	if (target.oper == NULL) throw std::exception("target.oper is NULL");

	output.oper = TF_GraphOperationByName(graph, "output");
	output.index = 0;

	if (output.oper == NULL) throw std::exception("output.oper is NULL");

	init_op = TF_GraphOperationByName(graph, "init");

	if (init_op == NULL) throw std::exception("init_op is NULL");

	train_op = TF_GraphOperationByName(graph, "Adam");

	if (train_op == NULL) throw std::exception("train_op is NULL");

	save_op = TF_GraphOperationByName(graph, "save/control_dependency");

	if (save_op == NULL) throw std::exception("save_op is NULL");

	restore_op = TF_GraphOperationByName(graph, "save/restore_all");

	if (restore_op == NULL) throw std::exception("restore_op is NULL");

	checkpoint_file.oper = TF_GraphOperationByName(graph, "save/Const");
	checkpoint_file.index = 0;

	if (checkpoint_file.oper == NULL) throw std::exception("checkpoint_file.oper is NULL");
};


ModelDescription::~ModelDescription() {
	TF_DeleteSession(session, status);

	if (!Okay()) std::cerr << "Could not delete tensorflow session" << std::endl;

	TF_DeleteGraph(graph);
	TF_DeleteStatus(status);
}


void ModelDescription::Init() {
	const TF_Operation* init_ops[] = {init_op};

	TF_SessionRun(
		session,
		NULL,
		/* Inputs */
		NULL,
		NULL,
		0,
		/* Outputs */
		NULL,
		NULL,
		0,
		/* Init operation */
		init_ops,
		1,
		NULL,
		status
	);

	if (!Okay()) throw std::exception("Could not start tensorflow session");
}


void ModelDescription::Checkpoint(const std::string& checkpoint_prefix, const CheckpointType& type) {
	TF_Tensor* tensor = ScalarStringTensor(checkpoint_prefix, status);

	if (!Okay()) {
		TF_DeleteTensor(tensor);
		throw std::exception("Could not create tensor from string");
	}

	TF_Output inputs[] = { checkpoint_file };
	TF_Tensor* input_values[] = { tensor };

	const TF_Operation* ops[] = {
		type == CheckpointType::Save ? save_op : restore_op
	};

	TF_SessionRun(
		session,
		NULL,
		/* Inputs */
		inputs,
		input_values,
		1,
		/* Outputs */
		NULL,
		NULL,
		0,
		/* Init operation */
		ops,
		1,
		NULL,
		status
	);

	TF_DeleteTensor(tensor);

	if (!Okay()) throw new std::exception("Failed to save/restore model");
}


static void VisualizePrediction(const float* prediction) {
	for (int i = 0; i < 10; i++) {
		std::cout << "digit: " << i << " probability: " << *(prediction + i) << std::endl;
	}
}


std::vector<float> ModelDescription::Predict(const float* input_data) {
	const int64_t dims[] = {1, 28 * 28};

	const size_t nbytes = 28 * 28 * sizeof(float);
	TF_Tensor* tensor = TF_AllocateTensor(TF_FLOAT, dims, 2, nbytes);

	memcpy(TF_TensorData(tensor), input_data, nbytes);

	TF_Output inputs[] = { input };
	TF_Tensor* input_values[] = { tensor };
	TF_Output outputs[] = { output };
	TF_Tensor* output_values[1] = {
		NULL
	};

	TF_SessionRun(
		session,
		NULL,
		/* Inputs */
		inputs,
		input_values,
		1,
		/* Outputs */
		outputs,
		output_values,
		1,
		/* Init operation */
		NULL,
		0,
		NULL,
		status
	);

	TF_DeleteTensor(tensor);

	if (!Okay()) throw std::exception("Could not run prediction session");

	if (TF_TensorByteSize(output_values[0]) != 10 * sizeof(float)) {
		throw std::exception("Predictions tensor do not match excepted nbytes");
	}

	//VisualizePrediction((float*)TF_TensorData(output_values[0]));

	std::vector<float> output = {};

	for (uint8_t i = 0; i < 10; i++) {
		output.push_back(*((float*)TF_TensorData(output_values[0]) + i));
	}

	TF_DeleteTensor(output_values[0]);

	return output;
}


static std::pair<TF_Tensor*, TF_Tensor*> CreateTrainTensors(const std::vector<float>& image_data, const uint8_t element) {
	const size_t input_data_size = image_data.size() * sizeof(float);

	const int64_t inputs_dims[] = { 1, image_data.size() };
	const int64_t targets_dims[] = { 1, 10 };

	TF_Tensor* inputs_tensor = TF_AllocateTensor(TF_FLOAT, inputs_dims, 2, input_data_size);
	TF_Tensor* targets_tensor = TF_AllocateTensor(TF_FLOAT, targets_dims, 2, 10 * sizeof(float));

	memcpy(TF_TensorData(inputs_tensor), image_data.data(), input_data_size);

	for (uint8_t i = 0; i < 10; i++) {
		if (element == i) {
			*(((float*)TF_TensorData(targets_tensor)) + i) = 1.0;
		}
		else {
			*(((float*)TF_TensorData(targets_tensor)) + i) = 0.0;
		}
	}

	//std::cout << "train target tensor data " << *(float*)(TF_TensorData(targets_tensor)) << std::endl;

	return { inputs_tensor, targets_tensor };
}


void ModelDescription::RunTrainStep(const std::vector<float>& image_data, const uint8_t element) {
	const auto& tensorsXY = CreateTrainTensors(image_data, element);

	const auto& x = tensorsXY.first;
	const auto& y = tensorsXY.second;

	TF_Output inputs[2] = {input, target};
	TF_Tensor* input_values[2] = {x, y};
	const TF_Operation* train_ops[1] = {train_op};
	TF_SessionRun(session, NULL, inputs, input_values, 2,
				/* No outputs */
				NULL, NULL, 0, train_ops, 1, NULL, status);
	TF_DeleteTensor(x);
	TF_DeleteTensor(y);

	if (!Okay()) throw std::exception("Could not run train step");
}


bool ModelDescription::Okay() const {
	if (TF_GetCode(status) != TF_OK) {
		std::cerr << "Got error status code! Message: " << TF_Message(status) << std::endl;
		return false;
	}

	return true;
}


TF_Buffer* ModelDescription::ReadFile(const std::string& filename) {
	/*
	std::ifstream fin(filename);

	std::string file_content = "";
	std::string line;

	while (std::getline(fin, line)) {
		file_content += line;
	}

	fin.close();
	*/


	std::string fileData(std::istreambuf_iterator<char>{std::ifstream(filename, std::ios::binary)}, {});

	return TF_NewBufferFromString(fileData.c_str(), fileData.size());
}

TF_Tensor* ModelDescription::ScalarStringTensor(const std::string& str, TF_Status* status) {
	size_t nbytes = str.size(); // TF_StringEncodedSize() ????

	TF_Tensor* tensor = TF_AllocateTensor(TF_STRING, NULL, 0, nbytes);

	void* data = TF_TensorData(tensor);

	memcpy(data, str.data(), nbytes);

	return tensor;
}
