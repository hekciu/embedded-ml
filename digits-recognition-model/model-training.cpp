#include <iostream>
#include <exception>
#include <fstream>

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
	target.oper = TF_GraphOperationByName(graph, "target");
	target.index = 0;
	output.oper = TF_GraphOperationByName(graph, "output");
	output.index = 0;

	init_op = TF_GraphOperationByName(graph, "init");
	train_op = TF_GraphOperationByName(graph, "train");
	save_op = TF_GraphOperationByName(graph, "save/control_dependency");
	restore_op = TF_GraphOperationByName(graph, "save/restore_all");

	checkpoint_file.oper = TF_GraphOperationByName(graph, "save/Const");
	checkpoint_file.index = 0;
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


void ModelDescription::Predict(const float* batch, const size_t batch_size) {
	const int64_t dims[] = { batch_size, 1, 1 };

	const size_t nbytes = batch_size * sizeof(float);
	TF_Tensor* tensor = TF_AllocateTensor(TF_FLOAT, dims, 3, nbytes);

	memcpy(TF_TensorData(tensor), batch, nbytes);

	TF_Output inputs[] = { input };
	TF_Tensor* input_values[] = { tensor };
	TF_Output outputs[] = { output };
	TF_Tensor* output_values[] = { NULL };

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

	if (TF_TensorByteSize(output_values[0]) != nbytes) {
		throw std::exception("Predictions tensor do not match excepted nbytes");
	}

	float* predictions = (float*)malloc(nbytes);

	if (predictions == NULL) throw std::exception("Could not allocate memory for predictons");

	memcpy(predictions, TF_TensorData(output_values[0]), nbytes);
	TF_DeleteTensor(output_values[0]);

	std::cout << "predictions: " << '\n';
	for (size_t i = 0; i < batch_size; i++) {
		std::cout << "x: " << batch[i] << " predicted: " << predictions[i] << std::endl;
	}

	free(predictions);
}


static void NextBatchForTraining(TF_Tensor** inputs_tensor,
	TF_Tensor** targets_tensor) {
#define BATCH_SIZE 10
	float inputs[BATCH_SIZE] = { 0 };
	float targets[BATCH_SIZE] = { 0 };
	for (int i = 0; i < BATCH_SIZE; ++i) {
		inputs[i] = (float)rand() / (float)RAND_MAX;
		targets[i] = 3.0 * inputs[i] + 2.0;
	}
	const int64_t dims[] = { BATCH_SIZE, 1, 1 };
	size_t nbytes = BATCH_SIZE * sizeof(float);
	*inputs_tensor = TF_AllocateTensor(TF_FLOAT, dims, 3, nbytes);
	*targets_tensor = TF_AllocateTensor(TF_FLOAT, dims, 3, nbytes);
	memcpy(TF_TensorData(*inputs_tensor), inputs, nbytes);
	memcpy(TF_TensorData(*targets_tensor), targets, nbytes);
#undef BATCH_SIZE
};


void ModelDescription::RunTrainStep() {
  TF_Tensor *x, *y;
  NextBatchForTraining(&x, &y);
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
	std::ifstream fin(filename);

	std::string file_content = "";
	std::string line;

	while (std::getline(fin, line)) {
		file_content += line;
	}

	fin.close();

	return TF_NewBufferFromString(file_content.c_str(), file_content.size());
}

TF_Tensor* ModelDescription::ScalarStringTensor(const std::string& str, TF_Status* status) {
	size_t nbytes = str.size(); // TF_StringEncodedSize() ????

	TF_Tensor* tensor = TF_AllocateTensor(TF_STRING, NULL, 0, nbytes);

	void* data = TF_TensorData(tensor);

	memcpy(data, str.data(), nbytes);

	return tensor;
}
