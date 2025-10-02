#pragma once

#include <string>

#include <tensorflow/c/c_api.h>


struct ModelDescription {
	ModelDescription(const std::string& graph_def_filename);
	~ModelDescription();

	static enum class CheckpointType {
		Save,
		Restore
	};

	void Init();
	std::vector<float> Predict(const float* input_data);
	void RunTrainStep(const std::vector<float>& image_data, const uint8_t element);
	void Checkpoint(const std::string& checkpoint_prefix, const CheckpointType& type);
	bool Okay() const;

	static TF_Buffer* ReadFile(const std::string& filename);
	static TF_Tensor* ScalarStringTensor(const std::string& data, TF_Status* status);
	static void StringDeallocator(void* data, size_t len, void* arg);

	TF_Graph* graph;
	TF_Session* session;
	TF_Status* status;

	TF_Output input, target, output;

	TF_Operation *init_op, *train_op, *save_op, *restore_op;
	TF_Output checkpoint_file;
};


