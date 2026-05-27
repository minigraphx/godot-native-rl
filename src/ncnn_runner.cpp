#include "ncnn_runner.h"

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

#include "net.h"

#include <cstring>

using namespace godot;

NcnnRunner::NcnnRunner() : net_(std::make_unique<ncnn::Net>()) {
}

NcnnRunner::~NcnnRunner() = default;

void NcnnRunner::_bind_methods() {
    ClassDB::bind_method(D_METHOD("load_model", "param_path", "bin_path"), &NcnnRunner::load_model);
    ClassDB::bind_method(D_METHOD("run_inference", "input"), &NcnnRunner::run_inference);
    ClassDB::bind_method(D_METHOD("set_input_blob_name", "name"), &NcnnRunner::set_input_blob_name);
    ClassDB::bind_method(D_METHOD("get_input_blob_name"), &NcnnRunner::get_input_blob_name);
    ClassDB::bind_method(D_METHOD("set_output_blob_name", "name"), &NcnnRunner::set_output_blob_name);
    ClassDB::bind_method(D_METHOD("get_output_blob_name"), &NcnnRunner::get_output_blob_name);

    ADD_PROPERTY(PropertyInfo(Variant::STRING, "input_blob_name"), "set_input_blob_name", "get_input_blob_name");
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "output_blob_name"), "set_output_blob_name", "get_output_blob_name");
}

bool NcnnRunner::load_model(const String &p_param_path, const String &p_bin_path) {
    if (p_param_path.is_empty() || p_bin_path.is_empty()) {
        UtilityFunctions::push_error("NcnnRunner.load_model: param_path and bin_path must be non-empty.");
        return false;
    }

    net_ = std::make_unique<ncnn::Net>();
    model_loaded_ = false;

    const CharString param_utf8 = p_param_path.utf8();
    const CharString bin_utf8 = p_bin_path.utf8();

    const int param_result = net_->load_param(param_utf8.get_data());
    if (param_result != 0) {
        UtilityFunctions::push_error("NcnnRunner.load_model: failed to load param file: ", p_param_path);
        net_.reset();
        return false;
    }

    const int bin_result = net_->load_model(bin_utf8.get_data());
    if (bin_result != 0) {
        UtilityFunctions::push_error("NcnnRunner.load_model: failed to load bin file: ", p_bin_path);
        net_.reset();
        return false;
    }

    model_loaded_ = true;
    return true;
}

PackedFloat32Array NcnnRunner::run_inference(const PackedFloat32Array &p_input) {
    PackedFloat32Array output;

    if (!model_loaded_ || !net_) {
        UtilityFunctions::push_error("NcnnRunner.run_inference: model is not loaded.");
        return output;
    }

    if (p_input.is_empty()) {
        UtilityFunctions::push_error("NcnnRunner.run_inference: input array is empty.");
        return output;
    }

    ncnn::Mat input_mat(static_cast<int>(p_input.size()));
    std::memcpy(input_mat.data, p_input.ptr(), static_cast<size_t>(p_input.size()) * sizeof(float));

    ncnn::Extractor extractor = net_->create_extractor();

    const CharString input_blob_utf8 = input_blob_name_.utf8();
    const int input_result = extractor.input(input_blob_utf8.get_data(), input_mat);
    if (input_result != 0) {
        UtilityFunctions::push_error("NcnnRunner.run_inference: failed to bind input blob: ", input_blob_name_);
        return output;
    }

    ncnn::Mat output_mat;
    const CharString output_blob_utf8 = output_blob_name_.utf8();
    const int output_result = extractor.extract(output_blob_utf8.get_data(), output_mat);
    if (output_result != 0) {
        UtilityFunctions::push_error("NcnnRunner.run_inference: failed to extract output blob: ", output_blob_name_);
        return output;
    }

    const size_t output_count = output_mat.total();
    output.resize(static_cast<int>(output_count));
    if (output_count > 0) {
        std::memcpy(output.ptrw(), output_mat.data, output_count * sizeof(float));
    }

    return output;
}

void NcnnRunner::set_input_blob_name(const String &p_name) {
    input_blob_name_ = p_name;
}

String NcnnRunner::get_input_blob_name() const {
    return input_blob_name_;
}

void NcnnRunner::set_output_blob_name(const String &p_name) {
    output_blob_name_ = p_name;
}

String NcnnRunner::get_output_blob_name() const {
    return output_blob_name_;
}
