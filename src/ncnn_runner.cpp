#include "ncnn_runner.h"

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

#include <mat.h>
#include <net.h>

#include <cstring>
#include <limits>

using namespace godot;

NcnnRunner::NcnnRunner() : net_(std::make_unique<ncnn::Net>()) {
}

NcnnRunner::~NcnnRunner() = default;

void NcnnRunner::_bind_methods() {
    ClassDB::bind_method(D_METHOD("load_model", "param_path", "bin_path"), &NcnnRunner::load_model);
    ClassDB::bind_method(D_METHOD("run_inference", "input"), &NcnnRunner::run_inference);
    ClassDB::bind_method(D_METHOD("run_inference_image", "image", "normalize_to_zero_one"), &NcnnRunner::run_inference_image, DEFVAL(true));
    ClassDB::bind_method(D_METHOD("run_discrete_action", "input"), &NcnnRunner::run_discrete_action);
    ClassDB::bind_method(D_METHOD("is_model_loaded"), &NcnnRunner::is_model_loaded);
    ClassDB::bind_method(D_METHOD("set_input_blob_name", "name"), &NcnnRunner::set_input_blob_name);
    ClassDB::bind_method(D_METHOD("get_input_blob_name"), &NcnnRunner::get_input_blob_name);
    ClassDB::bind_method(D_METHOD("set_output_blob_name", "name"), &NcnnRunner::set_output_blob_name);
    ClassDB::bind_method(D_METHOD("get_output_blob_name"), &NcnnRunner::get_output_blob_name);
    ClassDB::bind_method(D_METHOD("set_input_shape", "shape"), &NcnnRunner::set_input_shape);
    ClassDB::bind_method(D_METHOD("get_input_shape"), &NcnnRunner::get_input_shape);
    ClassDB::bind_method(D_METHOD("clear_input_shape"), &NcnnRunner::clear_input_shape);

    ADD_PROPERTY(PropertyInfo(Variant::STRING, "input_blob_name"), "set_input_blob_name", "get_input_blob_name");
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "output_blob_name"), "set_output_blob_name", "get_output_blob_name");
    ADD_PROPERTY(PropertyInfo(Variant::PACKED_INT32_ARRAY, "input_shape"), "set_input_shape", "get_input_shape");
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
    ncnn::Mat input_mat;
    if (!create_input_mat_from_array(p_input, input_mat)) {
        return PackedFloat32Array();
    }

    ncnn::Mat output_mat;
    if (!run_inference_internal(input_mat, output_mat)) {
        return PackedFloat32Array();
    }

    return output_mat_to_packed_float_array(output_mat);
}

PackedFloat32Array NcnnRunner::run_inference_image(const Ref<Image> &p_image, bool p_normalize_to_zero_one) {
    if (p_image.is_null()) {
        UtilityFunctions::push_error("NcnnRunner.run_inference_image: image is null.");
        return PackedFloat32Array();
    }

    if (p_image->is_empty()) {
        UtilityFunctions::push_error("NcnnRunner.run_inference_image: image is empty.");
        return PackedFloat32Array();
    }

    Ref<Image> working_image = p_image;
    if (working_image->get_format() != Image::FORMAT_RGB8) {
        working_image = working_image->duplicate();
        working_image->convert(Image::FORMAT_RGB8);
    }

    const PackedByteArray image_data = working_image->get_data();
    if (image_data.is_empty()) {
        UtilityFunctions::push_error("NcnnRunner.run_inference_image: failed to access image bytes.");
        return PackedFloat32Array();
    }

    ncnn::Mat input_mat = ncnn::Mat::from_pixels(
        image_data.ptr(),
        ncnn::Mat::PIXEL_RGB,
        working_image->get_width(),
        working_image->get_height()
    );

    if (p_normalize_to_zero_one) {
        const float normalize_values[3] = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
        input_mat.substract_mean_normalize(nullptr, normalize_values);
    }

    ncnn::Mat output_mat;
    if (!run_inference_internal(input_mat, output_mat)) {
        return PackedFloat32Array();
    }

    return output_mat_to_packed_float_array(output_mat);
}

int NcnnRunner::run_discrete_action(const PackedFloat32Array &p_input) {
    ncnn::Mat input_mat;
    if (!create_input_mat_from_array(p_input, input_mat)) {
        return -1;
    }

    ncnn::Mat output_mat;
    if (!run_inference_internal(input_mat, output_mat)) {
        return -1;
    }

    // Argmax over the logical outputs (w*h*d per channel). Avoid Mat::total() here: it counts the
    // SIMD-aligned cstep padding, so a w=3 head reports 4 and the padding slot could win the argmax.
    const PackedFloat32Array values = output_mat_to_packed_float_array(output_mat);
    if (values.is_empty()) {
        UtilityFunctions::push_error("NcnnRunner.run_discrete_action: output tensor is empty.");
        return -1;
    }

    int best_index = 0;
    float best_value = -std::numeric_limits<float>::infinity();

    for (int i = 0; i < values.size(); ++i) {
        const float current = values[i];
        if (current > best_value) {
            best_value = current;
            best_index = i;
        }
    }

    return best_index;
}

bool NcnnRunner::is_model_loaded() const {
    return model_loaded_ && static_cast<bool>(net_);
}

bool NcnnRunner::create_input_mat_from_array(const PackedFloat32Array &p_input, ncnn::Mat &r_input) const {
    if (p_input.is_empty()) {
        UtilityFunctions::push_error("NcnnRunner.run_inference: input array is empty.");
        return false;
    }

    if (input_shape_.is_empty()) {
        r_input = ncnn::Mat(static_cast<int>(p_input.size()));
        std::memcpy(r_input.data, p_input.ptr(), static_cast<size_t>(p_input.size()) * sizeof(float));
        return true;
    }

    if (input_shape_.size() < 1 || input_shape_.size() > 3) {
        UtilityFunctions::push_error("NcnnRunner.input_shape must have 1 to 3 dimensions.");
        return false;
    }

    int64_t expected_count = 1;
    for (int i = 0; i < input_shape_.size(); ++i) {
        const int32_t dim = input_shape_[i];
        if (dim <= 0) {
            UtilityFunctions::push_error("NcnnRunner.input_shape dimensions must all be > 0.");
            return false;
        }
        expected_count *= dim;
    }

    if (expected_count != static_cast<int64_t>(p_input.size())) {
        UtilityFunctions::push_error(
            "NcnnRunner.run_inference: input size does not match input_shape product. input_size=",
            p_input.size(),
            ", expected=",
            static_cast<int>(expected_count)
        );
        return false;
    }

    if (input_shape_.size() == 1) {
        r_input = ncnn::Mat(input_shape_[0]);
    } else if (input_shape_.size() == 2) {
        r_input = ncnn::Mat(input_shape_[0], input_shape_[1]);
    } else {
        r_input = ncnn::Mat(input_shape_[0], input_shape_[1], input_shape_[2]);
    }
    std::memcpy(r_input.data, p_input.ptr(), static_cast<size_t>(p_input.size()) * sizeof(float));
    return true;
}

bool NcnnRunner::run_inference_internal(const ncnn::Mat &p_input, ncnn::Mat &r_output) const {
    if (!model_loaded_ || !net_) {
        UtilityFunctions::push_error("NcnnRunner.run_inference: model is not loaded.");
        return false;
    }

    ncnn::Extractor extractor = net_->create_extractor();

    const CharString input_blob_utf8 = input_blob_name_.utf8();
    const int input_result = extractor.input(input_blob_utf8.get_data(), p_input);
    if (input_result != 0) {
        UtilityFunctions::push_error("NcnnRunner.run_inference: failed to bind input blob: ", input_blob_name_);
        return false;
    }

    const CharString output_blob_utf8 = output_blob_name_.utf8();
    const int output_result = extractor.extract(output_blob_utf8.get_data(), r_output);
    if (output_result != 0) {
        UtilityFunctions::push_error("NcnnRunner.run_inference: failed to extract output blob: ", output_blob_name_);
        return false;
    }

    return true;
}

PackedFloat32Array NcnnRunner::output_mat_to_packed_float_array(const ncnn::Mat &p_output) {
    // Use the logical element count (w*h*d per channel), NOT Mat::total(): ncnn aligns each
    // channel's cstep up to a SIMD boundary, so total() (== cstep*c) over-counts and would copy
    // garbage padding (e.g. a w=3, c=1 output reports total()==4). Copy channel-by-channel so the
    // packed array is exactly the real outputs, contiguous, regardless of internal stride.
    PackedFloat32Array output;
    const int per_channel = p_output.w * (p_output.h > 0 ? p_output.h : 1) * (p_output.d > 0 ? p_output.d : 1);
    const int channels = p_output.c > 0 ? p_output.c : 1;
    const int element_count = per_channel * channels;
    output.resize(element_count);
    if (element_count <= 0) {
        return output;
    }
    float *dst = output.ptrw();
    for (int q = 0; q < channels; ++q) {
        const float *src = p_output.channel(q);
        std::memcpy(dst + static_cast<size_t>(q) * per_channel, src, static_cast<size_t>(per_channel) * sizeof(float));
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

void NcnnRunner::set_input_shape(const PackedInt32Array &p_shape) {
    input_shape_ = p_shape;
}

PackedInt32Array NcnnRunner::get_input_shape() const {
    return input_shape_;
}

void NcnnRunner::clear_input_shape() {
    input_shape_ = PackedInt32Array();
}
