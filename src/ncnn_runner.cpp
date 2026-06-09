#include "ncnn_runner.h"

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

#include <mat.h>
#include <net.h>
#include <datareader.h>

#include <cstring>
#include <limits>
#include <vector>
#include <thread>
#include <algorithm>

using namespace godot;

NcnnRunner::NcnnRunner() : net_(std::make_unique<ncnn::Net>()) {
}

NcnnRunner::~NcnnRunner() = default;

void NcnnRunner::_bind_methods() {
    ClassDB::bind_method(D_METHOD("load_model", "param_path", "bin_path"), &NcnnRunner::load_model);
    ClassDB::bind_method(D_METHOD("load_model_from_buffers", "param", "bin"), &NcnnRunner::load_model_from_buffers);
    ClassDB::bind_method(D_METHOD("run_inference", "input"), &NcnnRunner::run_inference);
    ClassDB::bind_method(D_METHOD("run_inference_image", "image", "normalize_to_zero_one"), &NcnnRunner::run_inference_image, DEFVAL(true));
    ClassDB::bind_method(D_METHOD("run_inference_multi", "inputs", "output_names"), &NcnnRunner::run_inference_multi);
    ClassDB::bind_method(D_METHOD("run_inference_batch", "inputs", "num_threads"), &NcnnRunner::run_inference_batch, DEFVAL(-1));
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

bool NcnnRunner::load_model_from_buffers(const PackedByteArray &p_param, const PackedByteArray &p_bin) {
    if (p_param.is_empty() || p_bin.is_empty()) {
        UtilityFunctions::push_error("NcnnRunner.load_model_from_buffers: param and bin buffers must be non-empty.");
        return false;
    }

    net_ = std::make_unique<ncnn::Net>();
    model_loaded_ = false;

    // ncnn's load_param_mem() needs a NUL-terminated C string of the text .param.
    std::vector<char> param_text(p_param.size() + 1);
    std::memcpy(param_text.data(), p_param.ptr(), p_param.size());
    param_text[p_param.size()] = '\0';

    const int param_result = net_->load_param_mem(param_text.data());
    if (param_result != 0) {
        UtilityFunctions::push_error("NcnnRunner.load_model_from_buffers: failed to parse param buffer.");
        net_.reset();
        return false;
    }

    // The .bin weights load from an advancing memory cursor via DataReaderFromMemory.
    // DataReaderFromMemory carries no length bound, so the .bin/.param are trusted
    // app-bundled assets (same trust model as the path-based load_model).
    const unsigned char *bin_cursor = reinterpret_cast<const unsigned char *>(p_bin.ptr());
    ncnn::DataReaderFromMemory bin_reader(bin_cursor);
    const int bin_result = net_->load_model(bin_reader);
    if (bin_result != 0) {
        UtilityFunctions::push_error("NcnnRunner.load_model_from_buffers: failed to load bin buffer.");
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

Dictionary NcnnRunner::run_inference_multi(const Array &p_inputs, const PackedStringArray &p_output_names) {
    Dictionary result;
    if (!model_loaded_ || !net_) {
        UtilityFunctions::push_error("NcnnRunner.run_inference_multi: model is not loaded.");
        return result;
    }
    if (p_inputs.is_empty() || p_output_names.is_empty()) {
        UtilityFunctions::push_error("NcnnRunner.run_inference_multi: inputs and output_names must be non-empty.");
        return result;
    }

    ncnn::Extractor extractor = net_->create_extractor();

    for (int i = 0; i < p_inputs.size(); ++i) {
        const Dictionary spec = p_inputs[i];
        if (!spec.has("name") || !spec.has("data") || !spec.has("shape")) {
            UtilityFunctions::push_error("NcnnRunner.run_inference_multi: each input needs name/data/shape.");
            return Dictionary();
        }
        const String name = spec["name"];
        const PackedFloat32Array data = spec["data"];
        const PackedInt32Array shape = spec["shape"];
        ncnn::Mat mat;
        if (!build_mat_from_shape(data, shape, mat)) {
            return Dictionary();
        }
        const CharString name_utf8 = name.utf8();
        if (extractor.input(name_utf8.get_data(), mat) != 0) {
            UtilityFunctions::push_error("NcnnRunner.run_inference_multi: failed to bind input blob: ", name);
            return Dictionary();
        }
    }

    for (int i = 0; i < p_output_names.size(); ++i) {
        const String name = p_output_names[i];
        const CharString name_utf8 = name.utf8();
        ncnn::Mat out;
        if (extractor.extract(name_utf8.get_data(), out) != 0) {
            UtilityFunctions::push_error("NcnnRunner.run_inference_multi: failed to extract output blob: ", name);
            return Dictionary();
        }
        result[name] = output_mat_to_packed_float_array(out);
    }

    return result;
}

Array NcnnRunner::run_inference_batch(const Array &p_inputs, int p_num_threads) {
    Array result;
    if (!model_loaded_ || !net_) {
        UtilityFunctions::push_error("NcnnRunner.run_inference_batch: model is not loaded.");
        return result;
    }
    const int n = p_inputs.size();
    if (n == 0) {
        return result; // empty crowd is not an error: empty in -> empty out.
    }

    // Build every input Mat up front on the calling thread. create_input_mat_from_array
    // push_errors on a bad input (size mismatch when input_shape is set), so all logging stays
    // on the main thread; worker threads below never call into Godot's error reporter.
    std::vector<ncnn::Mat> input_mats(static_cast<size_t>(n));
    std::vector<bool> input_ok(static_cast<size_t>(n), false);
    for (int i = 0; i < n; ++i) {
        const PackedFloat32Array vec = p_inputs[i];
        if (create_input_mat_from_array(vec, input_mats[i])) {
            input_ok[i] = true;
        }
    }

    std::vector<PackedFloat32Array> outputs(static_cast<size_t>(n));

    // Worker count: clamp(requested>0 ? requested : hardware_concurrency, 1, n). WASM is
    // single-threaded (see docs/dev/building.md) -> always serial.
    unsigned int hw = std::thread::hardware_concurrency();
    if (hw == 0) {
        hw = 1;
    }
    int workers = (p_num_threads > 0) ? p_num_threads : static_cast<int>(hw);
    workers = std::min(workers, n);
    if (workers < 1) {
        workers = 1;
    }
#ifdef __EMSCRIPTEN__
    workers = 1;
#endif

    // ncnn::Net is safe for concurrent extractors; each worker owns its Extractor and writes only
    // its own output slots, so there is no shared mutation. The Net's opt.num_threads is pinned to 1
    // for this call (see below) so each Extractor runs single-threaded — no nesting with ncnn's
    // intra-layer OpenMP. Quiet on failure (no push_error off-thread): a failed slot is left empty
    // and reported once after join.
    // Hoist blob-name conversions: immutable for this call, so convert once on the calling thread.
    const CharString input_blob_utf8 = input_blob_name_.utf8();
    const CharString output_blob_utf8 = output_blob_name_.utf8();
    auto run_slice = [&](int begin, int end) {
        for (int i = begin; i < end; ++i) {
            if (!input_ok[i]) {
                continue;
            }
            ncnn::Extractor ex = net_->create_extractor();
            if (ex.input(input_blob_utf8.get_data(), input_mats[i]) != 0) {
                continue;
            }
            ncnn::Mat out;
            if (ex.extract(output_blob_utf8.get_data(), out) != 0) {
                continue;
            }
            outputs[i] = output_mat_to_packed_float_array(out);
        }
    };

    // ncnn::Extractor has no per-extractor thread control in this version, so pin the Net's thread
    // count to 1 for the duration of the batch: create_extractor() snapshots net_->opt at creation,
    // so every worker's Extractor runs single-threaded and our std::thread fan-out is the only
    // parallelism (no nested OpenMP oversubscription). Restored afterward so single-inference paths
    // (run_inference / _image / _multi) keep ncnn's default intra-op threading.
    const int prev_num_threads = net_->opt.num_threads;
    net_->opt.num_threads = 1;

    if (workers <= 1) {
        run_slice(0, n);
    } else {
        std::vector<std::thread> threads;
        threads.reserve(static_cast<size_t>(workers));
        const int base = n / workers;
        const int rem = n % workers;
        int start = 0;
        for (int w = 0; w < workers; ++w) {
            const int count = base + (w < rem ? 1 : 0);
            threads.emplace_back(run_slice, start, start + count);
            start += count;
        }
        for (std::thread &t : threads) {
            t.join();
        }
    }

    net_->opt.num_threads = prev_num_threads;

    int failures = 0;
    result.resize(n);
    for (int i = 0; i < n; ++i) {
        if (input_ok[i] && outputs[i].is_empty()) {
            // Input built fine but inference/extract failed (a malformed input is already
            // reported by create_input_mat_from_array on the main thread above).
            ++failures;
        }
        result[i] = outputs[i];
    }
    if (failures > 0) {
        UtilityFunctions::push_error("NcnnRunner.run_inference_batch: ", failures,
            " of ", n, " agent(s) failed inference (empty output slot).");
    }
    return result;
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

    return build_mat_from_shape(p_input, input_shape_, r_input);
}

bool NcnnRunner::build_mat_from_shape(const PackedFloat32Array &p_data, const PackedInt32Array &p_shape, ncnn::Mat &r_mat) const {
    if (p_data.is_empty()) {
        UtilityFunctions::push_error("NcnnRunner: input array is empty.");
        return false;
    }
    if (p_shape.size() < 1 || p_shape.size() > 3) {
        UtilityFunctions::push_error("NcnnRunner: input shape must have 1 to 3 dimensions.");
        return false;
    }
    int64_t expected_count = 1;
    for (int i = 0; i < p_shape.size(); ++i) {
        const int32_t dim = p_shape[i];
        if (dim <= 0) {
            UtilityFunctions::push_error("NcnnRunner: input shape dimensions must all be > 0.");
            return false;
        }
        expected_count *= dim;
    }
    if (expected_count != static_cast<int64_t>(p_data.size())) {
        UtilityFunctions::push_error("NcnnRunner: input size does not match shape product. input_size=",
            p_data.size(), ", expected=", static_cast<int>(expected_count));
        return false;
    }
    if (p_shape.size() == 1) {
        r_mat = ncnn::Mat(p_shape[0]);
    } else if (p_shape.size() == 2) {
        r_mat = ncnn::Mat(p_shape[0], p_shape[1]);
    } else {
        r_mat = ncnn::Mat(p_shape[0], p_shape[1], p_shape[2]);
    }
    std::memcpy(r_mat.data, p_data.ptr(), static_cast<size_t>(p_data.size()) * sizeof(float));
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
