#ifndef NCNN_RUNNER_H
#define NCNN_RUNNER_H

#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/classes/image.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/packed_int32_array.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>

#include <memory>
#include <atomic>
#include <thread>

namespace ncnn {
class Net;
class Mat;
}

namespace godot {

// Godot-facing wrapper around a statically linked ncnn::Net: loads a converted
// *.ncnn.{param,bin} model (from file paths or in-memory buffers) and runs CPU
// inference — flat float vectors, image inputs, multi-input/multi-output (e.g.
// recurrent hidden state), plus an argmax convenience for discrete policies.
// Input/output blob names and an optional explicit input shape are configurable
// to match whatever the converter emitted.
class NcnnRunner : public Node {
    GDCLASS(NcnnRunner, Node)

protected:
    static void _bind_methods();

public:
    NcnnRunner();
    ~NcnnRunner() override;

    bool load_model(const String &p_param_path, const String &p_bin_path);
    bool load_model_from_buffers(const PackedByteArray &p_param, const PackedByteArray &p_bin);
    PackedFloat32Array run_inference(const PackedFloat32Array &p_input);
    PackedFloat32Array run_inference_image(const Ref<Image> &p_image, bool p_normalize_to_zero_one = true);
    Dictionary run_inference_multi(const Array &p_inputs, const PackedStringArray &p_output_names);
    Array run_inference_batch(const Array &p_inputs, int p_num_threads = -1);
    int run_discrete_action(const PackedFloat32Array &p_input);
    bool is_model_loaded() const;

    // Non-blocking forward pass on a worker thread (#19): runs run_inference off the main
    // thread and emits `inference_completed(output)` on the main thread when done. Returns
    // true if the request was accepted (model loaded, input valid, no request in flight).
    // One request at a time — re-request after the signal (check is_inference_running()).
    bool run_inference_async(const PackedFloat32Array &p_input);
    bool is_inference_running() const;

    void set_input_blob_name(const String &p_name);
    String get_input_blob_name() const;
    void set_output_blob_name(const String &p_name);
    String get_output_blob_name() const;
    void set_input_shape(const PackedInt32Array &p_shape);
    PackedInt32Array get_input_shape() const;
    void clear_input_shape();

private:
    bool create_input_mat_from_array(const PackedFloat32Array &p_input, ncnn::Mat &r_input) const;
    bool build_mat_from_shape(const PackedFloat32Array &p_data, const PackedInt32Array &p_shape, ncnn::Mat &r_mat) const;
    bool run_inference_internal(const ncnn::Mat &p_input, ncnn::Mat &r_output) const;
    static PackedFloat32Array output_mat_to_packed_float_array(const ncnn::Mat &p_output);
    // Runs on the main thread via call_deferred from the async worker: clears the in-flight
    // flag, joins the finished worker, and emits inference_completed.
    void async_finish(const PackedFloat32Array &p_output);

    std::unique_ptr<ncnn::Net> net_;
    bool model_loaded_ = false;
    String input_blob_name_ = "input";
    String output_blob_name_ = "output";
    PackedInt32Array input_shape_;
    std::thread async_worker_;
    std::atomic<bool> inference_running_{false};
};

} // namespace godot

#endif // NCNN_RUNNER_H
