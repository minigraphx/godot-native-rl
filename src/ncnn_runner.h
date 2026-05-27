#ifndef NCNN_RUNNER_H
#define NCNN_RUNNER_H

#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/string.hpp>

#include <memory>

namespace ncnn {
class Net;
}

namespace godot {

class NcnnRunner : public Node {
    GDCLASS(NcnnRunner, Node)

protected:
    static void _bind_methods();

public:
    NcnnRunner();
    ~NcnnRunner() override;

    bool load_model(const String &p_param_path, const String &p_bin_path);
    PackedFloat32Array run_inference(const PackedFloat32Array &p_input);

    void set_input_blob_name(const String &p_name);
    String get_input_blob_name() const;
    void set_output_blob_name(const String &p_name);
    String get_output_blob_name() const;

private:
    std::unique_ptr<ncnn::Net> net_;
    bool model_loaded_ = false;
    String input_blob_name_ = "input";
    String output_blob_name_ = "output";
};

} // namespace godot

#endif // NCNN_RUNNER_H
