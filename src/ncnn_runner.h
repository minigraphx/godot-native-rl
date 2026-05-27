#ifndef NCNN_RUNNER_H
#define NCNN_RUNNER_H

#include <godot_cpp/classes/node.hpp>

namespace godot {

class NcnnRunner : public Node {
    GDCLASS(NcnnRunner, Node)

protected:
    static void _bind_methods();

public:
    NcnnRunner() = default;
    ~NcnnRunner() override = default;
};

} // namespace godot

#endif // NCNN_RUNNER_H
