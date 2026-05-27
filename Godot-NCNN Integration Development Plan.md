# ***Master Development Document: Godot Native RL (Training \+ NCNN Inference)***

## ***1\. Project Overview***

***Goal:** Create an all-in-one plug-and-play GDExtension for Godot 4.6+ that handles both RL agent training (via TCP sync) and native AI inference (via Tencent's `ncnn` framework). **Primary Use Case:** Completely replacing the `godot-rl-agents` engine plugin. Developers can train agents using a bundled GDScript TCP client (communicating with external Python frameworks), and then deploy those trained agents locally across Desktop, Mobile, Web, and Consoles using pure C++ `ncnn` inference. No C\# or .NET required. **Format:** This is a living Markdown document designed to be read by both Human developers and AI coding assistants.*

## ***2\. Context for AI Assistants (System Prompt Instructions)***

*If you are an AI assistant reading this file, adhere strictly to the following constraints during code generation:*

* ***Engine Version:** Target **Godot 4.6+**. Leverage the new optimizations (like default Jolt physics performance). Use modern `GDREGISTER_CLASS(ClassName)` registration macros. Do not hallucinate deprecated Godot 4.0 bindings.*  
* ***C++ Standard:** C++17 minimum.*  
* ***Memory Management & Tensors:** Be explicit about pointer ownership when converting `godot::PackedFloat32Array` to `ncnn::Mat`. Handle multi-dimensional reshaping (especially for Camera3D visual observations).*  
* ***Build System:** Use SCons. `ncnn` will be compiled statically alongside the GDExtension to avoid distributing separate `.dll`/`.so` files.*  
* ***Role:** Your goal is to write clean, self-contained, heavily commented C++ code. Produce one milestone at a time. Do not jump ahead unless explicitly asked.*

### ***2.1 Context Injection Strategy (For the Human Developer)***

*To ensure AI assistants do not hallucinate older API versions, inject the following files into the AI's context window alongside your prompt for specific tasks:*

* ***For GDExtension Boilerplate:** Upload `thirdparty/godot-cpp/test/src/example.cpp`. This file contains the exact, up-to-date Godot 4.6 macro syntax for registering classes and methods.*  
* ***For Array Math (M2.3):** Upload `thirdparty/godot-cpp/include/godot_cpp/variant/packed_float32_array.hpp`.*  
* ***For ncnn Integration:** Upload `thirdparty/ncnn/src/net.h` and `mat.h`.*  
* ***For GDScript TCP (M3.3):** Copy/paste the Godot 4.6 online documentation for `StreamPeerTCP`.*

## ***3\. Architecture & Tech Stack***

* ***Engine API:** `godot-cpp` (GDExtension, 4.6 API).*  
* ***Inference Engine (C++):** `ncnn` (Minimalist, dependency-free, cross-platform inference).*  
* ***Training Bridge (GDScript):** StreamPeerTCP (Sending batched state/action arrays to Python).*  
* ***Model Format:** `.param` and `.bin` (Converted from PyTorch `.pt` via the `pnnx` tool).*  
* ***CI/CD:** GitHub Actions (Matrix builds for Windows, Linux, macOS, Web).*

### ***3.1 Repository Structure***

*godot-native-rl/*

*├── .github/workflows/build.yml \# CI/CD Cross-compilation matrix*

*├── thirdparty/*

*│   ├── ncnn/                   \# git submodule*

*│   └── godot-cpp/              \# git submodule (Godot 4.6 branch)*

*├── src/*

*│   ├── register\_types.h / .cpp \# GDExtension entry point*

*│   ├── ncnn\_runner.h / .cpp    \# Core C++ wrapper handling ncnn::Net*

*│   └── ncnn\_tensor\_utils.cpp   \# Helpers for 1D/2D/3D (Vision) Mat conversions*

*├── godot\_project/*

*│   └── addons/*

*│       └── godot\_native\_rl/*

*│           ├── bin/            \# CI/CD injects compiled binaries here*

*│           ├── godot\_native\_rl.gdextension*

*│           ├── ncnn\_agent.gd   \# High-level C++ wrapper for INFERENCE*

*│           ├── sync\_node.gd    \# Coordinator for Multi-Agent Batching*

*│           └── tcp\_client.gd   \# GDScript socket client for TRAINING*

*└── SConstruct                  \# Build script linking godot-cpp and ncnn*

## ***4\. Development Milestones***

### ***Phase 1: Foundation & Build System***

* *$$$$*  
   ***M1.1 Repository Init:** Initialize Git repo, add `godot-cpp` (target: 4.6 branch) and `ncnn` as git submodules.*  
* *$$$$*  
   ***M1.2 Boilerplate GDExtension:** Create `register_types.cpp` and an empty `NcnnRunner` node class inheriting from `godot::Node`.*  
* *$$$$*  
   ***M1.3 SConstruct Creation:** Write the SCons script to compile the empty GDExtension.*  
* *$$$$*  
   ***M1.4 Test Build:** Successfully compile the empty node and verify it appears in the Godot Editor.*

### ***Phase 2: Core NCNN Integration & Tensor Math (C++)***

* *$$$$*  
   ***M2.1 Submodule Linking:** Update `SConstruct` to statically link and compile the `ncnn` source files directly into the Godot shared library.*  
* *$$$$*  
   ***M2.2 Model Loading:** Implement `NcnnRunner::load_model(String param_path, String bin_path)` using `ncnn::Net`. Ensure file paths are correctly resolved from Godot's `res://` to absolute OS paths.*  
* *$$$$*  
   ***M2.3 Observation Marshalling:** Implement the conversion logic from `godot::PackedFloat32Array` (1D Vectors) and `godot::Image` (3D Camera Vision) to `ncnn::Mat`.*  
* *$$$$*  
   ***M2.4 Action Space Parsing:** Implement `NcnnRunner::run_inference()`. Extract the forward pass output. Add logic to handle **Continuous** spaces (returning raw floats) and **Discrete** spaces (applying an `argmax` function over the output probabilities).*

### ***Phase 3: The All-In-One API (Inference & Training Bridges)***

* *$$$$*  
   ***M3.1 Inference Wrapper:** Create `ncnn_agent.gd` inside the `addons/` folder. This script instantiates the C++ object internally, exposing simple `get_action(obs: Array) -> Array` methods.*  
* *$$$$*  
   ***M3.2 Multi-Agent Sync Coordinator:** Create `sync_node.gd`. This node automatically detects all `ncnn_agent` nodes in the SceneTree, batches their observations into a single payload, and routes the batched actions back to each respective agent.*  
* *$$$$*  
   ***M3.3 TCP Training Bridge:** Port the `StreamPeerTCP` logic into `tcp_client.gd` to send the batched payload to Python over sockets.*  
* *$$$$*  
   ***M3.4 Unified Controller & Heuristic Override:** Update the `ncnn_agent.gd` with an `export var mode: String = "Training" | "Inference" | "Heuristic"`.*  
  * *Training: Waits for TCP actions.*  
  * *Inference: Calls the C++ ncnn runner.*  
  * *Heuristic: Listens to human input (e.g., `Input.get_vector()`) for debugging/expert demonstrations.*

### ***Phase 4: Automation & Distribution***

* *$$$$*  
   ***M4.1 GitHub Actions \- Desktop:** Create YAML workflow to compile `.dll` (Win), `.so` (Linux), and `.dylib` (Mac Universal) on every push against the Godot 4.6 bindings.*  
* *$$$$*  
   ***M4.2 GitHub Actions \- Web/Mobile (Optional but recommended):** Add Wasm and Android compilation to the matrix.*  
* *$$$$*  
   ***M4.3 Release Packaging:** Automate the zipping of the `addons/godot_native_rl/` folder containing the fresh binaries.*

### ***Phase 5: Documentation & Python Companion***

* *$$$$*  
   ***M5.1 Python Compatibility:** Ensure `tcp_client.gd` payload structures precisely match the expected inputs of `godot-rl-agents`'s StableBaselines3/CleanRL wrappers so users can use existing Python scripts out of the box.*  
* *$$$$*  
   ***M5.2 Model Conversion Docs:** Write a short tutorial for users on how to use `pnnx` to convert their StableBaselines3/CleanRL PyTorch models to ncnn format.*

## ***5\. Maintenance & Longevity Strategy***

*To keep the project alive, updated, and welcoming to the open-source community, we will implement the following strategies, drawing on motivational and social psychology principles to reduce contributor friction:*

### ***5.1 Automated Upkeep (Reducing Cognitive Load)***

* ***Dependabot for Submodules:** Configure GitHub Dependabot to automatically check for updates to the `godot-cpp` and `ncnn` submodules. This ensures the extension doesn't rot when Godot releases a new minor version (e.g., 4.7).*  
* ***Continuous Integration as a Shield:** The GitHub Actions matrix guarantees that a PR won't be merged if it breaks compilation on an obscure OS. This psychological safety net encourages developers to submit code.*

### ***5.2 Community Motivation & Open Source Psychology***

* ***The "Time-to-Dopamine" Metric:** The core metric for this addon's success is how fast a user can get an agent running. By offering an all-in-one Training & Inference package via the Asset Library, the user never has to download a separate Godot plugin or touch C\#.*  
* ***Clear "Good First Issues":** Create a label system in GitHub issues. For example, creating GDScript helper functions for parsing complex hybrid action spaces is an excellent entry point for Godot users who know GDScript but are intimidated by C++.*

### ***5.3 Versioning Strategy***

* *Tie the major/minor versions of this addon strictly to the Godot Engine version it supports (e.g., `v4.6.x` for Godot 4.6). GDExtension is notoriously sensitive to version mismatches.*

*Document Last Updated: May 2026*

