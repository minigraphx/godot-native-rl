#!/usr/bin/env python
import os

from SCons.Script import Default, Exit, File, Glob, SConscript

env = SConscript("godot-cpp/SConstruct")
sources = Glob("src/*.cpp")
env.Append(CPPPATH=["src"])

project_dir = os.path.abspath(".")
ncnn_root = os.path.join(project_dir, "thirdparty", "ncnn")
if not os.path.isdir(ncnn_root):
    print("Error: ncnn root was not found at {}".format(ncnn_root))
    Exit(1)

ncnn_include_candidates = [
    os.path.join(ncnn_root, "include"),
    os.path.join(ncnn_root, "src"),
    os.path.join(ncnn_root, "build", "install", "include"),
    os.path.join(ncnn_root, "build", "install", "include", "ncnn"),
]
ncnn_include_paths = [path for path in ncnn_include_candidates if os.path.isdir(path)]
if not ncnn_include_paths:
    print("Error: no ncnn include directories found under {}".format(ncnn_root))
    Exit(1)
env.Append(CPPPATH=ncnn_include_paths)

if env["platform"] == "windows":
    ncnn_static_candidates = [
        os.path.join(ncnn_root, "build", "install", "lib", "ncnn.lib"),
        os.path.join(ncnn_root, "build", "install", "lib", "libncnn.a"),
    ]
else:
    ncnn_static_candidates = [
        os.path.join(ncnn_root, "build", "install", "lib", "libncnn.a"),
        os.path.join(ncnn_root, "build", "src", "libncnn.a"),
    ]

ncnn_static_lib = next((path for path in ncnn_static_candidates if os.path.isfile(path)), None)
if ncnn_static_lib is None:
    print("Error: could not find static ncnn library. Looked in:")
    for path in ncnn_static_candidates:
        print("  - {}".format(path))
    Exit(1)

# Link ncnn statically into the extension.
env.Append(LIBS=[File(ncnn_static_lib)])

library = env.SharedLibrary(
    target=os.path.join("bin", "libncnn_runner{}{}".format(env["suffix"], env["SHLIBSUFFIX"])),
    source=sources,
)

Default(library)
