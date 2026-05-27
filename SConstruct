#!/usr/bin/env python
import os

from SCons.Script import ARGUMENTS, Default, SConscript

env = SConscript("godot-cpp/SConstruct")
env.add_source_files(env.modules_sources, "src/*.cpp")

ncnn_include = ARGUMENTS.get("ncnn_include", os.environ.get("NCNN_INCLUDE_DIR", ""))
ncnn_lib = ARGUMENTS.get("ncnn_lib", os.environ.get("NCNN_LIB_DIR", ""))

if ncnn_include:
    env.Append(CPPPATH=[ncnn_include])
if ncnn_lib:
    env.Append(LIBPATH=[ncnn_lib])

env.Append(LIBS=["ncnn"])

library = env.SharedLibrary(
    target=os.path.join("bin", "libncnn_runner{}{}".format(env["suffix"], env["SHLIBSUFFIX"])),
    source=env.modules_sources,
)

Default(library)
