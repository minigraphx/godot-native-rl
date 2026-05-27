#!/usr/bin/env python
import os

from SCons.Script import Default, SConscript

env = SConscript("godot-cpp/SConstruct")
env.add_source_files(env.modules_sources, "src/*.cpp")

library = env.SharedLibrary(
    target=os.path.join("bin", "libncnn_runner{}{}".format(env["suffix"], env["SHLIBSUFFIX"])),
    source=env.modules_sources,
)

Default(library)
