#!/usr/bin/env python
import os
import subprocess

from SCons.Script import ARGUMENTS, Default, Exit, File, Glob, SConscript

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

if env["platform"] == "macos":
    requested_arch = str(ARGUMENTS.get("arch", env.get("arch", "")))
    try:
        lipo_info = subprocess.check_output(["lipo", "-info", ncnn_static_lib], text=True).strip()
    except Exception:
        lipo_info = ""

    ncnn_arches = set()
    if " are: " in lipo_info:
        ncnn_arches.update(lipo_info.split(" are: ", 1)[1].split())
    elif " architecture: " in lipo_info:
        ncnn_arches.add(lipo_info.split(" architecture: ", 1)[1].strip())

    if requested_arch == "universal":
        missing = {"arm64", "x86_64"} - ncnn_arches
        if missing:
            print("Error: building macOS universal, but ncnn static lib is missing architectures: {}.".format(", ".join(sorted(missing))))
            print("ncnn lib architectures detected: {}".format(", ".join(sorted(ncnn_arches)) if ncnn_arches else "(unknown)"))
            print("Fix options:")
            print("  1) Build extension single-arch: scons platform=macos arch=arm64 target=template_debug")
            print("  2) Rebuild ncnn as universal: -DCMAKE_OSX_ARCHITECTURES='arm64;x86_64'")
            Exit(1)
    elif requested_arch in ("arm64", "x86_64") and ncnn_arches and requested_arch not in ncnn_arches:
        print("Error: macOS arch={} requested, but ncnn static lib has architectures: {}.".format(
            requested_arch, ", ".join(sorted(ncnn_arches))
        ))
        print("Rebuild ncnn for {} or build with matching arch.".format(requested_arch))
        Exit(1)

# Link ncnn statically into the extension.
env.Append(LIBS=[File(ncnn_static_lib)])

library = env.SharedLibrary(
    target=os.path.join("bin", "libncnn_runner{}{}".format(env["suffix"], env["SHLIBSUFFIX"])),
    source=sources,
)

Default(library)
