#!/usr/bin/env python
import os
import subprocess

from SCons.Script import ARGUMENTS, Default, Exit, File, Glob, SConscript

env = SConscript("godot-cpp/SConstruct")
sources = Glob("src/*.cpp")
env.Append(CPPPATH=["src"])
requested_arch = str(ARGUMENTS.get("arch", env.get("arch", "")))

project_dir = os.path.abspath(".")
ncnn_root = os.path.join(project_dir, "thirdparty", "ncnn")
if not os.path.isdir(ncnn_root):
    print("Error: ncnn root was not found at {}".format(ncnn_root))
    Exit(1)

ncnn_include_candidates = [
    os.path.join(ncnn_root, "include"),
    os.path.join(ncnn_root, "src"),
    os.path.join(ncnn_root, "build", "src"),
    os.path.join(ncnn_root, "build-arm64", "src"),
    os.path.join(ncnn_root, "build-x86_64", "src"),
    os.path.join(ncnn_root, "install", "include"),
    os.path.join(ncnn_root, "install", "include", "ncnn"),
    os.path.join(ncnn_root, "install-arm64", "include"),
    os.path.join(ncnn_root, "install-arm64", "include", "ncnn"),
    os.path.join(ncnn_root, "install-x86_64", "include"),
    os.path.join(ncnn_root, "install-x86_64", "include", "ncnn"),
    os.path.join(ncnn_root, "build", "install", "include"),
    os.path.join(ncnn_root, "build", "install", "include", "ncnn"),
]
ncnn_include_paths = []
for path in ncnn_include_candidates:
    if os.path.isdir(path) and path not in ncnn_include_paths:
        ncnn_include_paths.append(path)
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
            print("  2) Build arm64 and x86_64 ncnn separately, then merge libncnn.a with lipo (see README).")
            Exit(1)
    elif requested_arch in ("arm64", "x86_64") and ncnn_arches and requested_arch not in ncnn_arches:
        print("Error: macOS arch={} requested, but ncnn static lib has architectures: {}.".format(
            requested_arch, ", ".join(sorted(ncnn_arches))
        ))
        print("Rebuild ncnn for {} or build with matching arch.".format(requested_arch))
        Exit(1)

# Link ncnn statically into the extension.
env.Append(LIBS=[File(ncnn_static_lib)])

# ncnn is built with OpenMP enabled by default, so libncnn.a references the GNU OpenMP runtime
# (GOMP_parallel) and pthreads. These must be linked AFTER libncnn.a — so the linker's default
# --as-needed keeps libgomp in DT_NEEDED — or the extension fails to load on Linux with
# "undefined symbol: GOMP_parallel". (macOS resolves its own OpenMP runtime, so scope to Linux.)
if env["platform"] == "linux":
    env.Append(LIBS=["gomp", "pthread"])

library = env.SharedLibrary(
    target=os.path.join("bin", "libncnn_runner{}{}".format(env["suffix"], env["SHLIBSUFFIX"])),
    source=sources,
)

Default(library)
