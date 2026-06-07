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

# Per-target cross/native build dir (e.g. build-linux-x86_64, build-windows-x86_64), so a
# cross-compiled ncnn is preferred over the host build/install (which holds the host arch).
_target_arch = requested_arch or str(env.get("arch", ""))
ncnn_target_build = os.path.join(ncnn_root, "build-{}-{}".format(env["platform"], _target_arch))

ncnn_include_candidates = [
    os.path.join(ncnn_target_build, "install", "include"),
    os.path.join(ncnn_target_build, "install", "include", "ncnn"),
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
        os.path.join(ncnn_target_build, "install", "lib", "ncnn.lib"),
        os.path.join(ncnn_target_build, "install", "lib", "libncnn.a"),
        os.path.join(ncnn_root, "build", "install", "lib", "ncnn.lib"),
        os.path.join(ncnn_root, "build", "install", "lib", "libncnn.a"),
    ]
else:
    ncnn_static_candidates = [
        os.path.join(ncnn_target_build, "install", "lib", "libncnn.a"),
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
# ncnn built with OpenMP (the native default) needs libgomp; a build with NCNN_OPENMP=OFF
# (e.g. the zig cross-compile, which has no libgomp) must not link it. Gate with ncnn_openmp=.
_ncnn_openmp = str(ARGUMENTS.get("ncnn_openmp", "yes")).lower() not in ("0", "no", "false")
if env["platform"] == "linux":
    env.Append(LIBS=(["gomp", "pthread"] if _ncnn_openmp else ["pthread"]))

# When cross-compiling from a macOS host, SCons keeps host-derived suffixes: the shared
# library would be named ".dylib" (not ".so") and shared objects ".os" (which clang/zig
# reject as an "unrecognized file extension"). Pin the target's own suffixes. godot-cpp's
# own objects are already built at this point, so changing SHOBJSUFFIX is safe.
if env["platform"] == "linux":
    env["SHLIBSUFFIX"] = ".so"
    env["SHOBJSUFFIX"] = ".o"
elif env["platform"] == "windows":
    env["SHLIBSUFFIX"] = ".dll"
    env["SHOBJSUFFIX"] = ".o"
elif env["platform"] == "android":
    # SHLIBSUFFIX is already ".so" (set by godot-cpp's android tool); just fix the
    # shared-object extension the NDK clang would otherwise reject on a macOS host.
    env["SHOBJSUFFIX"] = ".o"

library = env.SharedLibrary(
    target=os.path.join("bin", "libncnn_runner{}{}".format(env["suffix"], env["SHLIBSUFFIX"])),
    source=sources,
)

Default(library)
