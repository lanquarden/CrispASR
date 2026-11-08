#!/usr/bin/env python3
"""Build the crispasr-voice experiment binary against the existing build-vk tree.

Why not CMake yet: adding a second executable to examples/cli/CMakeLists.txt
would recompile ~100 adapter sources. This script reuses the objects the CLI
already built, so the iteration loop is a few seconds. CMake integration lands
with the upstream-ready branch.

Usage: python3 tools/build_voice_loop.py [--build-dir build-vk] [-v]
"""
import os
import re
import shlex
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, **kw)


def ninja_commands(build_dir, target):
    out = subprocess.run(
        ["ninja", "-C", build_dir, "-t", "commands", target],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout
    return [line for line in out.splitlines() if line.strip()]


def find_compile_cmd(commands, needle):
    for line in commands:
        if needle in line and " -c " in line:
            return line
    return None


def find_link_cmd(commands, target):
    for line in commands:
        if f" -o {target} " in line or line.endswith(f" -o {target}"):
            return line
    return None


def split_flags(cmd):
    parts = shlex.split(cmd)
    # drop the compiler and the output/source pair
    assert parts[0].endswith("c++") or "++" in parts[0], parts[0]
    out, i = [], 1
    while i < len(parts):
        p = parts[i]
        if p == "-o" or p == "-MD" or p == "-MT" or p == "-MF":
            i += 2
            continue
        if p == "-c":
            i += 1
            continue
        if p.endswith(".cpp") or p.endswith(".o") or p.endswith(".d"):
            i += 1
            continue
        out.append(p)
        i += 1
    return out


def main():
    build_dir = "build-vk"
    verbose = "-v" in sys.argv
    for a in sys.argv[1:]:
        if a.startswith("--build-dir="):
            build_dir = a.split("=", 1)[1]

    build_path = os.path.join(ROOT, build_dir)
    target = "bin/crispasr"
    commands = ninja_commands(build_path, target)

    ref = find_compile_cmd(commands, "crispasr_backend_qwen3.cpp.o")
    if not ref:
        sys.exit("could not find a reference compile command; build bin/crispasr first")
    flags = split_flags(ref)
    # debug/dep flags from the reference build are harmless
    flags = [f for f in flags if not f.startswith("-M")]
    flags = [f for f in flags if not f.startswith("-fmodule") and not f.startswith("-fmodules")]

    link = find_link_cmd(commands, target)
    if not link:
        sys.exit("could not find the link command for bin/crispasr")

    voice_dir = os.path.join(ROOT, "examples", "cli", "voice")
    sources = sorted(
        os.path.join(voice_dir, f) for f in os.listdir(voice_dir) if f.endswith(".cpp")
    )
    if not sources:
        sys.exit("no voice sources found")

    obj_dir = os.path.join(build_path, "voice-obj")
    os.makedirs(obj_dir, exist_ok=True)
    objects = []
    for src in sources:
        obj = os.path.join(obj_dir, os.path.basename(src)[:-4] + ".o")
        cmd = ["c++"] + flags + ["-I" + os.path.join(ROOT, "examples", "cli"), "-I" + voice_dir, "-c", src, "-o", obj]
        if verbose:
            print(" ".join(shlex.quote(c) for c in cmd))
        run(cmd)
        objects.append(obj)

    # Reuse every CLI adapter object except the one that defines main().
    cli_obj_dir = os.path.join(build_path, "examples", "cli", "CMakeFiles", "crispasr-cli.dir")
    cli_objects = []
    for base, _dirs, files in os.walk(cli_obj_dir):
        for f in files:
            if not f.endswith(".o"):
                continue
            if f == "cli.cpp.o":
                continue
            cli_objects.append(os.path.join(base, f))

    parts = shlex.split(link)
    # The ninja line is `: && /usr/bin/c++ <flags> <objs> -o bin/crispasr <libs>`.
    # Keep everything after the compiler that is a linker argument.
    try:
        ci = next(i for i, p in enumerate(parts)
                  if p.endswith("c++") or p.endswith("g++") or p.endswith("clang++"))
    except StopIteration:
        sys.exit("could not locate the compiler in the link command")
    libs = []
    i = ci + 1
    while i < len(parts):
        p = parts[i]
        if p in ("&&", "&&:", ":"):
            break
        if p == "-o":
            i += 2
            continue
        if p.startswith("-Wl,--dependency-file="):
            i += 1
            continue
        if p == target or p.endswith(".o"):
            i += 1
            continue
        libs.append(p)
        i += 1

    out_bin = os.path.join(build_path, "bin", "crispasr-voice")
    cmd = ["c++"] + objects + cli_objects + libs + ["-o", out_bin]
    if verbose:
        print(" ".join(shlex.quote(c) for c in cmd))
    run(cmd, cwd=build_path)
    print(out_bin)


if __name__ == "__main__":
    main()
