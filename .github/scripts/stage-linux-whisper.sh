#!/usr/bin/env bash
#
# Stage a Linux whisper.cpp build into ./staging for packaging.
#
# Shared by the linux-vulkan and linux-cuda jobs of
# build-whispercpp-release.yml. Run from the whisper.cpp build root (the
# directory holding ./build).
#
# Why this is not just `cp build/bin/libggml*.so staging/` — that was the
# original one-liner, and it shipped three separate defects for four
# releases (SubtitleEdit issue #13680):
#
#   1. The glob never matched libwhisper.so*, so the library whisper-cli is
#      linked against was simply absent from the archive.
#   2. cmake leaves a symlink chain in build/bin
#      (libggml.so -> libggml.so.0 -> libggml.so.0.9.4). A bare `cp`
#      dereferences those, so the file landed under the *unversioned* name
#      while its SONAME — and whisper-cli's DT_NEEDED entry — said
#      libggml.so.0. The loader looks up the SONAME, so it never matched.
#   3. The RUNPATH cmake bakes in points at the CI build directory
#      (/home/runner/work/...), which does not exist on a user's machine,
#      and the loader does not search the working directory.
#
# Shipping the symlink chain as-is would not fix (2): Subtitle Edit unpacks
# these archives with .NET's ZipArchive, whose zip path writes a symlink
# entry out as an ordinary file containing the target path. So every library
# is staged as a real file named after its SONAME.

set -euo pipefail

# build/bin is where whisper.cpp puts both binaries and libraries; older /
# differently-configured trees split them, so accept the alternatives too.
lib_dirs=(build/bin build/lib build/src build/ggml/src)

mkdir -p staging
cp build/bin/whisper-cli staging/

soname_of() {
    # Empty output when the ELF has no DT_SONAME (true of the dlopen'd ggml
    # backend plugins: libggml-vulkan.so, libggml-cuda.so, libggml-cpu-*.so).
    readelf -d "$1" 2>/dev/null | sed -n 's/.*SONAME.*\[\(.*\)\].*/\1/p'
}

shopt -s nullglob
staged=0
for dir in "${lib_dirs[@]}"; do
    for src in "$dir"/libwhisper.so* "$dir"/libggml*.so*; do
        # Collapse the symlink chain to the one real ELF behind it, so each
        # library is staged exactly once no matter how many aliases point at it.
        real=$(readlink -f "$src")
        [ -f "$real" ] || continue

        soname=$(soname_of "$real")
        name=${soname:-$(basename "$real")}

        if [ ! -e "staging/$name" ]; then
            cp -L "$real" "staging/$name"
            staged=$((staged + 1))
        fi
    done
done

if [ "$staged" -eq 0 ]; then
    echo "::error::No whisper/ggml shared libraries found under: ${lib_dirs[*]}"
    exit 1
fi

# Point every ELF at its own directory. patchelf writes DT_RUNPATH, which is
# NOT inherited by transitive dependencies — libggml.so.0 does not benefit
# from whisper-cli's RUNPATH when it looks for libggml-base.so.0 — so the
# libraries need this just as much as the executable does.
for f in staging/whisper-cli staging/*.so*; do
    patchelf --set-rpath '$ORIGIN' "$f"
done

echo "Staged $staged shared libraries:"
ls -la staging
echo
echo "whisper-cli dynamic section:"
readelf -d staging/whisper-cli | grep -E 'NEEDED|RUNPATH|RPATH'
