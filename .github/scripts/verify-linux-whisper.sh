#!/usr/bin/env bash
#
# Regression guard for the Linux whisper.cpp archives (SubtitleEdit #13680).
#
# whispercpp-184, -186 and -191 all shipped a whisper-cli that dies on
# startup with "error while loading shared libraries: libwhisper.so.1"
# because the packaging step never copied the library. Nothing in the build
# noticed, because the build machine had the libraries sitting in build/bin
# the whole time.
#
# So verify the way a user's machine sees it: copy the staged payload to a
# fresh location, run the loader from an unrelated working directory with an
# empty LD_LIBRARY_PATH, and require every bundled library to resolve on the
# strength of the $ORIGIN RPATH alone.
#
# Libraries we deliberately do NOT ship stay out of scope: libvulkan.so.1
# comes from the user's GPU driver, and libcuda/libcudart/libcublas come
# from their CUDA install. Only lib{whisper,ggml}* must resolve from within
# the archive.

set -euo pipefail

verify_dir=$(mktemp -d)
cp -a staging/. "$verify_dir/"

# Anywhere except the library directory — if the working directory were on
# the search path, a broken RPATH would still look fine here.
cd /

status=0

for f in "$verify_dir"/*; do
    # staging holds only ELF files at this point (the silero .bin model is
    # added after this step), but skip anything else defensively.
    readelf -h "$f" >/dev/null 2>&1 || continue

    name=$(basename "$f")
    out=$(LD_LIBRARY_PATH= ldd "$f" 2>/dev/null || true)

    missing=$(echo "$out" | awk '/not found/ {print $1}' | grep -E '^lib(whisper|ggml)' || true)
    if [ -n "$missing" ]; then
        echo "::error::$name cannot resolve bundled libraries: $(echo "$missing" | tr '\n' ' ')"
        echo "$out"
        status=1
    fi
done

# whisper-cli is the one the user actually launches, and an empty NEEDED
# list would sail through the loop above. Assert it really did bind to the
# bundled libraries rather than to nothing at all.
cli_out=$(LD_LIBRARY_PATH= ldd "$verify_dir/whisper-cli" 2>/dev/null || true)
for required in libwhisper.so libggml.so; do
    if ! echo "$cli_out" | grep -q "$required"; then
        echo "::error::whisper-cli does not link against $required — check the build configuration"
        status=1
    fi
done

echo "whisper-cli resolved dependencies (LD_LIBRARY_PATH cleared, cwd=/):"
echo "$cli_out"

rm -rf "$verify_dir"

if [ "$status" -ne 0 ]; then
    echo "::error::Linux archive would ship broken — refusing to publish."
    exit 1
fi

echo "OK: every bundled whisper/ggml library resolves via \$ORIGIN."
