# Native ProPainter

The C++17 renderer loads the three ProPainter GGUF checkpoints directly. It implements bidirectional RAFT, recurrent flow completion, image and feature propagation, sparse temporal attention, and the inpainting decoder. The runtime requires no Python, PyTorch, or network access.

## Subtitle Edit stream

Place all three GGUF files under the executable's adjacent `models/` directory.

```sh
./propainter-native --stream WIDTHxHEIGHT < frames-and-masks.bin > repaired.rgb
```

Each input record consists of tightly packed RGB24 followed by a single-channel mask8, both with the declared dimensions. A mask byte of 128–255 marks a pixel to repair. The output contains one RGB24 frame per input record. Bytes outside the input mask are preserved exactly. The stream detects scene cuts and processes bounded chunks with temporal context. It crops and scales around the mask, then maps the result back to the original frame. Logs go to stderr.

The raw-crop interface is useful when frames and masks are already prepared:

```sh
./propainter-native models WIDTH HEIGHT FRAMES input.rgb masks.gray output.rgb 12 4
```

This interface takes separate RGB24 and mask8 files. Width and height must be multiples of eight, up to 640 pixels. Every nonzero byte in the raw mask marks a target. The last two optional arguments in the example are RAFT iterations and mask dilation in pixels. The raw interface composites through the dilated mask. The stream interface composites through the original mask.

## Build

```sh
git clone https://github.com/CrispStrobe/ggml.git
git -C ggml checkout 3a27b6c665b191cb230be367db5c52d81dd3e2ac
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
  -DGGML_NATIVE=OFF -DGGML_SOURCE="$PWD/ggml"
cmake --build build --config Release --target propainter-native
```

CPU is the default backend. To try GPU graph execution, compile with `-DGGML_METAL=ON` or `-DGGML_CUDA=ON`, then set `PROPAINTER_BACKEND=metal` or `PROPAINTER_BACKEND=cuda`. Unsupported ggml graphs and operations such as flow warping still run on the CPU. The GPU route is experimental.

## Models and verification

The three GGUF files are lossless conversions of the official v0.1.0 [ProPainter checkpoints](https://github.com/sczhou/ProPainter/releases/tag/v0.1.0). They are covered by the included S-Lab License 1.0, which permits non-commercial redistribution and use. The optional `convert-checkpoints.py` developer tool needs PyTorch and `gguf==0.19.0` if you want to reproduce the conversion from the upstream checkpoint files. The reference checks likewise use PyTorch only during development.

The full native CPU renderer matched PyTorch exactly on five real 96×56 frames and within one 8-bit RGB level on eight 464×168 frames. The reference checks compare the entire output, including unmasked pixels.

### Windows CUDA package

The Windows CUDA build bundles the CUDA runtime and cuBLAS libraries. It selects an NVIDIA
GPU automatically when available and falls back to ggml CPU when it cannot initialize one.
Use the same RGB24/mask8 stream interface as the CPU build. GPU acceleration requires a
CUDA 12.8-compatible NVIDIA driver and a GPU with compute capability 7.5 or later.

Set the PowerShell environment variable PROPAINTER_BACKEND to 'cuda' to require CUDA for a
test run, or to 'cpu' to force CPU inference. The renderer logs the selected device to stderr.
To run the stream smoke test on an NVIDIA machine, set the variable to 'cuda' before running
smoke.ps1 from this source tree.
