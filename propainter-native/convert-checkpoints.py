"""Convert the three official ProPainter checkpoints to lossless F32/I64 GGUF."""
import argparse
import hashlib
import math
from pathlib import Path
from tempfile import TemporaryDirectory

from gguf import GGUFWriter
import torch

CHECKPOINTS = {
    'raft-things.pth': 'propainter-raft',
    'recurrent_flow_completion.pth': 'propainter-flow-completion',
    'ProPainter.pth': 'propainter-generator',
}


def convert(source, destination, architecture):
    state = torch.load(source, map_location='cpu', weights_only=True)
    with source.open('rb') as checkpoint:
        digest = hashlib.file_digest(checkpoint, 'sha256').hexdigest()
    with TemporaryDirectory(dir=destination.parent) as temporary:
        target = Path(temporary) / destination.name
        writer = GGUFWriter(target, arch=architecture)
        writer.add_string('propainter.source_sha256', digest)
        for name, tensor in state.items():
            if tensor.dtype not in (torch.float32, torch.int64):
                raise ValueError(f'Unsupported checkpoint type for {name}: {tensor.dtype}')
            shape = tuple(tensor.shape)
            # ggml can store four dimensions. Keep PyTorch's shape separately.
            writer.add_uint32(f'propainter.rank.{name}', len(shape))
            if shape:
                writer.add_array(f'propainter.shape.{name}', list(shape))
            storage_shape = (math.prod(shape[:-3]), *shape[-3:]) if len(shape) > 4 else shape or (1,)
            writer.add_tensor(name, tensor.contiguous().reshape(storage_shape).numpy())
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        target.replace(destination)
    print(f'{destination.name}: {len(state)} tensors')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoints', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, architecture in CHECKPOINTS.items():
        convert(args.checkpoints / name, args.output / (Path(name).stem + '.gguf'), architecture)
