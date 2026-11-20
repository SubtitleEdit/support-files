"""Verify the shipped archive on Windows using a short masked RGB stream."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import tarfile
from tempfile import TemporaryDirectory

parser = argparse.ArgumentParser()
parser.add_argument('archive', type=Path)
parser.add_argument('--root', type=Path)
parser.add_argument('--cuda', action='store_true')
args = parser.parse_args()
archive = args.archive.resolve()
with archive.open('rb') as source:
    digest = hashlib.file_digest(source, 'sha256').hexdigest()
if args.root is None:
    assert digest == archive.with_name(archive.name + '.sha256').read_text().split()[0]
with TemporaryDirectory(prefix='ProPainter relocated ') as temp:
    root = args.root
    if root is None:
        with tarfile.open(archive, 'r:gz') as source:
            source.extractall(temp, filter='data')
        root = Path(temp, 'propainter-windows-x64')
    executable = root / 'python/python.exe'
    launch = subprocess.check_output([executable, '-I', '-B', root / 'launch.py', '--version'], cwd=temp, text=True)
    print(launch.strip())
    width, height = 96, 72
    frame = bytes(channel for y in range(height) for x in range(width)
                  for channel in (60 + x, 70 + y, 90 + (x+y) % 25))
    mark = bytes(255 if 35 <= x < 61 and 32 <= y < 39 else 0
                 for y in range(height) for x in range(width))
    dirty = bytes(245 if mark[index // 3] else value for index, value in enumerate(frame))
    blank = bytes(width * height)
    records = frame + blank + dirty + mark + frame + blank
    stream = root / ('cuda/stream-cuda.py' if args.cuda else 'stream.py')
    run = subprocess.run([executable, '-I', '-B', '-u', stream, '--stream', f'{width}x{height}'],
                         input=records, capture_output=True, timeout=240, cwd=temp, check=True)
    log = run.stderr.decode(errors='replace')
    print(log)
    if args.cuda:
        assert 'ProPainter CUDA package running on ' in log and 'Retrying' not in log
    length = len(frame)
    assert len(run.stdout) == 3 * length
    assert run.stdout[:length] == run.stdout[2 * length:] == frame
    restored = run.stdout[length:2 * length]
    assert all(a == b for i, (a, b) in enumerate(zip(restored, frame)) if not mark[i // 3])
    assert sum(abs(a - b) for a, b in zip(restored, frame)) < sum(abs(a - b) for a, b in zip(dirty, frame))
    print('Windows x64 decoded three frames, preserved clean pixels, and repaired the masked region.')
