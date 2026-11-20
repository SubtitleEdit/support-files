"""Verify the shipped archive on Windows using a short masked RGB stream."""
import hashlib
from pathlib import Path
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory

archive = Path(sys.argv[1]).resolve()
with archive.open('rb') as source:
    digest = hashlib.file_digest(source, 'sha256').hexdigest()
assert digest == archive.with_name(archive.name + '.sha256').read_text().split()[0]
with TemporaryDirectory(prefix='ProPainter relocated ') as temp:
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
    run = subprocess.run([executable, '-I', '-B', '-u', root / 'stream.py', '--stream', f'{width}x{height}'],
                         input=records, stdout=subprocess.PIPE, timeout=240, cwd=temp, check=True)
    length = len(frame)
    assert len(run.stdout) == 3 * length
    assert run.stdout[:length] == run.stdout[2 * length:] == frame
    restored = run.stdout[length:2 * length]
    assert all(a == b for i, (a, b) in enumerate(zip(restored, frame)) if not mark[i // 3])
    assert sum(abs(a - b) for a, b in zip(restored, frame)) < sum(abs(a - b) for a, b in zip(dirty, frame))
    print('Windows x64 decoded three frames, preserved clean pixels, and repaired the masked region.')
