"""Build the relocatable Windows x64 ProPainter CPU package."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory

PACKAGE = 'propainter-0.2.0-windows-x64-cpu.tar.gz'
BASE = 'https://github.com/SubtitleEdit/support-files/releases/download/propainter-0.2.0/propainter-0.2.0-macos-arm64.tar.gz'
BASE_SHA256 = '7d1711f39c1d8bf77d04de95a9bd8fb9d2610b95cdcd360508af016f0456fd82'
PYTHON_RELEASE = '20261110'
PYTHON_VERSION = '3.13.18'
PYTHON = ('https://github.com/astral-sh/python-build-standalone/releases/download/20261110/'
          'cpython-3.13.18%2B20261110-x86_64-pc-windows-msvc-install_only_stripped.tar.gz')
PYTHON_SHA256 = '2e0fed09d48989f64c9305da9ff2d37a41c0d13ab2aa6283405b2b620dfbf805'
WHEELS = (
    'https://download-r2.pytorch.org/whl/cpu/torch-2.11.0%2Bcpu-cp313-cp313-win_amd64.whl'
    '#sha256=ed70d4a4fc9f8b826c02fa1a9800a83820fb2fa6ae607680b53390f9ef394d85',
    'https://download-r2.pytorch.org/whl/cpu/torchvision-0.26.0%2Bcpu-cp313-cp313-win_amd64.whl'
    '#sha256=34ac55a1f614baca2e0f5cef20ddb36184ee3503423871260e1ddd72caf9cb5f',
)
DEPENDENCIES = '''
numpy==2.4.4 scipy==1.18.0 pillow==12.3.0 einops==0.8.2 requests==2.33.1 tqdm==4.67.3
filelock==3.28.0 typing-extensions==4.15.0 setuptools==81.0.0 sympy==1.14.0
networkx==3.6.1 Jinja2==3.1.6 fsspec==2026.3.0 mpmath==1.3.0 MarkupSafe==3.0.3
urllib3==2.6.3 idna==3.11 certifi==2026.2.25 charset-normalizer==3.4.7 colorama==0.4.6
'''.split()


def sha256(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def download(url, destination, expected=None):
    subprocess.run(['curl.exe', '-fsSL', '--retry', '3', url, '-o', str(destination)], check=True)
    if expected and sha256(destination) != expected:
        raise ValueError(f'SHA-256 mismatch for {url}')


def replace_once(path, old, new):
    text = path.read_text(encoding='utf-8')
    if text.count(old) != 1:
        raise ValueError(f'Support Files source changed: {path}')
    path.write_text(text.replace(old, new), encoding='utf-8')


def build(staging, output):
    content = staging / 'content'
    root = content / 'propainter-windows-x64'
    root.mkdir(parents=True)
    download(BASE, staging / 'base.tar.gz', BASE_SHA256)
    with tarfile.open(staging / 'base.tar.gz', 'r:gz') as archive:
        for member in archive:
            name = member.name.removeprefix('propainter-macos-arm64/')
            if name in ('launch.py', 'stream.py', 'manifest.json') or name.startswith('app/') or name.startswith('licenses/'):
                archive.extract(member, staging / 'base', filter='data')
    base = staging / 'base/propainter-macos-arm64'
    for name in ('launch.py', 'stream.py', 'manifest.json'):
        shutil.copy2(base / name, root / name)
    for name in ('app', 'licenses'):
        shutil.copytree(base / name, root / name)
    (root / 'licenses/ggml.txt').unlink(missing_ok=True)
    version_licence = f'https://raw.githubusercontent.com/python/cpython/v{PYTHON_VERSION}/LICENSE'
    download(version_licence, root / 'licenses/CPython.txt')

    download(PYTHON, staging / 'python.tar.gz', PYTHON_SHA256)
    with tarfile.open(staging / 'python.tar.gz', 'r:gz') as archive:
        archive.extractall(root, filter='data')
    python = root / 'python/python.exe'
    if not python.exists():
        raise FileNotFoundError(python)
    site = root / 'python/Lib/site-packages'
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--target', str(site), '--no-deps',
                    '--only-binary=:all:', '--no-compile', *DEPENDENCIES, *WHEELS], check=True)
    shutil.rmtree(site / 'bin', ignore_errors=True)

    stream = root / 'stream.py'
    replace_once(stream, """        self.metal = os.environ.get('PROPAINTER_CPU_ONLY') != '1' and subprocess.run(
            [ROOT / 'native/metal-probe'], stdout=sys.stderr).returncode == 0""", '        self.metal = False')
    replace_once(root / 'launch.py', '0.2.0 (macOS arm64)', '0.2.0 (Windows x64 CPU)')
    manifest = root / 'manifest.json'
    metadata = json.loads(manifest.read_text(encoding='utf-8'))
    metadata.pop('minimum_macos', None)
    metadata.pop('ggml_source_commit', None)
    metadata.update(platform='windows-x64', device='cpu', python=PYTHON_VERSION,
                    python_build_standalone_release=PYTHON_RELEASE, python_archive_sha256=PYTHON_SHA256,
                    python_packages={**dict(dep.split('==') for dep in DEPENDENCIES),
                                     'torch': '2.11.0+cpu', 'torchvision': '0.26.0+cpu'})
    manifest.write_text(json.dumps(metadata, indent=2, sort_keys=True) + '\n', encoding='utf-8')

    (root / 'README.md').write_text('''# ProPainter for Subtitle Edit (Windows x64 CPU)

This standalone package contains Python, PyTorch, the official ProPainter weights, and the
small-crop RAFT fix. ProPainter is licensed for non-commercial use. See `app/LICENSE`.
The runtime works offline. Subtitle Edit sends RGB24 + mask8 frame records and receives RGB24
frames in order. For standalone usage on Windows:

```
python\\python.exe -I -B -u launch.py --help
python\\python.exe -I -B -u stream.py --stream 960x400
```

If Windows reports a missing MSVC runtime DLL, install the Microsoft Visual C++ x64 Redistributable.
See the manifest for pinned components and `licenses/` for notices.
''', encoding='utf-8')
    archive_path = output / PACKAGE
    with tarfile.open(archive_path, 'w:gz', compresslevel=3) as archive:
        archive.add(root, arcname=root.name)
    (output / f'{PACKAGE}.sha256').write_bytes(f'{sha256(archive_path)}  {PACKAGE}\n'.encode('ascii'))
    print(f'Packaged {archive_path} ({archive_path.stat().st_size} bytes)')


if __name__ == '__main__':
    if sys.platform != 'win32' or sys.version_info[:2] != (3, 13):
        sys.exit('Run this build script with Python 3.13 on Windows x64.')
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as temp:
        build(Path(temp), out)
