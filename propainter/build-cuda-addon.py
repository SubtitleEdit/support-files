"""Make the small Windows x64 CUDA add-on from the published CPU package."""
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory

RELEASE = 'https://github.com/SubtitleEdit/support-files/releases/download/propainter-0.2.0/'
CPU = 'propainter-0.2.0-windows-x64-cpu.tar.gz'
CPU_HASH = '5118fc69fb290bca679a65a11cba43703e8778474659cc3031ea234692a115a3'
ADDON = 'propainter-0.2.0-windows-x64-cuda-addon.tar.gz'
ADDON_ROOT = 'propainter-windows-x64-cuda'


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Pinned source changed near {old!r}')
    return text.replace(old, new)


def digest(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def build(temp, out):
    cpu = temp / CPU
    subprocess.run(['curl.exe', '-fsSL', '--retry', '3', RELEASE + CPU, '-o', str(cpu)], check=True)
    if digest(cpu) != CPU_HASH:
        raise ValueError('CPU package checksum mismatch')

    root = temp / ADDON_ROOT
    root.mkdir()
    with tarfile.open(cpu, 'r:gz') as archive:
        launch = archive.extractfile('propainter-windows-x64/launch.py').read().decode('utf-8')
        stream = archive.extractfile('propainter-windows-x64/stream.py').read().decode('utf-8')
        license_text = archive.extractfile('propainter-windows-x64/app/LICENSE').read()
    (root / 'LICENSE').write_bytes(license_text)

    launch = replace_once(launch, "root = Path(__file__).resolve().parent\napp = root / 'app'",
                          """addon = Path(__file__).resolve().parent
root = addon.parent
sys.path.insert(0, str(addon / 'site'))
app = root / 'app'""")
    launch = replace_once(launch, "misc.get_device = lambda: torch.device('cpu')",
                          """device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'ProPainter CUDA package running on {device.type}', file=sys.stderr)
misc.get_device = lambda: device""")
    (root / 'launch-cuda.py').write_text(launch, encoding='utf-8')

    stream = replace_once(stream, 'ROOT = Path(__file__).resolve().parent',
                          'ADDON = Path(__file__).resolve().parent\nROOT = ADDON.parent')
    stream = replace_once(stream, '        self.metal = False', '        self.metal = False\n        self.cuda = True')
    stream = replace_once(stream, """            if not self.metal:
                result = temp / 'cpu'""", """            if self.cuda:
                result = temp / 'cuda'
                status = subprocess.run([PYTHON, '-I', '-B', '-u', ADDON / 'launch-cuda.py',
                                         '--frames', clip, '--masks', masks, '--output', result,
                                         '--raft-iterations', '12'], stdout=sys.stderr)
                self.cuda = status.returncode == 0
                if not self.cuda:
                    print('ProPainter CUDA did not complete. Retrying on the base CPU runtime.', file=sys.stderr)
            if not self.metal and not self.cuda:
                result = temp / 'cpu'""")
    (root / 'stream-cuda.py').write_text(stream, encoding='utf-8')
    shutil.copy2(Path(__file__).with_name('install-cuda.py'), root / 'install-cuda.py')
    (root / 'README.md').write_text('''# ProPainter CUDA 12.8 add-on for Windows x64

This add-on uses the ProPainter 0.2.0 Windows x64 CPU package for Python, models, and the
baseline CPU runtime. Subtitle Edit installs the add-on under the CPU runtime's `cuda`
directory. First-time setup downloads about 2.8 GB in pinned wheels directly from PyTorch.
An NVIDIA GPU with a CUDA 12.8-compatible driver is needed for acceleration. Initialization
or inference errors cause Subtitle Edit's runner to retry with the base CPU runtime.

ProPainter and its model weights are licensed for non-commercial use. See LICENSE. The CUDA
runtime is obtained from the official [PyTorch wheel repository](https://download.pytorch.org/whl/cu128/).
''', encoding='utf-8')
    bundle = out / ADDON
    with tarfile.open(bundle, 'w:gz') as archive:
        archive.add(root, arcname=ADDON_ROOT)
    (out / f'{ADDON}.sha256').write_bytes(f'{digest(bundle)}  {ADDON}\n'.encode('ascii'))
    return cpu, bundle


def smoke(cpu, bundle, temp):
    runtime = temp / 'relocated package'
    with tarfile.open(cpu, 'r:gz') as archive:
        archive.extractall(runtime, filter='data')
    root = runtime / 'propainter-windows-x64'
    with tarfile.open(bundle, 'r:gz') as archive:
        archive.extractall(root, filter='data')
    addon = root / ADDON_ROOT
    addon.rename(root / 'cuda')
    addon = root / 'cuda'
    python = root / 'python/python.exe'
    subprocess.run([python, '-I', '-B', addon / 'install-cuda.py'], check=True)
    assert (addon / 'installed.txt').is_file()
    subprocess.run([sys.executable, Path(__file__).with_name('smoke-windows.py'),
                    cpu, '--root', root, '--cuda'], check=True)


if __name__ == '__main__':
    if sys.platform != 'win32':
        sys.exit('Run the CUDA add-on build on Windows.')
    output = Path(sys.argv[1]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as directory:
        folder = Path(directory)
        cpu_archive, addon_archive = build(folder, output)
        smoke(cpu_archive, addon_archive, folder)
