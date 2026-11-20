"""Install the official CUDA 12.8 PyTorch wheels into the ProPainter add-on folder."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
WHEELS = (
    'https://download-r2.pytorch.org/whl/cu128/torch-2.11.0%2Bcu128-cp313-cp313-win_amd64.whl'
    '#sha256=6f367e62fd81b75cdf23ca4b75ced834d2db2cf98d1588ac935bde345de9de23',
    'https://download-r2.pytorch.org/whl/cu128/torchvision-0.26.0%2Bcu128-cp313-cp313-win_amd64.whl'
    '#sha256=0232cb219927a52d6c98ff202f32d1cdf4802c2195a85fc1f1a0c1b0b4983a4d',
)

if sys.platform != 'win32' or sys.version_info[:2] != (3, 13):
    sys.exit('The CUDA add-on requires the bundled Windows x64 Python 3.13 runtime.')

site = ROOT / 'site'
subprocess.run([sys.executable, '-I', '-m', 'pip', '--isolated', '--disable-pip-version-check',
                'install', '--no-input', '--no-cache-dir', '--no-deps', '--only-binary=:all:',
                '--progress-bar', 'raw', '--target', str(site), *WHEELS], check=True)
subprocess.run([sys.executable, '-I', '-c',
                'import sys; sys.path.insert(0, sys.argv[1]); import torch, torchvision; '
                'assert torch.version.cuda == "12.8"; '
                'assert torchvision.extension._has_ops(); '
                'print("PyTorch CUDA runtime verified:", torch.version.cuda, '
                '"NVIDIA GPU available:", torch.cuda.is_available())', str(site)], check=True)
(ROOT / 'installed.txt').write_text('PyTorch 2.11.0 CUDA 12.8\n', encoding='ascii')
