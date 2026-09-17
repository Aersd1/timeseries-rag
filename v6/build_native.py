"""Build the optional V6 candidate heap; run on the destination machine."""
import os
import shutil
import subprocess
from pathlib import Path


def build():
    root = Path(__file__).resolve().parent
    cc = os.environ.get('CC') or shutil.which('gcc') or shutil.which('clang')
    if not cc:
        raise SystemExit('Install GCC/Clang or set CC. V6 still works with its NumPy fallback.')
    output = root / ('native.dll' if os.name == 'nt' else 'native.so')
    subprocess.run([cc, '-O3', '-std=c11', '-shared', '-fPIC', str(root/'native_search.c'),
                    '-o', str(output)], check=True)
    print(output)
    return output


if __name__ == '__main__':
    build()
