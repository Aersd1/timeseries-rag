import os,shutil,subprocess
from pathlib import Path
root=Path(__file__).resolve().parent
gcc=os.environ.get('CC') or shutil.which('gcc')
if not gcc:
    raise SystemExit('C compiler not found. Add gcc to PATH or set CC to the compiler executable path.')
output=root/('native.dll' if os.name=='nt' else 'native.so')
subprocess.run([gcc,'-O3','-std=c11','-shared','-fPIC',str(root/'native.c'),'-o',str(output),'-lm'],check=True)
print(output)
