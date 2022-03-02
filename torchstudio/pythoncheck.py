import sys

print("Checking Python version...\n", file=sys.stderr)

import argparse
import importlib

parser = argparse.ArgumentParser()
parser.add_argument("--remote", help="check environment on a remote server and install ssh support if needed", action="store_true", default=False)
args, unknown = parser.parse_known_args()

#check python version first
python_version=(sys.version_info.major,sys.version_info.minor)
min_python_version=(3,7) if args.remote else (3,8) #3.7 required for ordered dicts and stdout/stderr utf8 encoding, 3.8 required for python parsing
if python_version<min_python_version:
    print("Error: Python "+str(min_python_version[0])+"."+str(min_python_version[1])+" minimum is required.", file=sys.stderr)
    print("This environment has Python "+str(python_version[0])+"."+str(python_version[1])+".", file=sys.stderr)
    exit(1)

print("Checking required packages...\n", file=sys.stderr)

#check all required packages are installed
checked_modules = ["torch", "torchvision"]
required_packages = ["pytorch", "torchvision"]
if not args.remote:
    checked_modules += ["torchaudio", "matplotlib", "graphviz"]
    required_packages += ["torchaudio", "matplotlib-base", "python-graphviz"]
missing_modules = []
for module_check in checked_modules:
    module = importlib.util.find_spec(module_check)
    if module is None:
        missing_modules.append(module_check)
    elif module_check=='torch':
        if python_version<(3,8):
            from importlib_metadata import version
        else:
            from importlib.metadata import version
        pytorch_version=tuple(int(i) for i in version('torch').split('.')[:2])
        min_pytorch_version=(1,9) #1.9 required for torch.package support, 1.10 preferred for stable torch.fx and profile-directed typing in torchscript
        if pytorch_version<min_pytorch_version:
            print("Error: PyTorch "+str(min_pytorch_version[0])+"."+str(min_pytorch_version[1])+" minimum is required.", file=sys.stderr)
            print("This environment has PyTorch "+str(pytorch_version[0])+"."+str(pytorch_version[1])+".", file=sys.stderr)
            exit(1)

if len(missing_modules)>0:
    #warn about missing modules
    print("Error: Missing Python modules:", file=sys.stderr)
    print(*missing_modules, sep = " ", file=sys.stderr)
    print("The following packages are required:", file=sys.stderr)
    print(' '.join(required_packages), file=sys.stderr)
    exit(1)
else:
    #install ssh support if necessary
    if not importlib.util.find_spec("paramiko"):
        if importlib.util.find_spec("conda"):
            print("Installing Paramiko (SSH for Python) using Conda...", file=sys.stderr)
            import conda.cli.python_api as Conda
            Conda.run_command(Conda.Commands.INSTALL,['paramiko','-c','conda-forge', "--quiet", "--quiet"]) #2 times quiet to remove all verbose
        elif importlib.util.find_spec("pip"):
            print("Installing Paramiko (SSH for Python) using Pip...", file=sys.stderr)
            import subprocess
            subprocess.run([sys.executable, "-m", "pip", "install", "paramiko", "--quiet", "--quiet"]) #2 times quiet to remove all verbose
        else:
            print("Error: Conda or Pip is required to install Paramiko (SSH for Python).", file=sys.stderr)
            exit(1)

    #finally, list available devices
    print("Loading PyTorch...\n", file=sys.stderr)

    import torch

    print("Listing devices...\n", file=sys.stderr)

    devices = {}
    devices['cpu'] = {'name': 'CPU', 'pin_memory': False}
    for i in range(torch.cuda.device_count()):
        devices['cuda:'+str(i)] = {'name': torch.cuda.get_device_name(i), 'pin_memory': True}
    #other possible devices:
    #'hpu' (https://docs.habana.ai/en/latest/PyTorch_User_Guide/PyTorch_User_Guide.html)
    #'dml' (https://docs.microsoft.com/en-us/windows/ai/directml/gpu-pytorch-windows)
    devices_string_list=[]
    for id in devices:
        devices_string_list.append(devices[id]['name']+" ("+id+")")
    print(("Online and functional " if args.remote else "Functional environment ")+"(Python "+str(python_version[0])+"."+str(python_version[1])+", PyTorch "+str(pytorch_version[0])+"."+str(pytorch_version[1])+", Devices: "+", ".join(devices_string_list)+")");


