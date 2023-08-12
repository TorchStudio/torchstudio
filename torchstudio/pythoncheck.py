#workaround until Pytorch 1.12.1 is released: https://github.com/pytorch/pytorch/issues/78490
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import sys

print("Checking Python version...\n", file=sys.stderr)

import platform
import argparse
import importlib

parser = argparse.ArgumentParser()
parser.add_argument("--remote", help="check environment on a remote server and install ssh support if needed", action="store_true", default=False)
args, unknown = parser.parse_known_args()

#check python version first
python_version=(sys.version_info.major,sys.version_info.minor,sys.version_info.micro)
min_python_version=(3,7,0) if args.remote else (3,8,0) #3.7 required for ordered dicts and stdout/stderr utf8 encoding, 3.8 required for python parsing
if python_version<min_python_version:
    print("Error: Python "+'.'.join((str(i) for i in min_python_version))+" minimum is required.", file=sys.stderr)
    print("This environment has Python "+'.'.join((str(i) for i in python_version))+".", file=sys.stderr)
    exit(1)

print("Checking required packages...\n", file=sys.stderr)

#check all required packages are installed
checked_modules = ["torch", "torchvision"]
required_packages = ["pytorch", "torchvision"]
if not args.remote:
    checked_modules += ["torchaudio", "torchtext", "matplotlib", "graphviz"]
    required_packages += ["torchaudio", "torchtext", "matplotlib-base", "python-graphviz"]
missing_modules = []
for module_check in checked_modules:
    module = importlib.util.find_spec(module_check)
    if module is None:
        missing_modules.append(module_check)
    elif module_check=='torch':
        if python_version<(3,8,0):
            from importlib_metadata import version
        else:
            from importlib.metadata import version
        pytorch_version=tuple(int(i) if i.isdigit() else 0 for i in version('torch').split('.')[:3])
        min_pytorch_version=(1,9,0) #1.9 required for torch.package support, 1.10 preferred for stable torch.fx and profile-directed typing in torchscript
        if pytorch_version<min_pytorch_version:
            print("Error: PyTorch "+'.'.join((str(i) for i in min_pytorch_version))+" minimum is required.", file=sys.stderr)
            print("This environment has PyTorch "+'.'.join((str(i) for i in pytorch_version))+".", file=sys.stderr)
            exit(1)

if len(missing_modules)>0:
    #warn about missing modules
    print("Error: Missing Python modules:", file=sys.stderr)
    print(*missing_modules, sep = " ", file=sys.stderr)
    print("", file=sys.stderr)
    print("The following packages are required:", file=sys.stderr)
    print(' '.join(required_packages), file=sys.stderr)
    exit(1)
else:
    #install ssh support if necessary
    if not importlib.util.find_spec("paramiko"):
        if importlib.util.find_spec("pip"):
            print("Installing Paramiko (SSH for Python) using Pip...", file=sys.stderr)
            import subprocess
            subprocess.run([sys.executable, "-m", "pip", "install", "paramiko", "--quiet", "--quiet"]) #2 times quiet to remove all verbose
        else:
            print("Error: Pip is required to install Paramiko (SSH for Python).", file=sys.stderr)
            exit(1)

    #finally, list available devices
    print("Loading PyTorch...\n", file=sys.stderr)
    import torch

    print("Listing devices...\n", file=sys.stderr)
    devices = {}
    devices['cpu'] = {'name': 'CPU', 'modes': ['FP32']}

    cuda_names = {}
    for i in range(torch.cuda.device_count()):
        name=torch.cuda.get_device_name(i)
        if name in cuda_names:
            cuda_names[name]+=1
            name+=" "+str(cuda_names[name])
        else:
            cuda_names[name]=1

        modes = ['FP32']
        #same as torch.cuda.is_bf16_supported() but compatible with PyTorch<1.10, and not limited to current cuda device only
        cu_vers = torch.version.cuda
        if cu_vers is not None:
            cuda_maj_decide = int(cu_vers.split('.')[0]) >= 11
        else:
            cuda_maj_decide = False
        compute_capability=torch.cuda.get_device_properties(torch.cuda.device(i)).major #https://developer.nvidia.com/cuda-gpus
        if compute_capability>=8 and cuda_maj_decide: #RTX 3000 and higher
            modes+=['TF32','FP16','BF16']
        if compute_capability==7: #RTX 2000
            modes+=['FP16']

        devices['cuda:'+str(i)] = {'name': name, 'modes': modes}

    if pytorch_version>=(1,12,0):
        if torch.backends.mps.is_available():
            devices['mps'] = {'name': 'Metal', 'modes': ['FP32']}

    #other possible devices:
    #'hpu' (https://docs.habana.ai/en/latest/PyTorch_User_Guide/PyTorch_User_Guide.html)
    #'dml' (https://docs.microsoft.com/en-us/windows/ai/directml/gpu-pytorch-windows)
    devices_string_list=[]
    for id in devices:
        devices_string_list.append(id+' "'+devices[id]['name']+'" ('+'/'.join(devices[id]['modes'])+')')
    print("Ready ("+platform.platform()+", Python "+'.'.join((str(i) for i in python_version))+", PyTorch "+'.'.join((str(i) for i in pytorch_version))+", Devices: "+", ".join(devices_string_list)+")");
