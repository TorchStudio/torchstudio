import sys
import importlib
import importlib.util
import argparse
import subprocess
import requests
parser = argparse.ArgumentParser()
parser.add_argument("--cuda", help="install nvidia gpu support", action="store_true", default=False)
parser.add_argument("--package", help="install specific package", action='append', nargs='+', default=[])
args, unknown = parser.parse_known_args()

if importlib.util.find_spec("pip") is None:
    print("Error: Pip is required to install the required packages.", file=sys.stderr)
    exit()

if not args.package:
    #NB: conda install are recommended before pip: https://www.anaconda.com/blog/using-pip-in-a-conda-environment
    pip_install="torch torchvision torchaudio torchtext"
    if (sys.platform.startswith('win') or sys.platform.startswith('linux')):
        if args.cuda:
            print("Checking the latest supported CUDA version...")
            highest_cuda_version=118 #11.8 highest supported cuda version for PyTorch 2.0
            try:
                pytorch_repo = requests.get("https://download.pytorch.org/whl/torch")
            except:
                print("Could not retrieve the latest supported CUDA version")
            else:
                import re
                regex_request=re.compile("cu([0-9]+)")
                results = re.findall(regex_request, pytorch_repo.text)
                highest_cuda_version=118
                for cuda_string in results:
                    cuda_version=int(cuda_string)
                    if  cuda_version > highest_cuda_version:
                        highest_cuda_version = cuda_version
            highest_cuda_string=str(highest_cuda_version)[:2]+"."+str(highest_cuda_version)[2:]
            print("Using CUDA "+highest_cuda_string)
            print("")
            pip_install+=" --index-url https://download.pytorch.org/whl/cu"+str(highest_cuda_version)

    print("Downloading and installing pytorch packages...")
    print("")

    result = subprocess.run([sys.executable, "-m", "pip", "install"]+pip_install.split())
    if result.returncode != 0:
        exit(result.returncode)
    print("")

    # onnx required for onnx export
    # datasets(+huggingface_hub) is required by hugging face hub
    # scipy required by torchvision: Caltech ImageNet SBD SVHN datasets and Inception v3 GoogLeNet models
    # pandas required by the dataset tutorial: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    # matplotlib-base required by torchstudio renderers
    # python-graphviz required by torchstudio graph
    # paramiko required for ssh connections (+updated cffi required on intel mac)
    # pysoundfile required by torchaudio datasets: https://pytorch.org/audio/stable/backend.html#soundfile-backend
    pip_install="onnx datasets scipy pandas matplotlib paramiko pysoundfile"

else:
    pip_install=" ".join(args.package[0])

print("Downloading and installing additional packages...")
print("")
result = subprocess.run([sys.executable, "-m", "pip", "install"]+pip_install.split())
if result.returncode != 0:
    exit(result.returncode)
