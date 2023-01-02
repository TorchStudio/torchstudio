import sys
import importlib
import importlib.util
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--channel", help="pytorch channel", type=str, default='pytorch')
parser.add_argument("--cuda", help="install nvidia gpu support", action="store_true", default=False)
parser.add_argument("--package", help="install specific package", action='append', nargs='+', default=[])
args, unknown = parser.parse_known_args()

if importlib.util.find_spec("conda") is None:
    print("Error: A Conda environment is required to install the required packages.", file=sys.stderr)
    exit()

import conda.cli.python_api as Conda

#increase rows (from default 20 when no terminal is found) to display all parallel packages downloads at once
from tqdm import tqdm
init_source=tqdm.__init__
def init_patch(self, **kwargs):
    kwargs['ncols']=80
    kwargs['nrows']=80
    init_source(self, **kwargs)
tqdm.__init__=init_patch

if not args.package:
    #https://edcarp.github.io/introduction-to-conda-for-data-scientists/03-using-packages-and-channels/index.html#alternative-syntax-for-installing-packages-from-specific-channels
    conda_install=f"{args.channel}::pytorch {args.channel}::torchvision {args.channel}::torchaudio {args.channel}::torchtext"
    if (sys.platform.startswith('win') or sys.platform.startswith('linux')):
        if args.cuda:
            print("Checking the latest supported CUDA version...")
            highest_cuda_version=(11,6) #highest supported cuda version for PyTorch 1.12
            import requests
            try:
                pytorch_repo = requests.get("https://anaconda.org/"+args.channel+"/pytorch/files")
            except:
                print("Could not retrieve the latest supported CUDA version")
            else:
                import re
                regex_request=re.compile("cuda([0-9]+.[0-9]+)")
                results = re.findall(regex_request, pytorch_repo.text)
                highest_cuda_version=(11,6)
                for cuda_string in results:
                    cuda_version=tuple(int(i) for i in cuda_string.split('.'))
                    if  cuda_version > highest_cuda_version:
                        highest_cuda_version = cuda_version
            highest_cuda_string='.'.join([str(value) for value in highest_cuda_version])
            print("Using CUDA "+highest_cuda_string)
            print("")
            conda_install+=f" {args.channel}::pytorch-cuda="+highest_cuda_string+" -c nvidia"
        else:
            conda_install+=f" {args.channel}::cpuonly"
    print(f"Downloading and installing {args.channel} packages...")
    print("")
    conda_install+=" -k" #allow insecure ssl connections
    # https://stackoverflow.com/questions/41767340/using-conda-install-within-a-python-script
    (stdout_str, stderr_str, return_code_int) = Conda.run_command(Conda.Commands.INSTALL,conda_install.split(),use_exception_handler=True,stdout=sys.stdout,stderr=sys.stderr)
    if return_code_int!=0:
        exit(return_code_int)
    print("")

    # datasets(+huggingface_hub) is required by hugging face hub
    # scipy required by torchvision: Caltech ImageNet SBD SVHN datasets and Inception v3 GoogLeNet models
    # pandas required by the dataset tutorial: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    # matplotlib-base required by torchstudio renderers
    # python-graphviz required by torchstudio graph
    # paramiko required for ssh connections (+updated cffi required on intel mac)
    # pysoundfile required by torchaudio datasets: https://pytorch.org/audio/stable/backend.html#soundfile-backend
    conda_install="datasets scipy pandas matplotlib-base python-graphviz paramiko pysoundfile"
    if sys.platform.startswith('darwin'):
        conda_install+=" cffi"

else:
    conda_install=" ".join(args.package[0])

print("Downloading and installing conda-forge packages...")
print("")
conda_install+=" -c conda-forge -k"
(stdout_str, stderr_str, return_code_int) = Conda.run_command(Conda.Commands.INSTALL,conda_install.split(),use_exception_handler=True,stdout=sys.stdout,stderr=sys.stderr)
if return_code_int!=0:
    exit(return_code_int)
