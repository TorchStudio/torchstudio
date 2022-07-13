import sys
import importlib
import importlib.util
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--base", help="install base packages", action="store_true", default=False)
parser.add_argument("--gpu", help="install nvidia gpu support", action="store_true", default=False)
parser.add_argument("--package", help="install specific package", action='append', nargs='+', default=[])
args, unknown = parser.parse_known_args()

if importlib.util.find_spec("conda") is None:
    print("Error: A Conda environment is required to install the required packages.", file=sys.stderr)
    exit()

import conda.cli.python_api as Conda

conda_install=""
if args.base:
    conda_install="pytorch torchvision torchaudio torchtext"
    if (sys.platform.startswith('win') or sys.platform.startswith('linux')):
        if args.gpu:
            conda_install+=" cudatoolkit"
        else:
            conda_install+=" cpuonly"
    print("Downloading and installing pytorch packages:")
    print(conda_install)
    print("")
    conda_install+=" -c pytorch  -k"
    # https://stackoverflow.com/questions/41767340/using-conda-install-within-a-python-script
    (stdout_str, stderr_str, return_code_int) = Conda.run_command(Conda.Commands.INSTALL,conda_install.split(),stdout=sys.stdout,stderr=sys.stderr)
    print("")

    # scipy required by torchvision: Caltech ImageNet SBD SVHN datasets and Inception v3 GoogLeNet models
    # pandas required by the dataset tutorial: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    # matplotlib-base required by torchstudio renderers
    # python-graphviz required by torchstudio graph
    # paramiko required for ssh connections (+updated cffi required on intel mac)
    # pysoundfile required by torchaudio datasets: https://pytorch.org/audio/stable/backend.html#soundfile-backend
    # datasets(+huggingface_hub) is required by hugging face hub
    conda_install="scipy pandas matplotlib-base python-graphviz paramiko pysoundfile datasets"
    if sys.platform.startswith('darwin'):
        conda_install+=" cffi"

if args.package:
    if args.base:
        conda_install+=" "
    conda_install+=" ".join(args.package[0])

print("Downloading and installing conda-forge packages:")
print(conda_install)
print("")
conda_install+=" -c conda-forge -k"

# https://stackoverflow.com/questions/41767340/using-conda-install-within-a-python-script
(stdout_str, stderr_str, return_code_int) = Conda.run_command(Conda.Commands.INSTALL,conda_install.split(),stdout=sys.stdout,stderr=sys.stderr)
