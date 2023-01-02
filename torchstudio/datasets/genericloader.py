import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision
import torchaudio
import numpy as np
import sys

class GenericLoader(Dataset):
    """A generic dataset loader.
    Suitable for classification, segmentation and regression datasets.
    Supports image, audio, and numpy array files.

    Args:
        path (str):
            path to the dataset

        classification (bool):
            True: classification dataset (single class prediction: class1, class2, ...)
            False: segmentation or regression dataset (multiple components: input, target, ...)

        separator (str or None):
            '/': folders will be used to determine classes or components
                (classes: class1/1.ext, class1/2.ext, class2/1.ext, class2/2.ext, ...)
                (components: inputs/1.ext, inputs/2.ext, targets/1.ext, targets/2.ext, ...)

            '_' or other separator: file name parts will be used to determine classes or components
                (classes: class1_1.ext, class1_2.ext, class2_1.ext, class2_2.ext, ...)
                (components: 1_input.ext, 1_output.ext, 2_input.ext, 2_output.ext, ...)

            '' or None: file names or their content will be used to determine components
                (one sample per folder: 1/input.ext, 1/output.ext, 2/input.ext, 2/output.ext, ...)
                (samples in one folder: 1.ext, 2.ext, ...)

        extensions (str):
            file extension to filters (such as: .jpg, .jpeg, .png, .mp3, .wav, .npy, .npz)

        transforms (list):
            list of transforms to apply to the different components of each sample (use None is some components need no transform)
            (ie: [torchvision.transforms.Compose([transforms.Resize(64)]), torchaudio.transforms.Spectrogram()])
    """

    def __init__(self, path:str='', classification:bool=True, separator:str='/', extensions:str='.jpg, .jpeg, .png, .mp3, .wav, .npy, .npz', transforms=[]):
        exts = tuple(extensions.replace(' ','').split(','))
        paths = []
        self.samples = []
        self.classes = []
        self.transforms = transforms
        if not os.path.exists(path):
            print("Path not found.", file=sys.stderr)
            return
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(exts):
                    paths.append(os.path.join(root, file).replace('\\','/'))
        paths=sorted(paths)
        if not paths:
            print("No files found.", file=sys.stderr)
            return
        self.classification=classification
        if classification:
            if separator == '/':
                for path in paths:
                    class_name=path.split('/')[-2]
                    if class_name not in self.classes:
                        self.classes.append(class_name)
                    self.samples.append([path, self.classes.index(class_name)])
            elif separator:
                for path in paths:
                    class_name = path.split('/')[-1].split(separator)[0]
                    if class_name not in self.classes:
                        self.classes.append(class_name)
                    self.samples.append([path, self.classes.index(class_name)])
            else:
                print("You need a separator with classication datasets", file=sys.stderr)
                return
        else:
            samples_index = dict()
            if separator == '/':
                for path in paths:
                    components_name=path.split('/')[-2]
                    sample_name = path.split('/')[-1].split('.')[-2]
                    if sample_name not in samples_index:
                        samples_index[sample_name] = len(self.samples)
                        self.samples.append([])
                    self.samples[samples_index[sample_name]].append(path)
            elif separator:
                for path in paths:
                    components_name = path.split('.')[-2].split(separator)[-1]
                    sample_name = path.split('/')[-1].split(separator)[0]
                    if sample_name not in samples_index:
                        samples_index[sample_name] = len(self.samples)
                        self.samples.append([])
                    self.samples[samples_index[sample_name]].append(path)
            else:
                single_folder=True
                file_root=path[:path.rfind("/")]
                for path in paths:
                    if not path.startswith(file_root):
                        single_folder=False
                        break
                if single_folder:
                    for path in paths:
                        sample_name = path.split('/')[-1].split('.')[-2]
                        if sample_name not in samples_index:
                            samples_index[sample_name] = len(self.samples)
                            self.samples.append([])
                        self.samples[samples_index[sample_name]].append(path)
                else:
                    for path in paths:
                        components_name = path.split('/')[-1].split('.')[-2]
                        sample_name = path.split('/')[-2]
                        if sample_name not in samples_index:
                            samples_index[sample_name] = len(self.samples)
                            self.samples.append([])
                        self.samples[samples_index[sample_name]].append(path)

    def to_tensors(self, path:str):
        if path.endswith('.jpg') or path.endswith('.jpeg') or path.endswith('.png'):
            img=Image.open(path)
            if img.getpalette():
                return [torch.from_numpy(np.array(img, dtype=np.uint8))]
            else:
                trans=torchvision.transforms.ToTensor()
                return [trans(img)]

        if path.endswith('.mp3') or path.endswith('.wav'):
            waveform, sample_rate = torchaudio.load(path)
            return [waveform]

        if path.endswith('.npy') or path.endswith('.npz'):
            arrays = np.load(path)
            if type(arrays) == dict:
                tensors = []
                for array in arrays:
                    tensors.append(torch.from_numpy(arrays[array]))
                return tensors
            else:
                return [torch.from_numpy(arrays)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, id):
        """
        Returns:
            A tuple of tensors.
        """

        if id < 0 or id >= len(self):
            raise IndexError

        components = []
        for component in self.samples[id]:
            if type(component) is str:
                components.extend(self.to_tensors(component))
            else:
                components.extend([torch.tensor(component)])

        if self.transforms:
            if type(self.transforms) is not list and type(self.transforms) is not tuple:
                self.transforms = [self.transforms]
            for i, transform in enumerate(self.transforms):
                if i < len(components) and transform is not None:
                    components[i] = transform(components[i])

        return tuple(components)
