import torch
from torch.utils.data import Dataset
import inspect

class RandomGenerator(Dataset):
    """A random generator that returns randomly generated tensors

    Args:
        size (int):
            Size of the dataset (number of samples)
        tensors:
            A list of tuples defining tensor properties: shape, type, range
            All properties are optionals. Defaults are null, torch.float, [0,1]
    """

    def __init__(self, size:int=256, tensors=[(3,64,64), (int,[0,9])]):
        torch.manual_seed(0)
        self.size = size
        self.tensors = tensors

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        """
        Returns:
            A tuple of tensors.
        """
        sample = []
        for properties in self.tensors:
            shape=[]
            dtype=torch.float
            drange=[0,1]
            for property in properties:
                if type(property)==int:
                    shape.append(property)
                elif type(property)==type or type(property)==torch.dtype:
                    dtype=property
                elif type(property) is list:
                    drange=property
            shape=tuple(shape)

            if 'int' in str(dtype):
                tensor=torch.randint(low=drange[0], high=drange[1]+1, size=shape, dtype=dtype)
            else:
                tensor=torch.rand(size=shape,dtype=dtype)*(drange[1]-drange[0])+drange[0]

            sample.append(tensor)

        return tuple(sample)
