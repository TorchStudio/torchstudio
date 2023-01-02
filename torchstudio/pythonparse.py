#workaround until Pytorch 1.12.1 is released: https://github.com/pytorch/pytorch/issues/78490
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import importlib
import inspect, sys
import ast
import re
from typing import Dict, List
from os import listdir
from os.path import isfile, join
import torchstudio.tcpcodec as tc
from torchstudio.modules import safe_exec

def gather_parameters(node):
    params=[]
    for param in inspect.signature(node).parameters.values():
        #name
        if param.kind==param.VAR_POSITIONAL:
            params.append("*"+param.name)
        elif param.kind==param.VAR_KEYWORD:
            params.append("**"+param.name)
        else:
            params.append(param.name)
        #annotation
        if param.annotation == param.empty:
            params.append('')
        else:
            params.append(param.annotation.__name__ if isinstance(param.annotation, type) else repr(param.annotation))
        #default value
        if param.default == param.empty:
            params.append('')
        elif inspect.isclass(param.default) or inspect.isfunction(param.default):
            params.append(param.default.__module__+'.'+param.default.__name__)
        else:
            value=repr(param.default)
            if "<class '" in value:
                value=value.replace("<class '","")
                value=value.replace("'>","")
            params.append(value)
    return params

def gather_objects(module):
    objects=[]
    for name, obj in inspect.getmembers(module):
        if (inspect.isclass(obj) and hasattr(obj, '__mro__') and ("torch.nn.modules.module.Module" in str(obj.__mro__) or "torch.utils.data.dataset.Dataset" in str(obj.__mro__))) or (inspect.isfunction(obj) and "return" in obj.__annotations__ and inspect.isclass(obj.__annotations__["return"]) and "torch.nn.modules.module.Module" in str(obj.__annotations__["return"].__mro__)): #filter unwanted torch objects
            object={}
            object['type']='class' if inspect.isclass(obj) else 'function'
            object['name']=name
            if obj.__doc__ is not None:
                object['doc']=inspect.cleandoc(obj.__doc__)

                # autofill class members when requested
                doc=object['doc']
                newstring = ''
                start = 0
                for m in re.finditer(".. autoclass:: [a-zA-Z0-9_.]+\n    :members:", doc):
                    end, newstart = m.span()
                    newstring += doc[start:end]
                    class_name = re.findall(".. autoclass:: ([a-zA-Z0-9_.]+)\n    :members:", m.group(0))
                    rep = class_name[0]+":\n"
                    sub_error_message, submodule = safe_exec(importlib.import_module,(class_name[0].rpartition('.')[0],))
                    if submodule is not None:
                        for member in dir(vars(submodule)[class_name[0].rpartition('.')[-1]]):
                            if not member.startswith('_'):
                                rep+='    '+member+'\n'
                    newstring += rep
                    start = newstart
                newstring += doc[start:]
                object['doc']=newstring.replace("    :noindex:","")
            else:
                object['doc']=name+(' class' if inspect.isclass(obj) else ' function')
            if hasattr(obj,'__getitem__') and obj.__getitem__.__doc__ is not None:
                itemdoc=inspect.cleandoc(obj.__getitem__.__doc__)
                if 'Returns:' in itemdoc:
                    object['doc']+='\n\n'+itemdoc[itemdoc.find('Returns:'):]
            object['params']=gather_parameters(obj.__init__ if inspect.isclass(obj) else obj)
            if inspect.isclass(obj):
                object['params']=object['params'][3:] #remove self parameter
            object['code']=''
            objects.append(object)
    return objects


def parse_parameters(node):
    #prepare defaults to be in sync with arguments
    defaults=[]
    for d in node.args.defaults:
        defaults.append(ast.get_source_segment(code,d))
    for d in range(len(node.args.args)-len(node.args.defaults)):
        defaults.insert(0,"")
    #scan through arguments
    params=[]
    for i,a in enumerate(node.args.args):
        params.append(a.arg)
        if a.annotation:
            params.append(ast.get_source_segment(code,a.annotation))
        else:
            params.append("")
        params.append(defaults[i])
    #add *args, if applicable
    if node.args.vararg:
        params.append("*"+node.args.vararg.arg)
        if node.args.vararg.annotation:
            params.append(ast.get_source_segment(code,node.args.vararg.annotation))
        else:
            params.append("")
        params.append("") #no default value
    #add **kwargs, if applicable
    if node.args.kwarg:
        params.append("**"+node.args.kwarg.arg)
        if node.args.kwarg.annotation:
            params.append(ast.get_source_segment(code,node.args.kwarg.annotation))
        else:
            params.append("")
        params.append("") #no default value
    return params

def parse_objects(module:ast.Module):
    objects=[]
    for node in module.body:
        if isinstance(node, ast.FunctionDef):
            object={}
            object['code']=ast.get_source_segment(code,node)
            object['type']='function'
            object['name']=node.name
            object['doc']=ast.get_docstring(node) if ast.get_docstring(node) else ""
            object['params']=parse_parameters(node)
            objects.append(object)
        if isinstance(node, ast.ClassDef):
            object={}
            object['code']=ast.get_source_segment(code,node)
            object['type']='class'
            object['name']=node.name
            object['doc']=ast.get_docstring(node) if ast.get_docstring(node) else ""
            object['params']=[]
            for subnode in node.body:
                if isinstance(subnode, ast.FunctionDef) and subnode.name=="__init__":
                    object['params']=parse_parameters(subnode)
            object['params']=object['params'][3:] #remove self parameter
            objects.append(object)
    return objects

def filter_parent_objects(objects:List[Dict]) -> List[Dict]:
    parent_objects=[]
    for object in objects:
        unique=True
        for subobject in objects:
            name=object['name']
            if subobject['name']!=name:
                if re.search('[ =+]'+name+'[ ]*\(', subobject['code']):
                    unique=False
        if unique:
            parent_objects.append(object)
    return parent_objects


generated_class="""\
import typing
import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import {0}
from {0} import transforms

class {1}({2}):
    \"\"\"{3}\"\"\"
    def __init__({4}):
        super().__init__({5})
"""

generated_function="""\
import typing
import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import {0}
from {0} import transforms

def {1}({2}):
    \"\"\"{3}\"\"\"
    {4}={5}({6})
    return {4}
"""

def generate_code(path,object):
    #write an inherited code code for each object
    name=object['name']
    params=object['params']
    if object['type']=='class':
        return generated_class.format(
            path.split('.')[0],
            name, path+'.'+name,
            object['doc'],
            ', '.join(['self']+[params[i]+(': ' if params[i+1] else'')+params[i+1]+(' = ' if params[i+2] else'')+params[i+2] for i in range(0,len(params),3)]),
            ', '.join([params[i] for i in range(0,len(params),3)])
            )
    else:
        return generated_function.format(
            path.split('.')[0],
            name, ', '.join([params[i]+(': ' if params[i+1] else'')+params[i+1]+(' = ' if params[i+2] else'')+params[i+2] for i in range(0,len(params),3)]),
            object['doc'],
            path.split('.')[1][:-1], path+'.'+name, ', '.join([params[i] for i in range(0,len(params),3)])
            )

def patch_parameters(path, name, params):
    patched_params=[]
    if 'datasets' in path:
        for state in ['train','valid']:
            for i in range(0,len(params)-1,3):
                #patch root for all modules
                if params[i]=='root' and not params[i+2]:
                    params[i+2]="'"+data_path+"'"

                #patch download for all modules
                if params[i]=="download" and params[i+2]=="False":
                    params[i+2]="True"

                #patch transform for vision modules
                if 'torchvision' in path and (params[i]=="transform" or params[i]=="target_transform"):
                    params[i+2]="transforms.Compose([])"

                #patch train/val for specific modules
                if state=='valid':
                    if params[i]=="train" and params[i+2]=="True":
                        params[i+2]="False"
                    if 'torchvision' in path:
                        if (name=="Cityscapes" or name=="ImageNet") and params[i]=="split":
                            params[i+2]="'val'"
                        if (name=="STL10" or name=="SVHN") and params[i]=="split":
                            params[i+2]="'test'"
                        if (name=="CelebA") and params[i]=="split":
                            params[i+2]="'valid'"
                        if (name=="Places365") and params[i]=="split":
                            params[i+2]="'val'"
                        if (name=="VOCDetection" or name=="VOCSegmentation") and params[i]=="image_set":
                            params[i+2]="'val'"

            patched_params.append(params.copy())
    else:
        patched_params.append(params)
    return patched_params

custom_classes={}
custom_classes['Custom Dataset']="""\
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
"""
custom_classes['Custom Renderer']="""\
from torchstudio.modules import Renderer
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import PIL

class MyRenderer(Renderer):
    def __init__(self):
        super().__init__()

    def render(self, title, tensor, size, dpi, shift=(0,0,0,0), scale=(1,1,1,1), input_tensors=[], target_tensor=None, labels=[]):
        pass
"""
custom_classes['Custom Analyzer']="""\
from torchstudio.modules import Analyzer
from typing import List
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import PIL

class MyAnalyzer(Analyzer):
    def __init__(self):
        super().__init__()

    def start_analysis(self, num_training_samples: int, num_validation_samples: int, input_tensors_id: List[int], output_tensors_id: List[int], labels: List[str]):
        pass

    def analyze_sample(self, sample: List[np.array], training_sample: bool):
        pass

    def finish_analysis(self):
        pass

    def generate_report(self, size: Tuple[int, int], dpi: int):
        pass
"""
custom_classes['Custom Model']="""\
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass
"""
custom_classes['Custom Loss']="""\
import torch.nn as nn

class MyLoss(nn.Modules._Loss):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass
"""
custom_classes['Custom Metric']="""\
from torchstudio.modules import Metric
import torch.nn.functional as F

class MyMetric(Metric):
    def __init__(self):
        pass

    def update(self, preds, target):
        pass

    def compute(self):
        pass

    def reset(self):
        pass
"""
custom_classes['Custom Optimizer']="""\
import torch.optim as optim

class MyOptimizer(optim.Optimizer):
    def __init__(self, params):
        super().__init__(params)
"""
custom_classes['Custom Scheduler']="""\
import torch.optim as optim

class MyScheduler(optim._LRScheduler):
    def __init__(self, optimizer):
        super().__init__(optimizer)
"""

def scan_folder(path):
    path=path.replace('.','/')
    codes=[]
    for filename in sorted(listdir(path)):
        if isfile(join(path, filename)):
            with open(join(path, filename), "r") as file:
                codes.append(file.read())
    return codes


app_socket = tc.connect()
while True:
    msg_type, msg_data = tc.recv_msg(app_socket)
    objects=[]
    if msg_type == 'SetDataDir':
        data_path=tc.decode_strings(msg_data)[0]

    if msg_type == 'Parse': #parse code or path, return a list of objects (class and functions) with their names, doc, parameters, doc and code
        decoded=tc.decode_strings(msg_data)
        path=decoded[0]
        if path in custom_classes:
            decoded.append(custom_classes[path])
        if 'torchstudio' in path:
            decoded+=scan_folder(path)
        if len(decoded)>1:
            #parse code chunks
            for code in decoded[1:]:
                error_msg, module = safe_exec(ast.parse,(code,))
                if error_msg is None and module is not None:
                    objects_batch=parse_objects(module)
                    objects_batch=filter_parent_objects(objects_batch) #only keep parent objets
                    for i in range(len(objects_batch)):
                        objects_batch[i]['code']=code #set whole source code for each object, as we don't know the dependencies
                    objects.extend(objects_batch)
                else:
                    print("Error parsing code:", error_msg, "\n", file=sys.stderr)
        else:
            #parse module
            error_msg, module = safe_exec(importlib.import_module,(path,))
            if error_msg is None and module is not None:
                objects=gather_objects(module)
                for i, object in enumerate(objects):
                    objects[i]['code']=generate_code(path,object) #generate inherited source code
            else:
                print("Error parsing module:", error_msg, "\n", file=sys.stderr)

        tc.send_msg(app_socket, 'ObjectsBegin', tc.encode_strings(path))
        for object in objects:
            patched_params = patch_parameters(path,object['name'],object['params'])
            for params in patched_params:
                tc.send_msg(app_socket, 'Object', tc.encode_strings([path,object['code'],object['type'],object['name'],object['doc']]+params))
        tc.send_msg(app_socket, 'ObjectsEnd', tc.encode_strings(path))

    if msg_type == 'RequestDefinitionName': #return default definition name
        tab=tc.decode_strings(msg_data)[0]
        if tab=='dataset':
            tc.send_msg(app_socket, 'SetDefinitionName', tc.encode_strings(['torchvision.datasets','MNIST']))
        if tab=='model':
            tc.send_msg(app_socket, 'SetDefinitionName', tc.encode_strings(['torchstudio.models','MNISTClassifier']))

    if msg_type == 'Exit':
        break
