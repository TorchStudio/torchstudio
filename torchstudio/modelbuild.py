#workaround until Pytorch 1.12.1 is released: https://github.com/pytorch/pytorch/issues/78490
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import sys
print("Loading PyTorch...\n", file=sys.stderr)

import torch
import torch.fx
from torch.fx.passes.shape_prop import ShapeProp
from torch.fx.graph_module import GraphModule
import torchstudio.tcpcodec as tc
from torchstudio.modules import safe_exec
import sys
import os
import io
import re
import graphviz
import linecache
import inspect

#monkey patch ssl to fix ssl certificate fail when downloading datasets on some configurations: https://stackoverflow.com/questions/27835619/urllib-and-ssl-certificate-verify-failed-error
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

original_path=sys.path
original_dir=os.getcwd()

level=0
max_depth=0

app_socket = tc.connect()
print("Build script connected\n", file=sys.stderr)
while True:
    msg_type, msg_data = tc.recv_msg(app_socket)

    if msg_type == 'SetCurrentDir':
        new_dir=tc.decode_strings(msg_data)[0]
        sys.path=original_path
        os.chdir(original_dir)
        if new_dir:
            sys.path.append(new_dir)
            os.chdir(new_dir)

    if msg_type == 'SetDataDir':
        data_dir=tc.decode_strings(msg_data)[0]
        torch.hub.set_dir(data_dir)

    if msg_type == 'SetModelCode':
        model_code=tc.decode_strings(msg_data)[0]

        #create a module space for the model definition
        #see https://stackoverflow.com/questions/5122465/can-i-fake-a-package-or-at-least-a-module-in-python-for-testing-purposes/27476659#27476659
        from types import ModuleType
        modelmodule = ModuleType("modelmodule")
        modelmodule.__file__ = modelmodule.__name__ + ".py"
        sys.modules[modelmodule.__name__] = modelmodule

        error_msg, model_env = safe_exec(model_code, context=vars(modelmodule), output=vars(modelmodule), description='model definition')
        if error_msg is not None or 'model' not in model_env:
            print("Unknown model definition error" if error_msg is None else error_msg, file=sys.stderr)

    if msg_type == 'InputTensorsID':
        input_tensors = tc.decode_torch_tensors(msg_data)
        for i, tensor in enumerate(input_tensors):
            input_tensors[i]=torch.unsqueeze(tensor, 0) #add batch dimension

    if msg_type == 'OutputTensorsID':
        output_tensors = tc.decode_torch_tensors(msg_data)
        for i, tensor in enumerate(output_tensors):
            output_tensors[i]=torch.unsqueeze(tensor, 0) #add batch dimension

    if msg_type == 'SetLabels':
        labels=tc.decode_strings(msg_data)

    if msg_type == 'Build': #generate the torchscript, graph, and suggest hyperparameters
        if 'model' in model_env and input_tensors and output_tensors:
            print("Building model...\n", file=sys.stderr)

            build_mode=tc.decode_strings(msg_data)[0]

            buffer=io.BytesIO()
            torchscript_model=None
            if build_mode=='package': #packaging
                with torch.package.PackageExporter(buffer) as exp:
                    intern_list=[]
                    for path in os.listdir():
                        if path.endswith(".py") and os.path.isfile(path):
                            intern_list.append(path[:-3]+".**")
                        if os.path.isdir(path):
                            intern_list.append(path+".**")
                    exp.extern('**',exclude=intern_list)
                    exp.intern(intern_list)
                    exp.save_source_string(modelmodule.__name__, model_code)
                    exp.save_pickle('model', 'model.pkl', modelmodule.model)
            elif build_mode=='script': #scripting
                #monkey patch linecache.getlines so that inspect.getsource called by torch.jit.script can work with a module coming from a string and not a file
                def monkey_patch(filename, module_globals=None):
                    if filename == '<string>':
                        return model_code.splitlines(keepends=True)
                    else:
                        return getlines(filename, module_globals)
                getlines = linecache.getlines
                linecache.getlines = monkey_patch
                error_msg, torchscript_model = safe_exec(torch.jit.script,{'obj':modelmodule.model}, description='model scripting')
                linecache.getlines = getlines
            else: #tracing
                error_msg, torchscript_model = safe_exec(torch.jit.trace,{'func':modelmodule.model, 'example_inputs':input_tensors, 'check_trace':False}, description='model tracing')

            if error_msg:
                print(error_msg, file=sys.stderr)
            else:
                if torchscript_model:
                    torch.jit.save(torchscript_model,buffer)
                    tc.send_msg(app_socket, 'TorchScriptData', buffer.getvalue())
                else:
                    tc.send_msg(app_socket, 'PackageData', buffer.getvalue())

                print("Building graph...\n", file=sys.stderr)

                level=0
                max_depth=0

                while level<=max_depth:
                    class LevelTracer(torch.fx.Tracer):
                        def is_leaf_module(self, m, qualname):
                            depth=re.sub(r'.[0-9]+', '', qualname).count('.')
                            if super().is_leaf_module(m, qualname)==False:
                                depth=depth+1
                            global max_depth
                            max_depth=max(max_depth,depth)
                            if depth>max_depth-level:
                                return True
                            else:
                                return super().is_leaf_module(m, qualname)

                    def level_trace(root):
                        tracer = LevelTracer()
                        graph = tracer.trace(root)
                        name = root.__class__.__name__ if isinstance(root, torch.nn.Module) else root.__name__
                        return GraphModule(tracer.root, graph, name)

                    error_msg, gm = safe_exec(level_trace,(model_env['model'],), description='model graph')
                    if error_msg or gm is None:
                        print("Unknown model graph error" if error_msg is None else error_msg, file=sys.stderr)
                    else:
                        modules = dict(gm.named_modules())
                        ShapeProp(gm).propagate(*input_tensors)

                        parsed_nodes={}
                        for rank, node in enumerate(gm.graph.nodes):
                            id=node.name
                            name=node.name
                            inputs=[str(i) for i in list(node.all_input_nodes)]
                            output_dtype=''
                            output_shape=''
                            if 'tensor_meta' in node.meta:
                                if type(node.meta['tensor_meta']) is tuple or type(node.meta['tensor_meta']) is list:
                                    for tensor_meta in node.meta['tensor_meta']:
                                        output_dtype+=str(tensor_meta.dtype)+' '
                                        output_shape+=','.join([str(i) for i in list(tensor_meta.shape)[1:]])+' '
                                    output_dtype=output_dtype[:-1]
                                    output_shape=output_shape[:-1]
                                else:
                                    output_dtype=str(node.meta['tensor_meta'].dtype)
                                    output_shape=','.join([str(i) for i in list(node.meta['tensor_meta'].shape)[1:]])

                            if node.op == 'placeholder':
                                node_type='input'
                                op_module=''
                                op=''
                                params=''
                            elif node.op == 'call_module':
                                node_type='module'
                                name=re.sub('\.([0-9]+)', r'[\1]', node.target)
                                op_module=modules[node.target].__module__
                                op_module='torch.nn' #prefer this shortcut for all modules
                                op=modules[node.target].__class__.__name__
                                params=modules[node.target].extra_repr()
                            elif node.op == 'call_function':
                                node_type='function'
                                op_module=node.target.__module__ if node.target.__module__ is not None else "torch"
                                op_module='operator' if op_module=='_operator' else op_module
                                op=node.target.__name__
                                params_list=[str(x) for x in node.args]
                                params_list.extend([f'{key}={value}' for key, value in node._kwargs.items()])
                                params=', '.join(params_list)
                            elif node.op == 'call_method':
                                node_type='function'
                                op_module="torch"
                                op=node.target
                                params_list=[str(x) for x in node.args]
                                params_list.extend([f'{key}={value}' for key, value in node._kwargs.items()])
                                params=', '.join(params_list)
                            elif node.op == 'output':
                                node_type='output'
                                op_module=''
                                op=''
                                params=''
                                for input_node in node.all_input_nodes:
                                    input_op=""
                                    if input_node.op == 'call_module':
                                        input_op=modules[input_node.target].__class__.__name__
                                    elif input_node.op == 'call_function':
                                        input_op=input_node.target.__name__
                            else:
                                node_type='unknown'
                                op_module=''
                                op=''
                                params=''

                            if node_type=='output' and len(inputs)>1:
                                for i, input in enumerate(inputs):
                                    parsed_nodes[id+"_"+str(i)]={'name':name+"["+str(i)+"]", 'type':node_type, 'op_module':op_module, 'op':op, 'params':params, 'output_dtype':output_dtype, 'output_shape':output_shape, 'inputs':inputs}
                            else:
                                parsed_nodes[id]={'name':name, 'type':node_type, 'op_module':op_module, 'op':op, 'params':params, 'output_dtype':output_dtype, 'output_shape':output_shape, 'inputs':inputs}
                        tc.send_msg(app_socket, 'GraphData', bytes(str(parsed_nodes),'utf-8'))
                        level+=1
                tc.send_msg(app_socket, 'GraphDataEnd')

                print("Model built ("+format(sum(p.numel() for p in model_env['model'].parameters() if p.requires_grad), ',d')+" parameters)") #from https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9?u=robin_lobel

                #suggest loss names
                loss=[]
                for i, tensor in enumerate(output_tensors):
                    if len(tensor.shape)==1 and "int" in str(tensor.dtype):
                        #multiclass crossentropy classification
                        loss.append("CrossEntropy")
                    elif len(tensor.shape)==2 and tensor.shape[1]==len(labels):
                        #multiclass multilabel classification
                        loss.append("BinaryCrossEntropy")
                    else:
                        #default back to MSE for everything else
                        loss.append("MeanSquareError")

                #suggest metric names
                metric=[]
                for tensor in output_tensors:
                    metric.append("Accuracy")

                tc.send_msg(app_socket, 'SetHyperParametersValues', tc.encode_ints([64,0,100,20]))
                tc.send_msg(app_socket, 'SetHyperParametersNames', tc.encode_strings(loss+metric+['Adam','Step']))

    if msg_type == 'Exit':
        break

