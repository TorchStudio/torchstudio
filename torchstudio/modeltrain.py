#workaround until Pytorch 1.12.1 is released: https://github.com/pytorch/pytorch/issues/78490
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import sys

print("Loading PyTorch...\n", file=sys.stderr)

import torch
from torch.utils.data import Dataset
import torchstudio.tcpcodec as tc
from torchstudio.modules import safe_exec
import os
import sys
import io
import tempfile
from tqdm.auto import tqdm
from collections.abc import Iterable


class CachedDataset(Dataset):
    def __init__(self, disk_cache=False):
        self.reset(disk_cache)

    def add_sample(self, data):
        if self.disk_cache:
            file=tempfile.TemporaryFile(prefix='torchstudio.'+str(len(self.index))+'.') #guaranteed to be deleted on win/mac/linux: https://bugs.python.org/issue4928
            file.write(data)
            self.index.append(file)
        else:
            self.index.append(tc.decode_torch_tensors(data))

    def reset(self, disk_cache=False):
        self.index = []
        self.disk_cache=disk_cache

    def __len__(self):
        return len(self.index)

    def __getitem__(self, id):
        if id<0 or id>=len(self):
            raise IndexError

        if self.disk_cache:
            file=self.index[id]
            file.seek(0)
            sample=tc.decode_torch_tensors(file.read())
        else:
            sample=self.index[id]
        return sample

def deepcopy_cpu(value):
    if isinstance(value, torch.Tensor):
        value = value.to("cpu")
        return value
    elif isinstance(value, dict):
        return {k: deepcopy_cpu(v) for k, v in value.items()}
    elif isinstance(value, Iterable):
        return type(value)(deepcopy_cpu(v) for v in value)
    else:
        return value

modules_valid=True

train_dataset = CachedDataset()
valid_dataset = CachedDataset()
train_bar = None

model = None

app_socket = tc.connect()
print("Training script connected\n", file=sys.stderr)
while True:
    msg_type, msg_data = tc.recv_msg(app_socket)

    if msg_type == 'SetDevice':
        print("Setting device...\n", file=sys.stderr)
        device_id=tc.decode_strings(msg_data)[0]
        device = torch.device(device_id)
        pin_memory = True if 'cuda' in device_id else False

    if msg_type == 'SetTorchScriptModel' and modules_valid:
        if msg_data:
            print("Setting torchscript model...\n", file=sys.stderr)
            buffer=io.BytesIO(msg_data)
            model = torch.jit.load(buffer)

    if msg_type == 'SetPackageModel' and modules_valid:
        if msg_data:
            print("Setting package model...\n", file=sys.stderr)
            buffer=io.BytesIO(msg_data)
            model = torch.package.PackageImporter(buffer).load_pickle('model', 'model.pkl')

    if msg_type == 'SetModelState' and modules_valid:
        if model is not None:
            if msg_data:
                buffer=io.BytesIO(msg_data)
                model.load_state_dict(torch.load(buffer))
            model.to(device)

    if msg_type == 'SetLossCodes' and modules_valid:
        print("Setting loss code...\n", file=sys.stderr)
        loss_definitions=tc.decode_strings(msg_data)
        criteria = []
        for definition in loss_definitions:
            error_msg, loss_env = safe_exec(definition, description='loss definition')
            if error_msg is not None or 'loss' not in loss_env:
                print("Unknown loss definition error" if error_msg is None else error_msg, file=sys.stderr)
                modules_valid=False
                tc.send_msg(app_socket, 'TrainingError')
                break
            else:
                criteria.append(loss_env['loss'])

    if msg_type == 'SetMetricCodes' and modules_valid:
        print("Setting metrics code...\n", file=sys.stderr)
        metric_definitions=tc.decode_strings(msg_data)
        metrics = []
        for definition in metric_definitions:
            error_msg, metric_env = safe_exec(definition, description='metric definition')
            if error_msg is not None or 'metric' not in metric_env:
                print("Unknown metric definition error" if error_msg is None else error_msg, file=sys.stderr)
                modules_valid=False
                tc.send_msg(app_socket, 'TrainingError')
                break
            else:
                metrics.append(metric_env['metric'])

    if msg_type == 'SetOptimizerCode' and modules_valid:
        print("Setting optimizer code...\n", file=sys.stderr)
        error_msg, optimizer_env = safe_exec(tc.decode_strings(msg_data)[0], context=globals(), description='optimizer definition')
        if error_msg is not None or 'optimizer' not in optimizer_env:
            print("Unknown optimizer definition error" if error_msg is None else error_msg, file=sys.stderr)
            modules_valid=False
            tc.send_msg(app_socket, 'TrainingError')
        else:
            optimizer = optimizer_env['optimizer']

    if msg_type == 'SetOptimizerState' and modules_valid:
        if msg_data:
            buffer=io.BytesIO(msg_data)
            optimizer.load_state_dict(torch.load(buffer))

    if msg_type == 'SetSchedulerCode' and modules_valid:
        print("Setting scheduler code...\n", file=sys.stderr)
        error_msg, scheduler_env = safe_exec(tc.decode_strings(msg_data)[0], context=globals(), description='scheduler definition')
        if error_msg is not None or 'scheduler' not in scheduler_env:
            print("Unknown scheduler definition error" if error_msg is None else error_msg, file=sys.stderr)
            modules_valid=False
            tc.send_msg(app_socket, 'TrainingError')
        else:
            scheduler = scheduler_env['scheduler']

    if msg_type == 'SetHyperParametersValues' and modules_valid: #set other hyperparameters values
        batch_size, shuffle, epochs, early_stop, restore_best = tc.decode_ints(msg_data)
        shuffle=True if shuffle==1 else False
        early_stop=True if early_stop==1 else False
        restore_best=True if restore_best==1 else False

    if msg_type == 'StartTrainingServer' and modules_valid:
        print("Caching...\n", file=sys.stderr)

        sshaddress, sshport, username, password, keydata = tc.decode_strings(msg_data)

        training_server, address = tc.generate_server()

        if sshaddress and sshport and username:
            import socket
            import paramiko
            import torchstudio.sshtunnel as sshtunnel

            if not password:
                password=None
            if not keydata:
                pkey=None
            else:
                import io
                keybuffer=io.StringIO(keydata)
                pkey=paramiko.RSAKey.from_private_key(keybuffer)

            sshclient = paramiko.SSHClient()
            sshclient.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            sshclient.connect(hostname=sshaddress, port=int(sshport), username=username, password=password, pkey=pkey, timeout=5)

            reverse_tunnel = sshtunnel.Tunnel(sshclient, sshtunnel.ReverseTunnel, 'localhost', 0, 'localhost', int(address[1]))
            address[1]=str(reverse_tunnel.lport)

        tc.send_msg(app_socket, 'TrainingServerRequestingAllSamples', tc.encode_strings(address))
        dataset_socket=tc.start_server(training_server)
        train_dataset.reset()
        valid_dataset.reset()

        while True:
            dataset_msg_type, dataset_msg_data = tc.recv_msg(dataset_socket)

            if dataset_msg_type == 'NumSamples':
                num_samples=tc.decode_ints(dataset_msg_data)[0]
                pbar=tqdm(total=num_samples, desc='Caching...', bar_format='{l_bar}{bar}| {remaining} left\n\n') #see https://github.com/tqdm/tqdm#parameters

            if dataset_msg_type == 'InputTensorsID' and modules_valid:
                input_tensors_id = tc.decode_ints(dataset_msg_data)

            if dataset_msg_type == 'OutputTensorsID' and modules_valid:
                output_tensors_id = tc.decode_ints(dataset_msg_data)

            if dataset_msg_type == 'TrainingSample':
                train_dataset.add_sample(dataset_msg_data)
                pbar.update(1)

            if dataset_msg_type == 'ValidationSample':
                valid_dataset.add_sample(dataset_msg_data)
                pbar.update(1)

            if dataset_msg_type == 'DoneSending':
                pbar.close()
                tc.send_msg(dataset_socket, 'DoneReceiving')
                dataset_socket.close()
                training_server.close()
                if sshaddress and sshport and username:
                    sshclient.close() #ssh connection must be closed only when all tcp socket data was received on the remote side, hence the DoneSending/DoneReceiving ping pong
                break

        train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)
        valid_loader = torch.utils.data.DataLoader(valid_dataset,batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

    if msg_type == 'StartTraining' and modules_valid:
        print("Training... epoch "+str(scheduler.last_epoch)+"\n", file=sys.stderr)

    if msg_type == 'TrainOneEpoch' and modules_valid:
        #training
        model.train()
        train_loss = 0
        train_metrics = []
        for metric in metrics:
            metric.reset()
        for batch_id, tensors in enumerate(train_loader):
            inputs = [tensors[i].to(device) for i in input_tensors_id]
            targets = [tensors[i].to(device) for i in output_tensors_id]
            optimizer.zero_grad()
            outputs = model(*inputs)
            outputs = outputs if type(outputs) is not torch.Tensor else [outputs]
            loss = 0
            for output, target, criterion in zip(outputs, targets, criteria): #https://discuss.pytorch.org/t/a-model-with-multiple-outputs/10440
                loss = loss + criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs[0].size(0)

            with torch.set_grad_enabled(False):
                for output, target, metric in zip(outputs, targets, metrics):
                    metric.update(output, target)

        train_loss = train_loss/len(train_dataset)
        train_metrics = 0
        for metric in metrics:
            train_metrics = train_metrics+metric.compute().item()
        train_metrics/=len(metrics)
        scheduler.step()

        #validation
        model.eval()
        valid_loss = 0
        valid_metrics = []
        for metric in metrics:
            metric.reset()
        with torch.set_grad_enabled(False):
            for batch_id, tensors in enumerate(valid_loader):
                inputs = [tensors[i].to(device) for i in input_tensors_id]
                targets = [tensors[i].to(device) for i in output_tensors_id]
                outputs = model(*inputs)
                outputs = outputs if type(outputs) is not torch.Tensor else [outputs]
                loss = 0
                for output, target, criterion in zip(outputs, targets, criteria): #https://discuss.pytorch.org/t/a-model-with-multiple-outputs/10440
                    loss = loss + criterion(output, target)
                valid_loss += loss.item() * inputs[0].size(0)

                for output, target, metric in zip(outputs, targets, metrics):
                    metric.update(output, target)

        valid_loss = valid_loss/len(valid_dataset)
        valid_metrics = 0
        for metric in metrics:
            valid_metrics = valid_metrics+metric.compute().item()
        valid_metrics/=len(metrics)

        tc.send_msg(app_socket, 'TrainingLoss', tc.encode_floats(train_loss))
        tc.send_msg(app_socket, 'ValidationLoss', tc.encode_floats(valid_loss))
        tc.send_msg(app_socket, 'TrainingMetric', tc.encode_floats(train_metrics))
        tc.send_msg(app_socket, 'ValidationMetric', tc.encode_floats(valid_metrics))

        buffer=io.BytesIO()
        torch.save(deepcopy_cpu(model.state_dict()), buffer)
        tc.send_msg(app_socket, 'ModelState', buffer.getvalue())

        buffer=io.BytesIO()
        torch.save(deepcopy_cpu(optimizer.state_dict()), buffer)
        tc.send_msg(app_socket, 'OptimizerState', buffer.getvalue())

        tc.send_msg(app_socket, 'Trained')

        #create train_bar only after first successful training to avoid ghost progress message after an error
        if train_bar is not None:
            train_bar.bar_format='{desc} epoch {n_fmt} |{rate_fmt}\n\n'
        else:
            train_bar = tqdm(total=epochs, desc='Training...', bar_format='{desc} epoch '+str(scheduler.last_epoch)+'\n\n', initial=scheduler.last_epoch-1)
        train_bar.update(1)

    if msg_type == 'StopTraining' and modules_valid:
        if train_bar is not None:
            train_bar.close()
            train_bar=None
        print("Training stopped at epoch "+str(scheduler.last_epoch-1), file=sys.stderr)

    if msg_type == 'SetInputTensors' or msg_type == 'InferTensors':
        input_tensors = tc.decode_torch_tensors(msg_data)
        for i, tensor in enumerate(input_tensors):
            input_tensors[i]=torch.unsqueeze(tensor, 0).to(device) #add batch dimension

    if msg_type == 'InferTensors':
        if model is not None:
            with torch.set_grad_enabled(False):
                model.eval()
                output_tensors=model(*input_tensors)
                output_tensors=[output.cpu() for output in output_tensors]
                tc.send_msg(app_socket, 'InferedTensors', tc.encode_torch_tensors(output_tensors))

    if msg_type == 'SaveWeights':
        path = tc.decode_strings(msg_data)[0]
        torch.save(deepcopy_cpu(model.state_dict()), path)
        print("Export complete")

    if msg_type == 'SaveTorchScript':
        path, mode = tc.decode_strings(msg_data)
        if "torch.jit" in str(type(model)):
            torch.jit.save(model, path) #already a torchscript, save as is
            print("Export complete")
        else:
            if mode=="trace":
                error_msg, torchscript_model = safe_exec(torch.jit.trace,{'func':model, 'example_inputs':input_tensors, 'check_trace':False}, description='model tracing')
            else:
                error_msg, torchscript_model = safe_exec(torch.jit.script,{'obj':model}, description='model scripting')
            if error_msg:
                print("Error exporting:", error_msg, file=sys.stderr)
            else:
                torch.jit.save(torchscript_model, path)
                print("Export complete")

    if msg_type == 'SaveONNX':
        error_msg=None
        torchscript_model=model
        if not "torch.jit" in str(type(model)):
            error_msg, torchscript_model = safe_exec(torch.jit.trace,{'func':model, 'example_inputs':input_tensors, 'check_trace':False}, description='model tracing')
        if error_msg:
            print("Error exporting:", error_msg, file=sys.stderr)
        else:
            error_msg, torchscript_model = safe_exec(torch.onnx.export,{'model':torchscript_model, 'args':input_tensors, 'f':tc.decode_strings(msg_data)[0], 'opset_version':12})
            if error_msg:
                print("Error exporting:", error_msg, file=sys.stderr)
            else:
                print("Export complete")

    if msg_type == 'Exit':
        break

