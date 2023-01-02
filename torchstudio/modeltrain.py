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
from tqdm.auto import tqdm
from collections.abc import Iterable
import threading


class CachedDataset(Dataset):
    def __init__(self, train=True, hash=None):
        self.index = []
        self.cache=None
        if hash:
            self.filename='cache/dataset-'+('training' if train==True else 'validation')
            if os.path.exists(self.filename):
                self.cache = open(self.filename, 'rb')
                cached_hash=self.cache.read(16)
                if cached_hash==hash:
                    size=self.cache.read(4)
                    while size:
                        data=self.cache.read(int.from_bytes(size, 'little'))
                        self.index.append(tc.decode_torch_tensors(data))
                        size=self.cache.read(4)
                    self.cache.close()
                    return
                else:
                    self.cache.close()
                    os.remove(self.filename)
            if os.path.exists(self.filename+'.tmp'):
                os.remove(self.filename+'.tmp')
            if not os.path.exists('cache'):
                os.mkdir('cache')
            self.cache = open(self.filename+'.tmp', 'wb')
            self.cache.write(hash)

    def add_sample(self, data=None):
        if data:
            if self.cache:
                self.cache.write(len(data).to_bytes(4, 'little'))
                self.cache.write(data)
            self.index.append(tc.decode_torch_tensors(data))
        else:
            if self.cache:
                self.cache.close()
                try:
                    os.rename(self.filename+'.tmp', self.filename)
                except:
                    pass

    def __len__(self):
        return len(self.index)

    def __getitem__(self, id):
        if id<0 or id>=len(self):
            raise IndexError
        return self.index[id]

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

train_dataset = CachedDataset(True)
valid_dataset = CachedDataset(False)
train_bar = None

model = None
sender_thread = None

cache = None

app_socket = tc.connect()
print("Training script connected\n", file=sys.stderr)
while True:
    msg_type, msg_data = tc.recv_msg(app_socket)

    if msg_type == 'SetDevice':
        print("Setting device...\n", file=sys.stderr)
        device_id=tc.decode_strings(msg_data)[0]
        device = torch.device(device_id)
        pin_memory = True if 'cuda' in device_id else False

    if msg_type == 'SetMode':
        print("Setting mode...\n", file=sys.stderr)
        mode=tc.decode_strings(msg_data)[0]

    if msg_type == 'SetCache':
        print("Setting cache...\n", file=sys.stderr)
        cache = True if tc.decode_ints(msg_data)[0]==1 else False

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
        batch_size, shuffle, epochs, early_stop = tc.decode_ints(msg_data)
        shuffle=True if shuffle==1 else False

    if msg_type == 'SetBestMetrics':
        best_train_loss, best_valid_loss, best_train_metric, best_valid_metric = tc.decode_floats(msg_data)

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
            sshclient.connect(hostname=sshaddress, port=int(sshport), username=username, password=password, pkey=pkey, timeout=10)

            reverse_tunnel = sshtunnel.Tunnel(sshclient, sshtunnel.ReverseTunnel, 'localhost', 0, 'localhost', int(address[1]))
            address[1]=str(reverse_tunnel.lport)

        tc.send_msg(app_socket, 'ServerRequestingDataset', tc.encode_strings(address))

        dataset_socket=tc.start_server(training_server)

        tc.send_msg(dataset_socket, 'RequestMetaInfos')
        tc.send_msg(dataset_socket, 'RequestHash')

        while True:
            dataset_msg_type, dataset_msg_data = tc.recv_msg(dataset_socket)

            if dataset_msg_type == 'InputTensorsID' and modules_valid:
                input_tensors_id = tc.decode_ints(dataset_msg_data)

            if dataset_msg_type == 'OutputTensorsID' and modules_valid:
                output_tensors_id = tc.decode_ints(dataset_msg_data)

            if dataset_msg_type == 'DatasetHash':
                train_dataset=CachedDataset(True)
                valid_dataset=CachedDataset(False)
                if cache:
                    train_dataset=CachedDataset(True, dataset_msg_data)
                    valid_dataset=CachedDataset(False, dataset_msg_data)
                if len(train_dataset)==0 and len(valid_dataset)==0:
                    tc.send_msg(dataset_socket, 'RequestAllSamples', tc.encode_strings(address))
                elif len(train_dataset)==0:
                    tc.send_msg(dataset_socket, 'RequestTrainingSamples', tc.encode_strings(address))
                elif len(valid_dataset)==0:
                    tc.send_msg(dataset_socket, 'RequestValidationSamples', tc.encode_strings(address))
                else:
                     break

            if dataset_msg_type == 'NumSamples':
                num_samples=tc.decode_ints(dataset_msg_data)[0]
                pbar=tqdm(total=num_samples, desc='Caching...', bar_format='{l_bar}{bar}| {remaining} left\n\n') #see https://github.com/tqdm/tqdm#parameters

            if dataset_msg_type == 'TrainingSample':
                train_dataset.add_sample(dataset_msg_data)
                pbar.update(1)

            if dataset_msg_type == 'ValidationSample':
                valid_dataset.add_sample(dataset_msg_data)
                pbar.update(1)

            if dataset_msg_type == 'DoneSending':
                train_dataset.add_sample()
                valid_dataset.add_sample()
                pbar.close()
                break

        tc.send_msg(dataset_socket, 'DisconnectFromWorkerServer')
        dataset_socket.close()
        training_server.close()
        if sshaddress and sshport and username:
            sshclient.close() #ssh connection must be closed only when all tcp socket data was received on the remote side, hence the DoneSending/DisconnectFromWorkerServer ping pong

        train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)
        valid_loader = torch.utils.data.DataLoader(valid_dataset,batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

    if msg_type == 'StartTraining' and modules_valid:
        scaler = None
        if 'cuda' in device_id:
            #https://pytorch.org/docs/stable/notes/cuda.html
            torch.backends.cuda.matmul.allow_tf32 = True if mode=='TF32' else False
            torch.backends.cudnn.allow_tf32 = True
            if mode=='FP16':
                scaler = torch.cuda.amp.GradScaler()
            if mode=='BF16':
                os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1" #https://discuss.pytorch.org/t/bfloat16-has-worse-performance-than-float16-for-conv2d/154373
        train_type=None
        if mode=='FP16':
            train_type=torch.float16
        if mode=='BF16':
            train_type=torch.bfloat16
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

            with torch.autocast(device_type='cuda' if 'cuda' in device_id else 'cpu', dtype=train_type, enabled=True if train_type else False):
                outputs = model(*inputs)
                outputs = outputs if type(outputs) is not torch.Tensor else [outputs]
                loss = 0
                for output, target, criterion in zip(outputs, targets, criteria): #https://discuss.pytorch.org/t/a-model-with-multiple-outputs/10440
                    loss = loss + criterion(output, target)

            if scaler:
                # Accumulates scaled gradients.
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_loss += loss.item() * inputs[0].size(0)

            with torch.set_grad_enabled(False):
                for output, target, metric in zip(outputs, targets, metrics):
                    metric.update(output, target)

        train_loss = train_loss/len(train_dataset)
        train_metric = 0
        for metric in metrics:
            train_metric = train_metric+metric.compute().item()
        train_metric/=len(metrics)
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
        valid_metric = 0
        for metric in metrics:
            valid_metric = valid_metric+metric.compute().item()
        valid_metric/=len(metrics)

        #threaded (async) results sending, so to send last metrics and best weights when available while calculating new ones
        if sender_thread:
            sender_thread.join()

        metrics_values=[train_loss, valid_loss, train_metric, valid_metric]

        model_state_buffer=None
        optimizer_state_buffer=None

        if valid_metric>best_valid_metric or (valid_metric==best_valid_metric and valid_loss<best_valid_loss):
            model_state_buffer=io.BytesIO()
            torch.save(deepcopy_cpu(model.state_dict()), model_state_buffer)
            optimizer_state_buffer=io.BytesIO()
            torch.save(deepcopy_cpu(optimizer.state_dict()), optimizer_state_buffer)

            best_train_loss=train_loss
            best_valid_loss=valid_loss
            best_train_metric=train_metric
            best_valid_metric=valid_metric

        def send_results_back():
            tc.send_msg(app_socket, 'Metrics', tc.encode_floats(metrics_values))
            if model_state_buffer:
                tc.send_msg(app_socket, 'ModelState', model_state_buffer.getvalue())
            if optimizer_state_buffer:
                tc.send_msg(app_socket, 'OptimizerState', optimizer_state_buffer.getvalue())
            tc.send_msg(app_socket, 'TrainingResultsSent')

        sender_thread=threading.Thread(target=send_results_back)
        sender_thread.start()

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
                error_msg, output_tensors = safe_exec(model, input_tensors, description='model inference')
                if error_msg:
                    print(error_msg, file=sys.stderr)
                else:
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

