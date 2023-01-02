#workaround until Pytorch 1.12.1 is released: https://github.com/pytorch/pytorch/issues/78490
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import sys
print("Loading PyTorch...\n", file=sys.stderr)

import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
import torchstudio.tcpcodec as tc
from torchstudio.modules import safe_exec
import random
import os
import io
import time
from collections.abc import Iterable
from tqdm.auto import tqdm
import hashlib

#monkey patch ssl to fix ssl certificate fail when downloading datasets on some configurations: https://stackoverflow.com/questions/27835619/urllib-and-ssl-certificate-verify-failed-error
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

meta_dataset = None
input_tensors_id = []
output_tensors_id = []

class MetaDataset(Dataset):
    def __init__(self, train, valid=None):
        self.train_dataset=train
        self.valid_dataset=valid
        self.train_count=None
        self.shuffle=0
        self.smp_usage=1.0
        self.training=True
        self.classes=train.classes if hasattr(train,'classes') else []
        self._gen_index()

    def _gen_index(self):
        self.index = []
        for i in range(len(self.train_dataset)):
            self.index.append((self.train_dataset,i))
        if self.valid_dataset is not None:
            for i in range(len(self.valid_dataset)):
                self.index.append((self.valid_dataset,i))
        if self.train_count is None:
            self.train_count=len(self.train_dataset)
            if self.valid_dataset is None:
                self.train_count=round(self.train_count*0.8)

        if self.shuffle>0:
            #Fisher–Yates shuffle: https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle
            random.seed(0)
            shuffle_count=self.train_count if self.shuffle==1 else len(self.index)
            for sample in range(shuffle_count):
                target_sample=random.randrange(sample,shuffle_count)
                self.index[sample], self.index[target_sample] = self.index[target_sample], self.index[sample]

    def set_num_train(self, num):
        self.train_count=min(num,len(self.index))
        self._gen_index()

    def set_smp_usage(self, ratio):
        self.smp_usage=min(max(ratio,0.0),1.0)
        self._gen_index()

    def set_shuffle(self, mode):
        self.shuffle=mode
        self._gen_index()

    def train(self, mode=True):
        self.training=mode
        return self

    def valid(self):
        self.training=False
        return self

    def __len__(self):
        if self.training==True:
            return round(self.train_count*self.smp_usage)
        else:
            return round((len(self.index)-self.train_count)*self.smp_usage)

    def __getitem__(self, id):
        if id<0 or id>=len(self):
            raise IndexError
        if self.training==True:
            sample_ref=self.index[id]
        else:
            sample_ref=self.index[id+self.train_count]
        sample=sample_ref[0][sample_ref[1]]

        #convert to list if needed
        if isinstance(sample, Iterable):
            if type(sample) is dict:
                sample=list(sample.values())
            else:
                sample=list(sample)
        else:
            sample=[sample]

        #convert each element of the list to a tensor if needed
        sample_tensors=[]
        for i in range(len(sample)):
            if type(sample[i]) is not torch.Tensor:
                if 'PIL' in str(type(sample[i])) or 'numpy' in str(type(sample[i])):
                    sample_tensors.append(to_tensor(sample[i]))
                else:
                    try:
                        sample_tensors.append(torch.tensor(sample[i]))
                    except:
                        pass
            else:
                sample_tensors.append(sample[i])

        #and finally solidify into a tuple
        sample_tensors=tuple(sample_tensors)

        return sample_tensors

original_path=sys.path
original_dir=os.getcwd()

app_socket = tc.connect()
print("Dataset script connected\n", file=sys.stderr)
while True:
    msg_type, msg_data = tc.recv_msg(app_socket)

    if msg_type == 'SetCurrentDir':
        new_dir=tc.decode_strings(msg_data)[0]
        sys.path=original_path
        os.chdir(original_dir)
        if new_dir:
            sys.path.append(new_dir)
            os.chdir(new_dir)

    if msg_type == 'SetDatasetCode':
        print("Loading dataset...\n", file=sys.stderr)

        meta_dataset = None
        error_msg, dataset_env = safe_exec(tc.decode_strings(msg_data)[0], description='dataset definition')
        if error_msg is not None or 'train' not in dataset_env:
            print("Unknown dataset definition error" if error_msg is None else error_msg, file=sys.stderr)
        else:
            meta_dataset=MetaDataset(dataset_env['train'], dataset_env['valid'] if 'valid' in dataset_env else None)
            tc.send_msg(app_socket, 'Labels', tc.encode_strings(meta_dataset.classes))
            tc.send_msg(app_socket, 'NumSamples', tc.encode_ints([len(meta_dataset.train()),len(meta_dataset.valid())]))
            sample=meta_dataset.train()[0]

            #suggest default formats
            type_id=[1 for i in range(len(sample))] #inputs
            if len(sample)==1:
                type_id[-1]=3 #input/output
            if len(sample)>1:
                type_id[-1]=2 #output
            tc.send_msg(app_socket, 'SetTypes', tc.encode_ints(type_id))

            renderer_name=[]
            for tensor in sample:
                if len(tensor.shape)==4:
                    renderer_name.append("Volume")
                elif len(tensor.shape)==3 and tensor.dtype==torch.complex64:
                    renderer_name.append("Spectrogram")
                elif len(tensor.shape)==3:
                    renderer_name.append("Bitmap")
                elif len(tensor.shape)==2:
                    renderer_name.append("Signal")
                elif len(tensor.shape)<2:
                    renderer_name.append("Labels")
                else:
                    renderer_name.append("Custom")
            tc.send_msg(app_socket, 'SetRendererNames', tc.encode_strings(renderer_name))

            if sample and len(sample[-1].shape)==0 and "int" in str(sample[-1].dtype):
                analyzer_name="Multiclass"
            elif sample and len(sample[-1].shape)==1:
                analyzer_name="MultiLabel"
            else:
                analyzer_name="ValuesDistribution"
            tc.send_msg(app_socket, 'SetAnalyzerName', tc.encode_strings(analyzer_name))

            print("Loading complete")

    if msg_type == 'RequestTrainingSamples' or msg_type == 'RequestValidationSamples':
        if meta_dataset is not None:
            meta_dataset.train(msg_type == 'RequestTrainingSamples')
            samples_id = tc.decode_ints(msg_data)
            for id in samples_id:
                tc.send_msg(app_socket, 'TensorData', tc.encode_torch_tensors(meta_dataset[id]))

    if msg_type == 'SetNumTrainingSamples':
        if meta_dataset is not None:
            meta_dataset.set_num_train(tc.decode_ints(msg_data)[0])

    if msg_type == 'SetSampleUsage':
        if meta_dataset is not None:
            meta_dataset.set_smp_usage(tc.decode_floats(msg_data)[0])

    if msg_type == 'SetShuffleMode':
        if meta_dataset is not None:
            meta_dataset.set_shuffle(tc.decode_ints(msg_data)[0])

    if msg_type == 'InputTensorsID':
        input_tensors_id = tc.decode_ints(msg_data)

    if msg_type == 'OutputTensorsID':
        output_tensors_id = tc.decode_ints(msg_data)

    if msg_type == 'ConnectToWorkerServer':
        name, sshaddress, sshport, username, password, keydata, address, port = tc.decode_strings(msg_data)
        port=int(port)

        print('Connecting to '+name+'...\n', file=sys.stderr)

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
            worker_socket = socket.socket()
            worker_socket.bind(('localhost', 0))
            freeport=worker_socket.getsockname()[1]
            worker_socket.close()
            forward_tunnel = sshtunnel.Tunnel(sshclient, sshtunnel.ForwardTunnel, 'localhost', freeport, address if address else 'localhost', port)
            port=freeport

        try:
            worker_socket = tc.connect((address,port),timeout=10)
            while True:
                worker_msg_type, worker_msg_data = tc.recv_msg(worker_socket)

                if worker_msg_type == 'RequestMetaInfos':
                    tc.send_msg(worker_socket, 'InputTensorsID', tc.encode_ints(input_tensors_id))
                    tc.send_msg(worker_socket, 'OutputTensorsID', tc.encode_ints(output_tensors_id))
                    tc.send_msg(worker_socket, 'Labels', tc.encode_strings(meta_dataset.classes))

                if worker_msg_type == 'RequestHash':
                    dataset_hash = hashlib.md5()
                    dataset_hash.update(int(len(meta_dataset.train())).to_bytes(4, 'little'))
                    if len(meta_dataset)>0:
                        dataset_hash.update(tc.encode_torch_tensors(meta_dataset[0]))
                    dataset_hash.update(int(len(meta_dataset.valid())).to_bytes(4, 'little'))
                    if len(meta_dataset)>0:
                        dataset_hash.update(tc.encode_torch_tensors(meta_dataset[0]))
                    tc.send_msg(worker_socket, 'DatasetHash', dataset_hash.digest())

                if worker_msg_type == 'RequestTrainingSamples' or worker_msg_type == 'RequestValidationSamples' or worker_msg_type == 'RequestAllSamples':
                    train_set=True if worker_msg_type == 'RequestTrainingSamples' or worker_msg_type == 'RequestAllSamples' else False
                    valid_set=True if worker_msg_type == 'RequestValidationSamples' or worker_msg_type == 'RequestAllSamples' else False
                    num_samples=(len(meta_dataset.train()) if train_set else 0) + (len(meta_dataset.valid()) if valid_set else 0)
                    tc.send_msg(worker_socket, 'NumSamples', tc.encode_ints(num_samples))

                    tc.send_msg(worker_socket, 'StartSending')
                    with tqdm(total=num_samples, desc='Sending samples to '+name+'...', bar_format='{l_bar}{bar}| {remaining} left\n\n') as pbar:
                        if train_set:
                            meta_dataset.train()
                            for i in range(len(meta_dataset)):
                                tc.send_msg(worker_socket, 'TrainingSample', tc.encode_torch_tensors(meta_dataset[i]))
                                pbar.update(1)
                        if valid_set:
                            meta_dataset.valid()
                            for i in range(len(meta_dataset)):
                                tc.send_msg(worker_socket, 'ValidationSample', tc.encode_torch_tensors(meta_dataset[i]))
                                pbar.update(1)

                    tc.send_msg(worker_socket, 'DoneSending')

                if worker_msg_type == 'DisconnectFromWorkerServer':
                    worker_socket.close()
                    print('Samples transfer to '+name+' completed')
                    break

        except:
            if sshaddress and sshport and username:
                time.sleep(.5) #let some time for threaded ssh error messages to print first
            print('Samples transfer to '+name+' interrupted', file=sys.stderr)

        if sshaddress and sshport and username:
            try:
                del forward_tunnel
            except:
                pass
            try:
                sshclient.close() #ssh connection must be closed only when all tcp socket data was received on the remote side, hence the DoneSending/DisconnectFromWorkerServer ping pong
            except:
                pass

    if msg_type == 'Exit':
        break

