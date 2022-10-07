import sys
import socket
import struct
import numpy as np
#import torch
import PIL

def generate_server(host='localhost', port=0):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))
    return server, [server.getsockname()[0], str(server.getsockname()[1])]

def start_server(server):
    server.listen()
    conn, addr = server.accept()
    return conn

def connect(server_address=None, timeout=0):
    if server_address==None and len(sys.argv)<3:
        print("Missing socket address and port", file=sys.stderr)
        exit()

    if not server_address:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--address", help="server address", type=str, default='localhost')
        parser.add_argument("--port", help="local port to which the script must connect", type=int, default=0)
        parser.add_argument("--timeout", help="max number of seconds without incoming messages before quitting", type=int, default=0)
        args, unknown = parser.parse_known_args()
        server_address = (args.address, args.port)
        timeout=args.timeout
    else:
        server_address = (server_address[0], int(server_address[1]))

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect(server_address)
    except socket.error as serr:
        print("Connection error: %s" % str(serr), file=sys.stderr)
        exit()
    if timeout>0:
        sock.settimeout(timeout)
    return sock

def send_msg(sock, type, data = bytearray()):
    type_bytes=bytes(type, 'utf-8')
    type_size=len(type_bytes)
    msg = struct.pack(f'<B{type_size}sI', type_size, type_bytes, len(data)) + data
    try:
        sock.sendall(msg)
    except:
        print("Lost connection", file=sys.stderr)
        exit()

def recv_msg(sock):
    def recvall(sock, n):
        data = bytearray()
        while len(data) < n:
            try:
                packet = sock.recv(n - len(data))
            except:
                print("Lost connection", file=sys.stderr)
                exit()
            if len(packet)==0:
                print("Lost connection", file=sys.stderr)
                exit()
            data.extend(packet)
        return data
    type_size = struct.unpack('<B', recvall(sock, 1))[0]
    type = struct.unpack(f'<{type_size}s', recvall(sock, type_size))[0]
    datalen = struct.unpack('<I', recvall(sock, 4))[0]
    return str(type, 'utf-8'), recvall(sock, datalen)



def encode_ints(ints):
    if ints is None:
        ints = []
    if type(ints) is not list:
        ints = [ints]
    size=len(ints)
    return struct.pack(f'<{size}l',*ints)

def decode_ints(data):
    count = len(data)//4
    return list(struct.unpack_from(f'<{count}l', data, 0))

def encode_floats(floats):
    if floats is None:
        floats = []
    if type(floats) is not list:
        floats = [floats]
    size=len(floats)
    return struct.pack(f'<{size}f',*floats)

def decode_floats(data):
    count = len(data)//4
    return list(struct.unpack_from(f'<{count}f', data, 0))

def encode_strings(strings):
    if strings is None:
        strings= []
    if type(strings) is not list:
        strings = [strings]
    data = bytearray()
    for string in strings:
        string = bytes(string, 'utf-8')
        size=len(string)
        data.extend(struct.pack(f'<L{size}s',size,string))
    return data

def decode_strings(data):
    shift = 0
    strings = []
    while shift<len(data):
        size = struct.unpack_from('<L', data, shift+0)[0]
        strings.append(str(struct.unpack_from(f'<{size}s', data, shift+4)[0], 'utf-8'))
        shift+=4+size
    return strings



def encode_torch_tensors(tensors):
    import torch
    from collections.abc import Iterable
    if not isinstance(tensors, Iterable):
        tensors = [tensors]
    return encode_numpy_tensors([tensor.numpy() for tensor in tensors])

def encode_numpy_tensors(tensors):
    from collections.abc import Iterable
    if not isinstance(tensors, Iterable):
        tensors = [tensors]
    buffer = bytes()
    for tensor in tensors:
        size=len(tensor.shape)
        buffer+=bytes(tensor.dtype.byteorder.replace('=','<' if sys.byteorder == 'little' else '>')+tensor.dtype.kind,'utf-8')+tensor.dtype.itemsize.to_bytes(1,byteorder='little')+struct.pack(f'<B{size}I',size,*tensor.shape)+tensor.tobytes()
    return buffer

def decode_torch_tensors(data):
    import torch
    tensors = decode_numpy_tensors(data)
    tensors = [torch.from_numpy(tensor) for tensor in tensors]
    return tensors

def decode_numpy_tensors(data):
    tensors = []
    while data:
        dtype = str(data[:2],'utf-8')
        dtype += str(data[2])
        size = data[3]
        shape = struct.unpack_from(f'<{size}I', data, 4)
        datasize=int(dtype[2])
        for dimension in shape:
            datasize*=dimension
        tensors.append(np.ndarray(shape, dtype=dtype, buffer=data[4+size*4:]))
        data=data[4+size*4+datasize:]
    return tensors


def encode_image(img):
    width, height = img.size
    size = width*height*4
    data=struct.pack(f'<II{size}s', width, height, img.convert(mode='RGBA').tobytes())
    return data
