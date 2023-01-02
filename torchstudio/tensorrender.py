import torchstudio.tcpcodec as tc
from torchstudio.modules import safe_exec
import inspect
import sys
import os

title = ''
tensor = None
resolution = (256,256, 96)
shift = (0,0,0,0)
scale = (1,1,1,1)
input_tensors = []
target_tensor = None
labels = []

app_socket = tc.connect()
while True:
    msg_type, msg_data = tc.recv_msg(app_socket)

    if msg_type == 'SetRendererCode':
        error_msg, renderer_env = safe_exec(tc.decode_strings(msg_data)[0],description='renderer definition')
        if error_msg is not None or 'renderer' not in renderer_env:
            print("Unknown renderer definition error" if error_msg is None else error_msg, file=sys.stderr)
        else:
            tc.send_msg(app_socket, 'Documentation', tc.encode_strings(inspect.cleandoc(renderer_env['renderer'].__doc__) if renderer_env['renderer'].__doc__ is not None else ""))

    if msg_type == 'Clear':
        tensor = None
        input_tensors = []
        target_tensor = None

    if msg_type == 'SetTitle':
        title = tc.decode_strings(msg_data)[0]

    if msg_type == 'TensorData':
        tensor = tc.decode_numpy_tensors(msg_data)[0]

    if msg_type == 'SetResolution':
        resolution = tc.decode_ints(msg_data)

    if msg_type == 'SetShift':
        shift = tc.decode_floats(msg_data)
    if msg_type == 'SetScale':
        scale = tc.decode_floats(msg_data)

    if msg_type == 'SetInputTensors':
        input_tensors = tc.decode_numpy_tensors(msg_data)

    if msg_type == 'SetTargetTensors':
        target_tensors = tc.decode_numpy_tensors(msg_data)
        if target_tensors:
            target_tensor=target_tensors[0]
        else:
            target_tensor=None

    if msg_type == 'SetLabels':
        labels = tc.decode_strings(msg_data)

    if msg_type == 'Render':
        if 'renderer' in renderer_env and tensor is not None and resolution[0]>0 and resolution[1]>0:
            error_msg, img = safe_exec(renderer_env['renderer'].render, (title, tensor,resolution[0:2],resolution[2],shift,scale,input_tensors,target_tensor,labels), description='renderer definition')
            if error_msg is not None:
                print(error_msg, file=sys.stderr)
            if img is None:
                tc.send_msg(app_socket, 'ImageError')
            else:
                tc.send_msg(app_socket, 'ImageData', tc.encode_image(img))
        else:
            tc.send_msg(app_socket, 'ImageError')

    if msg_type == 'Exit':
        break
