import sys

import torchstudio.tcpcodec as tc
from torchstudio.modules import safe_exec
import os
import io
from collections.abc import Iterable
from tqdm.auto import tqdm
import pickle

original_path=sys.path

app_socket = tc.connect()
print("Analyze script connected\n", file=sys.stderr)
while True:
    msg_type, msg_data = tc.recv_msg(app_socket)

    if msg_type == 'SetAnalyzerCode':
        print("Setting analyzer code...\n", file=sys.stderr)
        analyzer = None
        analyzer_code = tc.decode_strings(msg_data)[0]
        error_msg, analyzer_env = safe_exec(analyzer_code, description='analyzer definition')
        if error_msg is not None or 'analyzer' not in analyzer_env:
            print("Unknown analyzer definition error" if error_msg is None else error_msg, file=sys.stderr)

    if msg_type == 'StartAnalysisServer' and 'analyzer' in analyzer_env:
        print("Analyzing...\n", file=sys.stderr)

        analysis_server, address = tc.generate_server()
        tc.send_msg(app_socket, 'ServerRequestingDataset', tc.encode_strings(address))

        dataset_socket=tc.start_server(analysis_server)

        tc.send_msg(dataset_socket, 'RequestMetaInfos')

        if analyzer_env['analyzer'].train is None:
            request_msg='RequestAllSamples'
        elif analyzer_env['analyzer'].train==True:
            request_msg='RequestTrainingSamples'
        elif analyzer_env['analyzer'].train==False:
            request_msg='RequestValidationSamples'
        tc.send_msg(dataset_socket, request_msg, tc.encode_strings(address))

        while True:
            dataset_msg_type, dataset_msg_data = tc.recv_msg(dataset_socket)

            if dataset_msg_type == 'InputTensorsID':
                input_tensors_id=tc.decode_ints(dataset_msg_data)

            if dataset_msg_type == 'OutputTensorsID':
                output_tensors_id=tc.decode_ints(dataset_msg_data)

            if dataset_msg_type == 'Labels':
                labels=tc.decode_strings(dataset_msg_data)

            if dataset_msg_type == 'NumSamples':
                num_samples=tc.decode_ints(dataset_msg_data)[0]
                pbar=tqdm(total=num_samples, desc='Analyzing...', bar_format='{l_bar}{bar}| {remaining} left\n\n') #see https://github.com/tqdm/tqdm#parameters

            if dataset_msg_type == 'StartSending':
                error_msg, return_value = safe_exec(analyzer_env['analyzer'].start_analysis, (num_samples, input_tensors_id, output_tensors_id, labels), description='analyzer definition')
                if error_msg is not None:
                    pbar.close()
                    print(error_msg, file=sys.stderr)
                    dataset_socket.close()
                    analysis_server.close()
                    break

            if dataset_msg_type == 'TrainingSample':
                pbar.update(1)
                error_msg, return_value = safe_exec(analyzer_env['analyzer'].analyze_sample, (tc.decode_numpy_tensors(dataset_msg_data), True), description='analyzer definition')
                if error_msg is not None:
                    pbar.close()
                    print(error_msg, file=sys.stderr)
                    dataset_socket.close()
                    analysis_server.close()
                    break

            if dataset_msg_type == 'ValidationSample':
                pbar.update(1)
                error_msg, return_value = safe_exec(analyzer_env['analyzer'].analyze_sample, (tc.decode_numpy_tensors(dataset_msg_data), False), description='analyzer definition')
                if error_msg is not None:
                    pbar.close()
                    print(error_msg, file=sys.stderr)
                    dataset_socket.close()
                    analysis_server.close()
                    break

            if dataset_msg_type == 'DoneSending':
                pbar.close()
                error_msg, return_value = safe_exec(analyzer_env['analyzer'].finish_analysis, description='analyzer definition')
                tc.send_msg(dataset_socket, 'DisconnectFromWorkerServer')
                dataset_socket.close()
                analysis_server.close()
                if error_msg is not None:
                    print(error_msg, file=sys.stderr)
                else:
                    buffer=io.BytesIO()
                    pickle.dump(analyzer_env['analyzer'].state_dict(), buffer)
                    tc.send_msg(app_socket, 'AnalyzerState',buffer.getvalue())
                    tc.send_msg(app_socket, 'AnalysisWeights',tc.encode_floats(analyzer_env['analyzer'].weights))
                    print("Analysis complete")
                break

    if msg_type == 'LoadAnalyzerState':
        if 'analyzer' in analyzer_env:
            buffer=io.BytesIO(msg_data)
            analyzer_env['analyzer'].load_state_dict(pickle.load(buffer))
            print("Analyzer state loaded")

    if msg_type == 'RequestAnalysisReport':
        resolution = tc.decode_ints(msg_data)
        if 'analyzer' in analyzer_env and resolution[0]>0 and resolution[1]>0:
            error_msg, return_value = safe_exec(analyzer_env['analyzer'].generate_report, (resolution[0:2],resolution[2]), description='analyzer definition')
            if error_msg is not None:
                print(error_msg, file=sys.stderr)
            if return_value is not None:
                tc.send_msg(app_socket, 'ReportImage', tc.encode_image(return_value))
        else:
            tc.send_msg(app_socket, 'ReportImage')

    if msg_type == 'Exit':
        break
