from typing import List, Tuple, Union, Sequence, Optional
import sys, traceback

#Base class for dataset analyzers
import numpy as np
import PIL.Image
class Analyzer:
    def __init__(self, train=True):
        self.train=train #True: analyze the training dataset, False: analyze the validation dataset, None: analyze the entire dataset
        self.weights=[] #List of float weights, if relevant

    def start_analysis(self, num_samples: int, input_tensors_id: List[int], output_tensors_id: List[int], labels: List[str]):
        pass

    def analyze_sample(self, sample: List[np.array], training_sample: bool):
        pass

    def finish_analysis(self):
        pass

    def generate_report(self, size: Tuple[int, int], dpi: int):
        pass

    def state_dict(self):
            return self.__dict__

    def load_state_dict(self, state_dict):
            self.__dict__.update(state_dict)


#Base class for renderers
import numpy as np
import PIL.Image
class Renderer:
    def render(self, title:str, tensor: np.array, size: Tuple[int, int], dpi: int, shift=(0,0,0,0), scale=(1,1,1,1), input_tensors=[], target_tensor=None, labels=[]) -> Union[PIL.Image.Image, str]:
        pass

#Base class for metrics
class Metric:
    #should be called with torch.no_grad(): to save time and space
    def update(self, preds, target):
        pass

    def compute(self):
        pass

    def reset(self):
        pass


def safe_exec(cmd, parameters=None, context=None, output=None, description='code'):
    error_msg=None
    return_value=None
    try:
        if type(cmd) is str:
            return_value = output if output is not None else {}
            exec(cmd, context if context is not None else return_value, return_value)
        elif parameters==None:
            return_value=cmd()
        elif type(parameters)==list or type(parameters)==tuple:
            return_value=cmd(*parameters)
        elif type(parameters)==dict:
            return_value=cmd(**parameters)
    except SyntaxError as err:
        error_class = err.__class__.__name__
        detail = err.args[0]
        file = "File "+err.filename if err.filename!='<string>' else description
        line_number = err.lineno
        offset = err.offset
        text = err.text
        error_msg = "%s, line %d\n%s%s\n%s: %s" % (file, line_number, text,  str('^').rjust(offset," "), error_class, detail)
    except Exception as err:
        error_class = err.__class__.__name__
        detail = err.args[0]
        cl, exc, tb = sys.exc_info()
        file = "File "+traceback.extract_tb(tb)[-1][0] if traceback.extract_tb(tb)[-1][0]!='<string>' else description
        line_number = traceback.extract_tb(tb)[-1][1]
        function = traceback.extract_tb(tb)[-1][2]
        text = traceback.extract_tb(tb)[-1][3]
        error_msg = "%s, line %d, in %s\n%s\n%s: %s" % (file, line_number, function, text, error_class, detail)
    return error_msg, return_value
