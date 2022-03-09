from torchstudio.modules import Analyzer
from typing import List
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import PIL
import sys

class MultiLabel(Analyzer):
    """Analyze the distribution of multi-label datasets
    (multiple output values, multiple label predictions)
    https://en.wikipedia.org/wiki/Multi-label_classification

    Args:
        train: If True, analyze the training set.
               If False, analyze the validation set.
               If None, analyze the entire dataset.
    """
    def __init__(self, train=True):
        super().__init__(train)

    def start_analysis(self, num_samples: int, input_tensors_id: List[int], output_tensors_id: List[int], labels: List[str]):
        self.num_samples=num_samples
        self.input_tensors_id=input_tensors_id
        self.output_tensors_id=output_tensors_id
        self.labels=labels

        self.classes_weight=[]
        self.classes_label=[]
        self.classes_randomness=0

        self.tensor_id=None
        self.classes={}
        self.classes_sequence=bytearray()

    def analyze_sample(self, sample: List[np.array], training_sample: bool):
        if self.tensor_id is None:
            for id in self.output_tensors_id:
                if len(sample[id].shape)==1:
                    self.tensor_id=id
                    break
            if self.tensor_id is None:
                raise ValueError('Multi-label analysis requires a 1D output tensor')
            else:
                self.classes=np.zeros_like(sample[self.tensor_id], dtype=np.float32)
                self.classes_per_sample={}

        self.classes=self.classes+np.float32(sample[self.tensor_id])
        classes_count=np.count_nonzero(sample[self.tensor_id])
        if classes_count not in self.classes_per_sample:
            self.classes_per_sample[classes_count]=1
        else:
            self.classes_per_sample[classes_count]+=1

    def finish_analysis(self):
            #prepare weights and labels
            self.classes_weight=self.classes.tolist()
            self.classes_label=[]
            for class_id, class_weight in enumerate(self.classes_weight):
                if class_id>=len(self.labels):
                    self.classes_label.append(str(class_id))
                else:
                    self.classes_label.append(self.labels[class_id])

            #sort classes_per_sample by keys, and separate keys and values in 2 sub-lists
            self.classes_per_sample=sorted(self.classes_per_sample.items())
            self.classes_per_sample=list(zip(*self.classes_per_sample))

            return self.classes_weight

    def generate_report(self, size, dpi):
        if not self.classes_weight:
            print("Nothing to report", file=sys.stderr)
            return None

        #set up matplotlib renderer, style, figure and axis
        mpl.use('agg') #https://www.namingcrisis.net/post/2019/03/11/interactive-matplotlib-ipython/
        plt.style.use('dark_background')
        plt.rcParams.update({'font.size': 8})
        fig, [ax1, ax2] = plt.subplots(1 if size[0]>size[1] else 2, 2 if size[0]>size[1] else 1, figsize=(size[0]/dpi, size[1]/dpi), dpi=dpi)

        ax1.set_title("Class Distribution")
        ax1.pie(self.classes_weight, labels=self.classes_label, autopct='%1.1f%%', colors=plt.cm.tab10.colors, startangle=90, counterclock=False)
        ax2.set_title("Classes Per Sample")
        ax2.pie(self.classes_per_sample[1], labels=self.classes_per_sample[0], autopct='%1.1f%%', colors=plt.cm.tab10.colors, startangle=90, counterclock=False)
        plt.tight_layout(pad=0)

        canvas = plt.get_current_fig_manager().canvas
        canvas.draw()
        img = PIL.Image.frombytes('RGB',canvas.get_width_height(),canvas.tostring_rgb())
        plt.close()
        return img

