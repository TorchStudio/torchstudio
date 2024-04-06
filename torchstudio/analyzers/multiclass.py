from torchstudio.modules import Analyzer
from typing import List
import numpy as np
from random import randint
import zlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import PIL
import sys

class Multiclass(Analyzer):
    """Analyze the distribution of multiclass datasets
    (single integer output, single label prediction)
    https://en.wikipedia.org/wiki/Multiclass_classification

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
                if "int" in str(sample[id].dtype) and len(sample[id].shape)==0:
                    self.tensor_id=id
                    break
            if self.tensor_id is None:
                raise ValueError('Multiclass analysis requires a single integer output tensor')

        class_id=sample[self.tensor_id].item()
        self.classes_sequence.extend(class_id.to_bytes(2,'little'))
        if class_id not in self.classes:
            self.classes[class_id]=1
        else:
            self.classes[class_id]+=1

    def finish_analysis(self):
        num_registered_classes=len(self.classes)
        random_sequence=bytearray()
        for i in range(self.num_samples):
            random_sequence.extend(randint(0, num_registered_classes).to_bytes(2,'little'))
        self.classes_randomness=len(zlib.compress(self.classes_sequence))/len(zlib.compress(random_sequence))

        #prepare weights and labels
        last_class_id=0
        for class_id in self.classes:
            last_class_id=max(last_class_id,class_id)
        self.classes_weight=[]
        self.classes_label=[]
        for class_id in range(last_class_id+1):
            if class_id not in self.classes:
                self.classes_weight.append(0)
            else:
                self.classes_weight.append(self.classes[class_id])
            if class_id>=len(self.labels):
                self.classes_label.append(str(class_id))
            else:
                self.classes_label.append(self.labels[class_id])

        return self.classes_weight

    def generate_report(self, size, dpi):
        if not self.classes_weight:
            raise ValueError("Nothing to report")

        #set up matplotlib renderer, style, figure and axis
        mpl.use('agg') #https://www.namingcrisis.net/post/2019/03/11/interactive-matplotlib-ipython/
        plt.style.use('dark_background')
        plt.rcParams.update({'font.size': 8})
        fig, [ax1, ax2] = plt.subplots(1 if size[0]>size[1] else 2, 2 if size[0]>size[1] else 1, figsize=(size[0]/dpi, size[1]/dpi), dpi=dpi)

        ax1.set_title("Class Distribution")
        ax1.pie(self.classes_weight, labels=self.classes_label, autopct='%1.1f%%', colors=plt.cm.tab10.colors, startangle=90, counterclock=False)

        ax2.set_title("Class Randomness")
        _, _, autopct = ax2.pie([self.classes_randomness,max(0,1-self.classes_randomness)], autopct='%1.1f%%', textprops={'fontsize': 16}, pctdistance=0, colors=["#b03070","black"], startangle=90, counterclock=False)
        autopct[1].set_visible(False)
        ax2.add_patch(Circle( (0,0), 0.6, color='black'))

        plt.tight_layout(pad=0)

        canvas = plt.get_current_fig_manager().canvas
        canvas.draw()
        img = PIL.Image.frombytes('RGBA',canvas.get_width_height(),canvas.buffer_rgba())
        plt.close()
        return img

