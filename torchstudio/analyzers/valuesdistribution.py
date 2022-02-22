from torchstudio.modules import Analyzer
from typing import List
import numpy as np
import math
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import PIL
import sys

class ValuesDistribution(Analyzer):
    """Analyze the range, mean and standard deviation for all tensors
    https://en.wikipedia.org/wiki/Standard_deviation

    Args:
        train: If True, analyze the training set.
               If False, analyze the validation set.
               If None, analyze the entire dataset.
    """
    def __init__(self, train=True):
        super().__init__(train)

    def start_analysis(self, num_samples: int, input_tensors_id: List[int], output_tensors_id: List[int], labels: List[str]):
        #one pass standard deviation:
        #https://www.strchr.com/standard_deviation_in_one_pass
        #https://pballew.blogspot.com/2008/10/standard-deviation-computation-in-one.html
        #https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
        self.num_samples=num_samples
        self.input_tensors_id=input_tensors_id
        self.output_tensors_id=output_tensors_id
        self.labels=labels

        self.name = []
        self.index = []
        self.min = []
        self.max = []
        self.n = []
        self.sum = []
        self.sq_sum = []
        self.mean = []
        self.var = []
        self.std_dev = []

        for id, input_tensor_id in enumerate(input_tensors_id):
            self.name.append("Input")
            self.index.append(input_tensor_id)
            self.min.append(math.inf)
            self.max.append(-math.inf)
            self.n.append(0)
            self.sum.append(0)
            self.sq_sum.append(0)
        for id, output_tensor_id in enumerate(output_tensors_id):
            self.name.append("Target")
            self.index.append(output_tensor_id)
            self.min.append(math.inf)
            self.max.append(-math.inf)
            self.n.append(0)
            self.sum.append(0)
            self.sq_sum.append(0)

    def analyze_sample(self, sample: List[np.array], training_sample: bool):
        for i, tensor_id in enumerate(self.index):
            self.min[i]=min(self.min[i],np.amin(sample[tensor_id]))
            self.max[i]=max(self.max[i],np.amax(sample[tensor_id]))
            self.n[i]+=sample[tensor_id].size
            self.sum[i]+=np.sum(sample[tensor_id])
            self.sq_sum[i]+=np.sum(np.square(sample[tensor_id]))

    def finish_analysis(self):
        for i in range(len(self.index)):
            self.mean.append(self.sum[i]/self.n[i])
            self.var.append(self.sq_sum[i]/self.n[i]-self.mean[i]*self.mean[i])
            self.std_dev.append(math.sqrt(self.var[i]))


    def generate_report(self, size, dpi):
        if not self.name:
            print("Nothing to report", file=sys.stderr)
            return None

        #set up matplotlib renderer, style, figure and axis
        mpl.use('agg') #https://www.namingcrisis.net/post/2019/03/11/interactive-matplotlib-ipython/
        plt.style.use('dark_background')
        plt.rcParams.update({'font.size': 8})

        fig, axs = plt.subplots(len(self.name), 1, figsize=(size[0]/dpi, size[1]/dpi), dpi=dpi)

        for id, ax in enumerate(axs):
            if id==0:
                ax.set_title("Values Distribution")
                legend_elements = [mpl.lines.Line2D([0], [0], color='#00000000', label='Mean', marker='o', markerfacecolor='m'),
                                   mpl.lines.Line2D([0], [0], color='c', label='Standard Deviation'),
                                   mpl.lines.Line2D([0], [0], color='w', label='Range')]
                ax.legend(handles=legend_elements, frameon=False, loc='upper left')
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_ylim([-1, 20])
            ax.set_xlim([self.min[id], self.max[id]])
            ax.set_yticks([0])
            ax.set_yticklabels([self.name[id]])
            ax.tick_params(axis='y', which='both', length=0)
            ax.errorbar(self.mean[id], 0, xerr=self.std_dev[id], fmt='mo', ecolor='c', label=self.name[id])

        plt.tight_layout(pad=0)

        canvas = plt.get_current_fig_manager().canvas
        canvas.draw()
        img = PIL.Image.frombytes('RGB',canvas.get_width_height(),canvas.tostring_rgb())
        plt.close()
        return img

