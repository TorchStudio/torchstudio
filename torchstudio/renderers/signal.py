from torchstudio.modules import Renderer
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import PIL
import sys

class Signal(Renderer):
    """Signal Renderer
    Renders 2D tensors (CW)

    Usage:
        Drag: pan
        Scroll: zoom vertically
        Alt Scroll: zoom horizontally

    Args:
        min: Minimum value
        max: Maximum value
        auto_resize: Auto resize if needed
        colors: List of colors for each channel
        grid: Display grid
        legend: Display legend with more than one channel
    """
    def __init__(self, min=-1, max=1, auto_resize=False, colors=['#ff0000','#00ff00','#5050ff','#ffff00','#00ffff','#ff00ff'], grid=True, legend=True):
        super().__init__()
        self.min=min
        self.max=max
        self.auto_resize=auto_resize
        self.colors=colors
        self.grid=grid
        self.legend=legend

    def render(self, title, tensor, size, dpi, shift=(0,0,0,0), scale=(1,1,1,1), input_tensors=[], target_tensor=None, labels=[]):
        #check dimensions
        if len(tensor.shape)!=2:
            print("Signal renderer requires a 2D tensor, got a "+str(len(tensor.shape))+"D tensor.", file=sys.stderr)
            return None

        #set up matplotlib renderer, style, figure and axis
        mpl.use('agg') #https://www.namingcrisis.net/post/2019/03/11/interactive-matplotlib-ipython/
        plt.style.use('dark_background')
        plt.rcParams.update({'font.size': 7})
        plt.figure(figsize=(size[0]/dpi, size[1]/dpi), dpi=dpi)
        plt.title(title)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_color('#707070')
        plt.gca().spines['bottom'].set_color('#707070')

        #fit
        xmin=0
        xmax=tensor.shape[1]
        ymin=self.min if self.auto_resize==False else min(self.min,np.amin(tensor))
        ymax=self.max if self.auto_resize==False else max(self.max,np.amax(tensor))

        #shift
        render_size=(xmax-xmin,ymax-ymin)
        xmin-=shift[0]/scale[0]*render_size[0]
        xmax-=shift[0]/scale[0]*render_size[0]
        ymin-=shift[1]/scale[1]*render_size[1]
        ymax-=shift[1]/scale[1]*render_size[1]

        #scale
        render_center=(xmin+render_size[0]/2,ymin+render_size[1]/2)
        xmin=render_center[0]-(render_size[0]/scale[0]/2)
        xmax=render_center[0]+(render_size[0]/scale[0]/2)
        ymin=render_center[1]-(render_size[1]/scale[1]/2)
        ymax=render_center[1]+(render_size[1]/scale[1]/2)

        #render
        plt.axis(xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax)
        for i in range(tensor.shape[0]):
            plt.plot(tensor[[i]].flatten().tolist(),label=str(i) if i>=len(labels) else labels[i],color=self.colors[i%len(self.colors)])
        if(self.legend and tensor.shape[0]>1):
            plt.legend(bbox_to_anchor=(1, 1), loc='upper right', ncol=1, prop={'size': 8})
        if(self.grid):
            plt.grid(color = '#303030')
        plt.tight_layout(pad=0)

        canvas = plt.get_current_fig_manager().canvas
        canvas.draw()
        img = PIL.Image.frombytes('RGB',canvas.get_width_height(),canvas.tostring_rgb())
        plt.close()
        return img

#tensor = (np.random.rand(3,16)-.5)*2
#img = cw_2d(tensor, (400,300), 192)
#img.save('output.png')
