from torchstudio.modules import Renderer
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import PIL
import sys

class BoundingBox(Renderer):
    """Bounding Box Renderer
    Renders 1D or 2D tensors with last dimension being min_x, min_y, max_x, max_y

    Usage:
        Drag: pan
        Scroll: zoom
        Alt Scroll: adjust brightness
        Ctrl/Cmd Scroll: adjust gamma

    Args:
        colors: List of colors for each bounding box channel
        input_tensor_id: Id of the input tensor to use as background bitmap.
            Values can be 'auto' or an integer number.
        colormap (str): Colormap to be used for single channel bitmaps.
            Values can be 'viridis', 'plasma', 'inferno', 'magma', 'cividis'
    """
    def __init__(self, colors=['#ff0000','#00ff00','#5050ff','#ffff00','#00ffff','#ff00ff'], input_tensor_id='auto', colormap='inferno'):
        super().__init__()
        self.colors=colors
        self.input_tensor_id=input_tensor_id
        self.colormap=colormap

    def render(self, title, tensor, size, dpi, shift=(0,0,0,0), scale=(1,1,1,1), input_tensors=[], target_tensor=None, labels=[]):
        #check dimensions
        if len(tensor.shape)<1 or len(tensor.shape)>2 or tensor.shape[-1]!=4:
            print("Bounding box renderer requires a 1D or 2D tensor with last dimension of size 4, got a "+str(len(tensor.shape))+"D tensor with last dimension of size "+str(tensor.shape[-1]), file=sys.stderr)
            return None
        bb_tensor=tensor

        input_tensor_id=0
        if self.input_tensor_id=='auto':
            for id,input_tensor in enumerate(input_tensors):
                if len(input_tensor.shape)==3:
                    input_tensor_id=id
                    break

        if len(input_tensors[input_tensor_id].shape)!=3:
            print("Input tensor does not match, 3D tensor (CHW) required", file=sys.stderr)
            return None

        tensor=input_tensors[input_tensor_id]

        #flatten
        if tensor.shape[0]==2: #2 channels, pad with a third channel
            zero = np.zeros((1,tensor.shape[1], tensor.shape[2]))
            tensor = np.concatenate((tensor,zero),0)
        if tensor.shape[0]>3: #more than 3 channels, add additional channels into the first 3
            for i in range(3,tensor.shape[0]):
                tensor[[i%3]]+=tensor[[i]]
                if i%6>=3: #add R G B R G B to RG GB BR R G B
                    tensor[[(i+1)%3]]+=tensor[[i]]
            tensor=tensor[[0,1,2]]

        #apply brightness, gamma and conversion to uint8, then transform CHW to HWC
        tensor = np.multiply(np.clip(np.power(tensor*scale[0],1/scale[3]),0,1),255).astype(np.uint8)
        tensor = tensor.transpose((1, 2, 0))

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
        tensor_ratio=tensor.shape[1]/tensor.shape[0]
        render_ratio=size[0]/size[1]
        if tensor_ratio>render_ratio:
            xmin=0
            xmax=tensor.shape[1]
            ymin=tensor.shape[1]/render_ratio
            ymax=0
            ymax-=(ymin-tensor.shape[0])/2.0
            ymin+=ymax
        else:
            ymax=0
            ymin=tensor.shape[0]
            xmin=0
            xmax=tensor.shape[0]*render_ratio
            xmin-=(xmax-tensor.shape[1])/2.0
            xmax+=xmin

        #shift
        render_size=(xmax-xmin,ymin-ymax)
        xmin-=shift[0]/scale[1]*render_size[0]
        xmax-=shift[0]/scale[1]*render_size[0]
        ymin+=shift[1]/scale[1]*render_size[1]
        ymax+=shift[1]/scale[1]*render_size[1]

        #scale
        render_center=(xmin+render_size[0]/2.0,ymax+render_size[1]/2.0)
        xmin=render_center[0]-(render_size[0]/scale[1]/2.0)
        xmax=render_center[0]+(render_size[0]/scale[1]/2.0)
        ymax=render_center[1]-(render_size[1]/scale[1]/2.0)
        ymin=render_center[1]+(render_size[1]/scale[1]/2.0)

        #render
        plt.axis(xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax)
        plt.imshow(tensor,cmap=self.colormap,vmin=0,vmax=255)
        plt.tight_layout(pad=0)
        plt.text(1,1,'\u00d7{:.2f}\n\u0263{:.2f}'.format(scale[0],1/scale[3]),transform=plt.gcf().transFigure,size=8,ha='right',va='top')

        #render bounding box on top
        if len(bb_tensor.shape)==1:
            plt.gca().add_patch(Rectangle((bb_tensor[0],bb_tensor[1]),bb_tensor[2]-bb_tensor[0],bb_tensor[3]-bb_tensor[1],linewidth=1,edgecolor=self.colors[0],facecolor='none'))
        else:
            for channel in range(bb_tensor.shape[0]):
                plt.gca().add_patch(Rectangle((bb_tensor[channel][0],bb_tensor[channel][1]),bb_tensor[channel][2]-bb_tensor[channel][0],bb_tensor[channel][3]-bb_tensor[channel][1],linewidth=1,edgecolor=self.colors[channel%len(self.colors)],facecolor='none'))

        canvas = plt.get_current_fig_manager().canvas
        canvas.draw()
        img = PIL.Image.frombytes('RGBA',canvas.get_width_height(),canvas.buffer_rgba())
        plt.close()
        return img

#from PIL import ImageDraw
#img  = PIL.Image.new( mode = "RGB", size = (512, 512), color = (209, 123, 193) )
#draw = PIL.ImageDraw.Draw(img)
#font = PIL.ImageFont.truetype("arial.ttf", 72)
#draw.text((40, 100),"Sample Text\nSecond Line\nThird Line\nEnd",fill=(255,255,255), font=font)
#tensor = (np.array(img).astype(np.float32)/255).transpose((2,0,1))[[0,1]]
#print(tensor[0][0][0],tensor.dtype)
#img = chw_3d(tensor, (400,300), 192)
#img.save('output.png')
