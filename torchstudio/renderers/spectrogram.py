from torchstudio.modules import Renderer
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import PIL
import sys

class Spectrogram(Renderer):
    """Spectrogram Renderer
    Renders 3D tensors (CHW) with either complex values or real/imaginary channel couples

    Usage:
        Drag: pan
        Scroll: zoom
        Alt Scroll: adjust brightness
        Ctrl/Cmd Scroll: adjust gamma

    Args:
        colormap (str): Colormap to be used for single channel spectrograms.
            Values can be 'viridis', 'plasma', 'inferno', 'magma', 'cividis'
        colors: List of colors for each channel for multi channels spectrograms (looped if necessary)
        rotate (int): Number of time to rotate the bitmap by 90 degree (counter-clockwise)
        normalize (bool): Normalize values
    """
    def __init__(self, colormap='inferno', colors=['#ff0000','#00ff00','#0000ff','#ffff00','#00ffff','#ff00ff'], rotate=0, invert=False, normalize=False):
        super().__init__()
        self.colormap=colormap
        self.colors=colors
        self.rotate=rotate
        self.invert=invert
        self.normalize=normalize

    def render(self, title, tensor, size, dpi, shift=(0,0,0,0), scale=(1,1,1,1), input_tensors=[], target_tensor=None, labels=[]):
        #check dimensions
        if len(tensor.shape)!=3:
            print("Spectrogram renderer requires a 3D tensor, got a "+str(len(tensor.shape))+"D tensor.", file=sys.stderr)
            return None

        if np.iscomplexobj(tensor)==False and tensor.shape[0]%2!=0:
            #add missing channel (needs pairs to be interpred as complex channels)
            tensor=np.append(tensor, np.zeros((1,tensor.shape[1],tensor.shape[2])), axis=0)

        #convert complex spectrogram to amplitude spectrogram
        if np.iscomplexobj(tensor):
            tensor=np.absolute(tensor)
        else:
            tensor=np.absolute(tensor[::2]+1j*tensor[1::2])

        #flatten
        if tensor.shape[0]>1:
            zero = np.zeros((3,tensor.shape[1], tensor.shape[2]))
            for i in range(tensor.shape[0]):
                color=np.array(mpl.colors.to_rgb(self.colors[i%len(self.colors)])).reshape(3,1,1)
                zero+=tensor[[i]]*color
            tensor=zero

        if self.rotate>0:
            tensor=np.rot90(tensor, self.rotate, axes=(1, 2))

        tensor=tensor.astype(np.float32)
        if self.normalize:
            max_value=np.amax(tensor)
            if max_value>0:
                tensor=tensor/max_value

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
        if self.invert:
            ymin-=shift[1]/scale[1]*render_size[1]
            ymax-=shift[1]/scale[1]*render_size[1]
        else:
            ymin+=shift[1]/scale[1]*render_size[1]
            ymax+=shift[1]/scale[1]*render_size[1]

        #scale
        render_center=(xmin+render_size[0]/2.0,ymax+render_size[1]/2.0)
        xmin=render_center[0]-(render_size[0]/scale[1]/2.0)
        xmax=render_center[0]+(render_size[0]/scale[1]/2.0)
        ymax=render_center[1]-(render_size[1]/scale[1]/2.0)
        ymin=render_center[1]+(render_size[1]/scale[1]/2.0)
        if self.invert:
            ymin, ymax = ymax, ymin

        #render
        plt.axis(xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax)
        plt.imshow(tensor,cmap=self.colormap,vmin=0,vmax=255)
        plt.tight_layout(pad=0)
        plt.text(1,1,'\u00d7{:.2f}\n\u0263{:.2f}'.format(scale[0],1/scale[3]),transform=plt.gcf().transFigure,size=8,ha='right',va='top')

        canvas = plt.get_current_fig_manager().canvas
        canvas.draw()
        img = PIL.Image.frombytes('RGB',canvas.get_width_height(),canvas.tostring_rgb())
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
