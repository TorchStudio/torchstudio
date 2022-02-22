from torchstudio.modules import Renderer
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import PIL
import sys

class Labels(Renderer):
    """Labels Renderer
    Renders 0D or 1D tensors

    Usage:
        Drag: pan
        Scroll: zoom

    Args:
        normalize_function (str): Function to compute the normalized score of the prediction
            Can be 'auto', 'none', 'softmax', 'exp', 'sigmoid'
        horizontal_lock (bool): Lock horizontal range to [0;1]
    """
    def __init__(self, normalize_function='auto', horizontal_lock=True):
        super().__init__()
        self.normalize_function=normalize_function
        self.horizontal_lock=horizontal_lock

    def render(self, title, tensor, size, dpi, shift=(0,0,0,0), scale=(1,1,1,1), input_tensors=[], target_tensor=None, labels=[]):
        #check dimensions
        if len(tensor.shape)>1:
            print("Labels renderer requires a 0D or 1D tensor, got a "+str(len(tensor.shape))+"D tensor.", file=sys.stderr)
            return None

        #set up matplotlib renderer, style, figure and axis
        mpl.use('agg') #https://www.namingcrisis.net/post/2019/03/11/interactive-matplotlib-ipython/
        plt.style.use('dark_background')
        plt.rcParams.update({'font.size': 7})
        plt.figure(figsize=(size[0]/dpi, size[1]/dpi), dpi=dpi)
        plt.title(title)

        if len(tensor.shape)==0:
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(False)
            plt.gca().get_xaxis().set_visible(False)
            plt.gca().get_yaxis().set_visible(False)

            #render
            if "int" in str(tensor.dtype) and tensor.item()<len(labels):
                plt.text(0.5,0.5,labels[tensor.item()],transform=plt.gcf().transFigure,size=20,ha='center',va='center')
            else:
                plt.text(0.5,0.5,str(tensor.item()),transform=plt.gcf().transFigure,size=20,ha='center',va='center')
            plt.tight_layout(pad=0)

        else:
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_color('#707070')
            plt.gca().spines['bottom'].set_color('#707070')

            #fit
            xmin=0
            xmax=1
            ymin=min(tensor.shape[0],10)-.5
            ymax=-.5

            #shift
            render_size=(xmax-xmin,ymin-ymax)
            if self.horizontal_lock==False:
                xmin-=shift[0]/scale[0]*render_size[0]
                xmax-=shift[0]/scale[0]*render_size[0]
            ymin+=shift[1]/scale[1]*render_size[1]
            ymax+=shift[1]/scale[1]*render_size[1]

            #scale
            render_center=(xmin+render_size[0]/2,ymax+render_size[1]/2)
            if self.horizontal_lock==False:
                xmin=render_center[0]-(render_size[0]/scale[0]/2)
                xmax=render_center[0]+(render_size[0]/scale[0]/2)
            ymax=render_center[1]-(render_size[1]/scale[1]/2)
            ymin=render_center[1]+(render_size[1]/scale[1]/2)

            normalize_function=self.normalize_function
            if normalize_function=='auto':
                normalize_function='none'
                if target_tensor is not None:
                    min_value=np.amin(tensor)
                    max_value=np.amax(tensor)
                    if len(target_tensor.shape)==0:
                        #multiclass
                        if min_value<0 and max_value>0: #no logsoftmax was applied, apply softmax
                            normalize_function='softmax'
                        if min_value<0 and max_value<=0: #logsoftmax was applied, rectify with exp
                            normalize_function='exp'
                    if len(target_tensor.shape)==1 and target_tensor.shape[0]==len(labels):
                        #multiclass multilabel
                        if min_value<0 or max_value>1: #no sigmoid was applied
                            normalize_function='sigmoid'

            if normalize_function=='softmax':
                tensor=(np.exp(tensor)/np.sum(np.exp(tensor), axis=0))
            if normalize_function=='exp':
                tensor=np.exp(tensor)
            if normalize_function=='sigmoid':
                tensor=1/(1+np.exp(-tensor))


            #render
            start=max(min(int(ymax),tensor.shape[0]-1),0)
            end=max(min(int(ymin+1),tensor.shape[0]-1),0)
            cropped_tensor=tensor[start:end+1]
            if len(labels)==tensor.shape[0]:
                cropped_labels=labels[start:end+1]


            plt.gca().set_yticks(range(start,end+1)) #force one tick/label per bar first
            plt.axis(xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax)
            plt.barh(range(start,end+1),cropped_tensor.flatten().tolist(), align='center')
            if len(labels)==tensor.shape[0]:
                plt.gca().set_yticklabels(cropped_labels)


#            plt.gca().set_yticks(range(tensor.shape[0])) #force one tick/label per bar first
#            plt.axis(xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax)
#            plt.barh(range(tensor.shape[0]),tensor.flatten().tolist(), align='center')
#            if len(labels)==tensor.shape[0]:
#                plt.gca().set_yticklabels(labels)

            plt.tight_layout(pad=0)

        canvas = plt.get_current_fig_manager().canvas
        canvas.draw()
        img = PIL.Image.frombytes('RGB',canvas.get_width_height(),canvas.tostring_rgb())
        plt.close()
        return img

#tensor = np.random.randint(0,15,size=())
#img = value_0d(tensor, (400,300), 192, labels=['zero','one','two','three','four','five','six','seven','height','nine','ten','eleven','twelve','thirteen','fourteen','fifteen'])
#img.save('output.png')
