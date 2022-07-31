import torchstudio.tcpcodec as tc
import inspect
import sys
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import PIL

def plot_metrics(prefix, size, dpi, samples=100, labels=[],
                loss=[], loss_colors=[], loss_shift=(0,0), loss_scale=(1,1),
                metric=[], metric_colors=[], metric_shift=(0,0), metric_scale=(1,1)):
    """Metrics Plot

    Usage:
        Drag: pan
        Scroll: zoom vertically
    """
    #set up matplotlib renderer, style, figure and axis
    mpl.use('agg') #https://www.namingcrisis.net/post/2019/03/11/interactive-matplotlib-ipython/
    plt.style.use('dark_background')
    plt.rcParams.update({'font.size': 7})

    fig, [ax1, ax2] = plt.subplots(1 if size[0]>size[1] else 2, 2 if size[0]>size[1] else 1, figsize=(size[0]/dpi, size[1]/dpi), dpi=dpi)

    #LOSS
    ax1.set_title(prefix+"Loss")

    #fit
    loss_xmin=0
    loss_xmax=samples
    loss_ymin=0
    loss_ymax=1
    for l in loss:
        loss_xmax=max(loss_xmax,len(l))
#        if(len(l)>0):
#            loss_ymax=max(loss_ymax,max(l))

#    #shift
#    render_size=(loss_xmax-loss_xmin,loss_ymax-loss_ymin)
#    loss_xmin-=loss_shift[0]/loss_scale[0]*render_size[0]
#    loss_xmax-=loss_shift[0]/loss_scale[0]*render_size[0]
#    loss_ymin-=loss_shift[1]/loss_scale[1]*render_size[1]
#    loss_ymax-=loss_shift[1]/loss_scale[1]*render_size[1]

#    #scale
#    render_center=(loss_xmin+render_size[0]/2,loss_ymin+render_size[1]/2)
#    loss_xmin=render_center[0]-(render_size[0]/loss_scale[0]/2)
#    loss_xmax=render_center[0]+(render_size[0]/loss_scale[0]/2)
#    loss_ymin=render_center[1]-(render_size[1]/loss_scale[1]/2)
#    loss_ymax=render_center[1]+(render_size[1]/loss_scale[1]/2)

#    loss_xmin=max(0,loss_xmin)
#    loss_ymin=max(0,loss_ymin)

    loss_ymin-=loss_shift[1]/loss_scale[1]
    loss_ymax-=loss_shift[1]/loss_scale[1]
    loss_ymax=loss_ymax/loss_scale[1]

    ax1.axis(xmin=loss_xmin,xmax=loss_xmax,ymin=loss_ymin,ymax=loss_ymax)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color('#707070')
    ax1.spines['bottom'].set_color('#707070')
    for i in range(len(loss)):
        ax1.plot(loss[i],label=str(i) if i>=len(labels) else labels[i],color=loss_colors[i%len(loss_colors)])
    if labels and loss and loss[0]:
        ax1.legend(bbox_to_anchor=(1, 1), loc='upper right', ncol=1, prop={'size': 8})
    ax1.grid(color = '#303030')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

    #METRIC
    ax2.set_title(prefix+"Metric")

    #fit
    metric_xmin=0
    metric_xmax=samples
    metric_ymin=0
    metric_ymax=1
    for m in metric:
        metric_xmax=max(metric_xmax,len(m))

#    #shift
#    render_size=(metric_xmax-metric_xmin,metric_ymax-metric_ymin)
#    metric_xmin-=metric_shift[0]/metric_scale[0]*render_size[0]
#    metric_xmax-=metric_shift[0]/metric_scale[0]*render_size[0]
#    metric_ymin-=metric_shift[1]/metric_scale[1]*render_size[1]
#    metric_ymax-=metric_shift[1]/metric_scale[1]*render_size[1]

#    #scale
#    render_center=(metric_xmin+render_size[0]/2,metric_ymin+render_size[1]/2)
#    metric_xmin=render_center[0]-(render_size[0]/metric_scale[0]/2)
#    metric_xmax=render_center[0]+(render_size[0]/metric_scale[0]/2)
#    metric_ymin=render_center[1]-(render_size[1]/metric_scale[1]/2)
#    metric_ymax=render_center[1]+(render_size[1]/metric_scale[1]/2)

#    metric_xmin=max(0,metric_xmin)

    metric_ymin-=metric_shift[1]/metric_scale[1]
    metric_ymax-=metric_shift[1]/metric_scale[1]
    metric_ymin=(metric_ymin-metric_ymax)/metric_scale[1]+metric_ymax

    ax2.axis(xmin=metric_xmin,xmax=metric_xmax,ymin=metric_ymin,ymax=metric_ymax)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color('#707070')
    ax2.spines['bottom'].set_color('#707070')
    for i in range(len(metric)):
        ax2.plot(metric[i],color=metric_colors[i%len(metric_colors)])
    ax2.grid(color = '#303030')
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout(pad=0)

    canvas = plt.get_current_fig_manager().canvas
    canvas.draw()
    img = PIL.Image.frombytes('RGB',canvas.get_width_height(),canvas.tostring_rgb())
    plt.close()
    return img


prefix = ''
resolution = (256,256, 96)
samples=100
labels = []

loss=[]
loss_colors=[]
loss_shift = (0,0)
loss_scale = (1,1)

metric=[]
metric_colors=[]
metric_labels = []
metric_shift = (0,0)
metric_scale = (1,1)

app_socket = tc.connect()
while True:
    msg_type, msg_data = tc.recv_msg(app_socket)

    if msg_type == 'RequestDocumentation':
        tc.send_msg(app_socket, 'Documentation', tc.encode_strings(inspect.cleandoc(plot_metrics.__doc__)))
    if msg_type == 'SetPrefix':
        prefix=tc.decode_strings(msg_data)[0]

    if msg_type == 'SetResolution':
        resolution = tc.decode_ints(msg_data)

    if msg_type == 'NumSamples':
        samples = tc.decode_ints(msg_data)[0]
    if msg_type == 'SetLabels':
        labels=tc.decode_strings(msg_data)

    if msg_type == 'ClearLoss':
        loss=[]
    if msg_type == 'AppendLoss':
        loss.append(tc.decode_floats(msg_data))
    if msg_type == 'SetLossColors':
        loss_colors=tc.decode_strings(msg_data)
    if msg_type == 'SetLossShift':
        loss_shift = tc.decode_floats(msg_data)
    if msg_type == 'SetLossScale':
        loss_scale = tc.decode_floats(msg_data)

    if msg_type == 'ClearMetric':
        metric=[]
    if msg_type == 'AppendMetric':
        metric.append(tc.decode_floats(msg_data))
    if msg_type == 'SetMetricColors':
        metric_colors=tc.decode_strings(msg_data)
    if msg_type == 'SetMetricShift':
        metric_shift = tc.decode_floats(msg_data)
    if msg_type == 'SetMetricScale':
        metric_scale = tc.decode_floats(msg_data)

    if msg_type == 'Render':
        if resolution[0]>0 and resolution[1]>0:
            img=plot_metrics(prefix,resolution[0:2],resolution[2],samples,labels,loss,loss_colors,loss_shift,loss_scale,metric,metric_colors,metric_shift,metric_scale)
            tc.send_msg(app_socket, 'ImageData', tc.encode_image(img))

    if msg_type == 'Exit':
        break
