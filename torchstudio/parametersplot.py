import torchstudio.tcpcodec as tc
import inspect
import sys
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import PIL

def sorted(l,reverse=False):
    floats=True
    for x in l:
        try:
            float(x)
        except:
            floats=False
            break
    l.sort(key=float if floats else None,reverse=reverse)
    return l

#inspired by https://stackoverflow.com/questions/8230638/parallel-coordinates-plot-in-matplotlib
def plot_parameters(size, dpi,
                parameters=[], #parameters is a list of parameters
                values=[], #values is a list of list containing string values
                order=[]): #sorting order for each parameter(1 or -1)
    """Parameters Plot

    Usage:
        Click: invert parameter sorting order
    """
    #set up matplotlib renderer, style, figure and axis
    mpl.use('agg') #https://www.namingcrisis.net/post/2019/03/11/interactive-matplotlib-ipython/
    plt.style.use('dark_background')
    plt.rcParams.update({'font.size': 7})

    if len(parameters)<2:
        parameters=['Name', 'Validation\nMetric']

#    parameters=['Name', 'feature_channels', 'depth', 'Metric Value']
#    values=[['Model 1','32','3','.95'],['Model 2','24','4','.9'],['Model 3','16','3','.98'],['Model 4','16','3']]

    if len(order)<len(parameters):
        order=[1]*len(parameters)
    order[0]=-1

    param_values=[[] for i in range(len(parameters))]
    for value in values:
        for i, v in enumerate(value):
            if v not in param_values[i]:
                param_values[i].append(v)
    for i, v in enumerate(param_values):
        param_values[i]=sorted(param_values[i], True if order[i]==-1 else False)

    fig, host = plt.subplots(figsize=(size[0]/dpi, size[1]/dpi), dpi=dpi)

    axes = [host] + [host.twinx() for i in range(len(parameters)-1)]

    for i, ax in enumerate(axes):
        ax.set_ylim(0, 1)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_position(("axes", i / (len(parameters) - 1)))
        ax.spines['left'].set_color((0.2,0.2,0.2))
        ax.yaxis.set_tick_params(width=0)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_tick_params(width=0)
        ax.set_yticks([j/(len(param_values[i])-1) if len(param_values[i])>1 else .5 for j in range(len(param_values[i]))])
        ax.set_yticklabels(param_values[i])
    #first parameter is the model name, keep the set_ticks
    axes[0].yaxis.set_tick_params(width=1)
    #last parameter is the metric, let the colorbar do the metric
    axes[-1].yaxis.set_ticks_position('none')
    axes[-1].set_yticklabels([])
    axes[-1].spines['left'].set_visible(False)

    #set the colorbar for the metric
    if param_values[-1]:
        max_metric=min_metric=float(param_values[-1][0])
        for metric_value in param_values[-1]:
            min_metric=min(min_metric,float(metric_value))
            max_metric=max(max_metric,float(metric_value))
    else:
        max_metric=min_metric=0

    cmap = plt.get_cmap('viridis') # 'viridis' or 'rainbow'
    sc = host.scatter([0,0], [0,0], s=[0,0], c=[min_metric, max_metric], cmap=cmap)
    cbar = fig.colorbar(sc, ax=axes[-1], pad=0)
    cbar.outline.set_visible(False)
#    cbar.set_ticks([])

    #set horizontal axe settings
    host.set_xlim(0, len(parameters) - 1)
    host.set_xticks(range(len(parameters)))
    host.set_xticklabels(parameters)
    host.tick_params(axis='x', which='major', pad=7)
    host.spines['right'].set_visible(False)
    host.xaxis.tick_top()



    from matplotlib.path import Path
    import matplotlib.patches as patches
    import numpy as np
    for tokens in values:
        values_num=[]
        for i, token in enumerate(tokens):
            if i<len(parameters)-1:
                values_num.append(param_values[i].index(token)/(len(param_values[i])-1) if len(param_values[i])>1 else .5)
            else:
                values_num.append((float(token)-min_metric)/(max_metric-min_metric) if len(param_values[i])>1 and max_metric>min_metric else .5)

        # create bezier curves
        # for each axis, there will a control vertex at the point itself, one at 1/3rd towards the previous and one
        #   at one third towards the next axis; the first and last axis have one less control vertex
        # x-coordinate of the control vertices: at each integer (for the axes) and two inbetween
        # y-coordinate: repeat every point three times, except the first and last only twice
        verts = list(zip([x for x in np.linspace(0, len(values_num) - 1, len(values_num) * 3 - 2, endpoint=True)],
                         np.repeat(values_num, 3)[1:-1]))
        # for x,y in verts: host.plot(x, y, 'go') # to show the control points of the beziers
        codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(verts) - 1)]
        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor='none', lw=1, edgecolor=cmap(values_num[-1]) if len(values_num)==len(parameters) else (0.33, 0.33, 0.33), zorder=values_num[-1] if len(values_num)==len(parameters) else -1)
        host.add_patch(patch)

    plt.tight_layout(pad=0)

    canvas = plt.get_current_fig_manager().canvas
    canvas.draw()
    img = PIL.Image.frombytes('RGB',canvas.get_width_height(),canvas.tostring_rgb())
    plt.close()
    return img


resolution = (256,256, 96)

parameters=[]
values=[]
order=[]


app_socket = tc.connect()
while True:
    msg_type, msg_data = tc.recv_msg(app_socket)

    if msg_type == 'RequestDocumentation':
        tc.send_msg(app_socket, 'Documentation', tc.encode_strings(inspect.cleandoc(plot_parameters.__doc__)))

    if msg_type == 'SetResolution':
        resolution = tc.decode_ints(msg_data)

    if msg_type == 'SetParameters':
        parameters=tc.decode_strings(msg_data)

    if msg_type == 'ClearValues':
        values = []
    if msg_type == 'AppendValues':
        values.append(tc.decode_strings(msg_data))

    if msg_type == 'SetOrder':
        order=tc.decode_ints(msg_data)

    if msg_type == 'Render':
        if resolution[0]>0 and resolution[1]>0:
            img=plot_parameters(resolution[0:2],resolution[2],parameters,values,order)
            tc.send_msg(app_socket, 'ImageData', tc.encode_image(img))

    if msg_type == 'Exit':
        break
