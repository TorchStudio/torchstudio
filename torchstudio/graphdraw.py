import torchstudio.tcpcodec as tc
import os
import graphviz
import copy

#from https://raw.githubusercontent.com/pytorch/pytorch/master/docs/source/torch.rst
#from https://raw.githubusercontent.com/pytorch/pytorch/master/docs/source/nn.rst
#from https://raw.githubusercontent.com/pytorch/pytorch/master/docs/source/nn.functional.rst

creation_ops="""
tensor
sparse_coo_tensor
as_tensor
as_strided
from_numpy
frombuffer
zeros
zeros_like
ones
ones_like
arange
range
linspace
logspace
eye
empty
empty_like
empty_strided
full
full_like
quantize_per_tensor
quantize_per_channel
dequantize
complex
polar
heaviside
"""

manipulation_ops="""
cat
concat
conj
chunk
dsplit
column_stack
dstack
gather
hsplit
hstack
index_select
masked_select
movedim
moveaxis
narrow
nonzero
permute
reshape
row_stack
scatter
scatter_add
split
squeeze
stack
swapaxes
swapdims
t
take
take_along_dim
tensor_split
tile
transpose
unbind
unsqueeze
vsplit
vstack
where
"""

convolution_ops="""
nn.Conv1d
nn.Conv2d
nn.Conv3d
nn.ConvTranspose1d
nn.ConvTranspose2d
nn.ConvTranspose3d
nn.LazyConv1d
nn.LazyConv2d
nn.LazyConv3d
nn.LazyConvTranspose1d
nn.LazyConvTranspose2d
nn.LazyConvTranspose3d
nn.Unfold
nn.Fold
conv1d
conv2d
conv3d
conv_transpose1d
conv_transpose2d
conv_transpose3d
unfold
fold
"""

pooling_ops="""
nn.MaxPool1d
nn.MaxPool2d
nn.MaxPool3d
nn.MaxUnpool1d
nn.MaxUnpool2d
nn.MaxUnpool3d
nn.AvgPool1d
nn.AvgPool2d
nn.AvgPool3d
nn.FractionalMaxPool2d
nn.FractionalMaxPool3d
nn.LPPool1d
nn.LPPool2d
nn.AdaptiveMaxPool1d
nn.AdaptiveMaxPool2d
nn.AdaptiveMaxPool3d
nn.AdaptiveAvgPool1d
nn.AdaptiveAvgPool2d
nn.AdaptiveAvgPool3d
avg_pool1d
avg_pool2d
avg_pool3d
max_pool1d
max_pool2d
max_pool3d
max_unpool1d
max_unpool2d
max_unpool3d
lp_pool1d
lp_pool2d
adaptive_max_pool1d
adaptive_max_pool2d
adaptive_max_pool3d
adaptive_avg_pool1d
adaptive_avg_pool2d
adaptive_avg_pool3d
fractional_max_pool2d
fractional_max_pool3d
"""

activation_ops="""
nn.ELU
nn.Hardshrink
nn.Hardsigmoid
nn.Hardtanh
nn.Hardswish
nn.LeakyReLU
nn.LogSigmoid
nn.MultiheadAttention
nn.PReLU
nn.ReLU
nn.ReLU6
nn.RReLU
nn.SELU
nn.CELU
nn.GELU
nn.Sigmoid
nn.SiLU
nn.Mish
nn.Softplus
nn.Softshrink
nn.Softsign
nn.Tanh
nn.Tanhshrink
nn.Threshold
nn.GLU
nn.Softmin
nn.Softmax
nn.Softmax2d
nn.LogSoftmax
nn.AdaptiveLogSoftmaxWithLoss
threshold
threshold_
relu
relu_
hardtanh
hardtanh_
hardswish
relu6
elu
elu_
selu
celu
leaky_relu
leaky_relu_
prelu
rrelu
rrelu_
glu
gelu
logsigmoid
hardshrink
tanhshrink
softsign
softplus
softmin
softmax
softshrink
gumbel_softmax
log_softmax
tanh
sigmoid
hardsigmoid
silu
mish
batch_norm
group_norm
instance_norm
layer_norm
local_response_norm
normalize
"""

normalization_ops="""
nn.BatchNorm1d
nn.BatchNorm2d
nn.BatchNorm3d
nn.LazyBatchNorm1d
nn.LazyBatchNorm2d
nn.LazyBatchNorm3d
nn.GroupNorm
nn.SyncBatchNorm
nn.InstanceNorm1d
nn.InstanceNorm2d
nn.InstanceNorm3d
nn.LazyInstanceNorm1d
nn.LazyInstanceNorm2d
nn.LazyInstanceNorm3d
nn.LayerNorm
nn.LocalResponseNorm
"""

linear_ops="""
nn.Identity
nn.Linear
nn.Bilinear
nn.LazyLinear
linear
bilinear
"""

dropout_ops="""
nn.Dropout
nn.Dropout2d
nn.Dropout3d
nn.AlphaDropout
nn.FeatureAlphaDropout
dropout
alpha_dropout
feature_alpha_dropout
dropout2d
dropout3d
"""

vision_ops="""
nn.PixelShuffle
nn.PixelUnshuffle
nn.Upsample
nn.UpsamplingNearest2d
nn.UpsamplingBilinear2d
pixel_shuffle
pixel_unshuffle
pad
interpolate
upsample
upsample_nearest
upsample_bilinear
grid_sample
affine_grid
"""

math_ops="""
abs
absolute
acos
arccos
acosh
arccosh
add
addcdiv
addcmul
angle
asin
arcsin
asinh
arcsinh
atan
arctan
atanh
arctanh
atan2
bitwise_not
bitwise_and
bitwise_or
bitwise_xor
bitwise_left_shift
bitwise_right_shift
ceil
clamp
clip
conj_physical
copysign
cos
cosh
deg2rad
div
divide
digamma
erf
erfc
erfinv
exp
exp2
expm1
fake_quantize_per_channel_affine
fake_quantize_per_tensor_affine
fix
float_power
floor
floor_divide
fmod
frac
frexp
gradient
imag
ldexp
lerp
lgamma
log
log10
log1p
log2
logaddexp
logaddexp2
logical_and
logical_not
logical_or
logical_xor
logit
hypot
i0
igamma
igammac
mul
multiply
mvlgamma
nan_to_num
neg
negative
nextafter
polygamma
positive
pow
quantized_batch_norm
quantized_max_pool1d
quantized_max_pool2d
rad2deg
real
reciprocal
remainder
round
rsqrt
sigmoid
sign
sgn
signbit
sin
sinc
sinh
sqrt
square
sub
subtract
tan
tanh
true_divide
trunc
xlogy
"""

reduction_ops="""
argmax
argmin
amax
amin
aminmax
all
any
max
min
dist
logsumexp
mean
nanmean
median
nanmedian
mode
norm
nansum
prod
quantile
nanquantile
std
std_mean
sum
unique
unique_consecutive
var
var_mean
count_nonzero
"""

comparison_ops="""
allclose
argsort
eq
equal
ge
greater_equal
gt
greater
isclose
isfinite
isin
isinf
isposinf
isneginf
isnan
isreal
kthvalue
le
less_equal
lt
less
maximum
minimum
fmax
fmin
ne
not_equal
sort
topk
msort
"""

other_ops="""
atleast_1d
atleast_2d
atleast_3d
bincount
block_diag
broadcast_tensors
broadcast_to
broadcast_shapes
bucketize
cartesian_prod
cdist
clone
combinations
corrcoef
cov
cross
cummax
cummin
cumprod
cumsum
diag
diag_embed
diagflat
diagonal
diff
einsum
flatten
flip
fliplr
flipud
kron
rot90
gcd
histc
histogram
meshgrid
lcm
logcumsumexp
ravel
renorm
repeat_interleave
roll
searchsorted
tensordot
trace
tril
tril_indices
triu
triu_indices
vander
view_as_real
view_as_complex
resolve_conj
resolve_neg
"""

creation_ops=[op.split('.')[-1] for op in creation_ops.split('\n') if op]
manipulation_ops=[op.split('.')[-1] for op in manipulation_ops.split('\n') if op]
convolution_ops=[op.split('.')[-1] for op in convolution_ops.split('\n') if op]
pooling_ops=[op.split('.')[-1] for op in pooling_ops.split('\n') if op]
activation_ops=[op.split('.')[-1] for op in activation_ops.split('\n') if op]
normalization_ops=[op.split('.')[-1] for op in normalization_ops.split('\n') if op]
linear_ops=[op.split('.')[-1] for op in linear_ops.split('\n') if op]
dropout_ops=[op.split('.')[-1] for op in dropout_ops.split('\n') if op]
vision_ops=[op.split('.')[-1] for op in vision_ops.split('\n') if op]
math_ops=[op.split('.')[-1] for op in math_ops.split('\n') if op]
reduction_ops=[op.split('.')[-1] for op in reduction_ops.split('\n') if op]
comparison_ops=[op.split('.')[-1] for op in comparison_ops.split('\n') if op]
other_ops=[op.split('.')[-1] for op in other_ops.split('\n') if op]

ops_color={}
ops_color.update({op : '#707070' for op in creation_ops})
ops_color.update({op : '#803080' for op in manipulation_ops})
ops_color.update({op : '#3080c0' for op in convolution_ops})
ops_color.update({op : '#109010' for op in pooling_ops})
ops_color.update({op : '#b03030' for op in activation_ops})
ops_color.update({op : '#6080a0' for op in normalization_ops})
ops_color.update({op : '#30b060' for op in linear_ops})
ops_color.update({op : '#c09020' for op in dropout_ops})
ops_color.update({op : '#509090' for op in vision_ops})
ops_color.update({op : '#d06000' for op in math_ops})
ops_color.update({op : '#906000' for op in reduction_ops})
ops_color.update({op : '#90a060' for op in comparison_ops})
ops_color.update({op : '#b03070' for op in other_ops})

text_color="#f0f0f0"
default_color="#908070";
input_color="#606060"
output_color="#808080"

link_text_color="#d0d0d0"
link_color="#a0a0a0"

app_socket = tc.connect()
while True:
    msg_type, msg_data = tc.recv_msg(app_socket)

    if msg_type == 'SetGraph':
        nodes=eval(str(msg_data,'utf-8'))

    if msg_type == 'Render':
        batch, legend = tc.decode_ints(msg_data)
        filtered_nodes=copy.deepcopy(nodes)

        #merge referenced getitems
        for id, node in nodes.items():
            filtered_nodes[id]['input_shape']={}
            for input in node['inputs']:
                if nodes[input]['op_module']=='operator' and nodes[input]['op']=='getitem':
                    for sub_input in nodes[input]['inputs']:
                        filtered_nodes[id]['inputs'].remove(input)
                        filtered_nodes[id]['inputs'].append(sub_input)
                        filtered_nodes[id]['input_shape'][sub_input]=nodes[input]['output_shape']
                    del filtered_nodes[input]
        #del non-referenced getitems
        nodes=copy.deepcopy(filtered_nodes)
        for id, node in nodes.items():
            if node['op_module']=='operator' and node['op']=='getitem':
                del filtered_nodes[id]
        graph = graphviz.Digraph(graph_attr={'peripheries':'0', 'dpi': '0.0', 'bgcolor': 'transparent', 'ranksep': '0.25', 'margin': '0'},
                                      node_attr={'style': 'filled', 'shape': 'Mrecord', 'fillcolor': default_color, 'penwidth':'0', 'fontcolor': text_color,'fontsize':'20', 'fontname':'Source Code Pro'},
                                      edge_attr={'color': link_color, 'fontcolor': link_text_color,'fontsize':'16', 'fontname':'Source Code Pro'})
        inputs_graph = graphviz.Digraph(name='cluster_input', node_attr={'shape': 'oval', 'fillcolor': input_color, 'margin': '0'})
        outputs_graph = graphviz.Digraph(name='cluster_output', node_attr={'shape': 'oval', 'fillcolor': output_color, 'margin': '0'})

        for id, node in filtered_nodes.items():
            if node['type']=='input':
                inputs_graph.node(id, '<<b>'+node['name']+'</b>>', tooltip=node['name'])
            elif node['type']=='output':
                outputs_graph.node(id, '<<b>'+node['name']+'</b>>')
            else:
                if node['op'] in ops_color:
                    node_color=ops_color[node['op']]
                else:
                    node_color=default_color
                label=node['op']
                label_start, label_end=('<<i>','</i>>') if node['type']=='function' else ('','')
                if node['op']=='':
                    node_tooltip=node['name']
                else:
                    node_tooltip=(node['name']+' = ' if node['type']=='module' else '')+node['op_module']+"."+node['op']+'('+node['params']+')'
                graph.node(id, label_start+label+label_end, {'fillcolor': node_color, 'tooltip': node_tooltip})

        graph.subgraph(inputs_graph)
        graph.subgraph(outputs_graph)

        for id, node in filtered_nodes.items():
            for input in node['inputs']:
                output_shape=filtered_nodes[input]['output_shape']
                if input in node['input_shape']:
                    output_shape=node['input_shape'][input]
                if batch==1:
                    output_shape=('N,' if output_shape else 'N')+output_shape
                graph.edge(input,id,"  "+output_shape.replace(',','\u00d7')) #replace comma by multiplication sign

        if legend==1:
            with graph.subgraph(name='cluster_legend', node_attr={'shape': 'box', 'margin':'0', 'fontsize':'16', 'style':''}) as legend:
                table= '<tr><td bgcolor="'+input_color+'">Input</td>'
                table+='    <td bgcolor="'+ops_color[creation_ops[0]]+'">Creation</td></tr>'
                table+='<tr><td bgcolor="'+ops_color[manipulation_ops[0]]+'">Manipulation</td>'
                table+='    <td bgcolor="'+ops_color[convolution_ops[0]]+'">Convolution</td></tr>'
                table+='<tr><td bgcolor="'+ops_color[pooling_ops[0]]+'">Pooling</td>'
                table+='    <td bgcolor="'+ops_color[activation_ops[0]]+'">Activation</td></tr>'
                table+='<tr><td bgcolor="'+ops_color[normalization_ops[0]]+'">Normalization</td>'
                table+='    <td bgcolor="'+ops_color[linear_ops[0]]+'">Linear</td></tr>'
                table+='<tr><td bgcolor="'+ops_color[dropout_ops[0]]+'">Dropout</td>'
                table+='    <td bgcolor="'+ops_color[vision_ops[0]]+'">Vision</td></tr>'
                table+='<tr><td bgcolor="'+ops_color[math_ops[0]]+'">Math</td>'
                table+='    <td bgcolor="'+ops_color[reduction_ops[0]]+'">Reduction</td></tr>'
                table+='<tr><td bgcolor="'+ops_color[comparison_ops[0]]+'">Comparison</td>'
                table+='    <td bgcolor="'+ops_color[other_ops[0]]+'">Other</td></tr>'
                table+='<tr><td bgcolor="'+default_color+'">Unknown</td>'
                table+='    <td bgcolor="'+output_color+'">Output</td></tr>'
                legend.node('legend', '<<table border="0" cellspacing="2">'+table+'</table>>')

        svg=graph.pipe(format='svg')
        tc.send_msg(app_socket, 'SVGData', svg)

#        with open('/Users/divide/Documents/output.txt','w') as file:
#            print(graph.source, file=file)
#        with open('/Users/divide/Documents/output.svg','w') as file:
#            print(str(svg, 'utf-8'), file=file)
#        with open('/Users/divide/Documents/output.png','wb') as file:
#            file.write(graph.pipe(format='png'))

    if msg_type == 'Exit':
        break

