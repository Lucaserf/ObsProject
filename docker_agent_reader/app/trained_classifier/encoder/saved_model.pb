��
�!�!
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
Z
BroadcastTo

input"T
shape"Tidx
output"T"	
Ttype"
Tidxtype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
R
Einsum
inputs"T*N
output"T"
equationstring"
Nint(0"	
Ttype
,
Exp
x"T
y"T"
Ttype:

2
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
�
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
$

LogicalAnd
x

y

z
�
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
U
NotEqual
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(�
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	
�
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
�
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	�
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
2
Round
x"T
y"T"
Ttype:
2
	
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
d
Shape

input"T&
output"out_type��out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	�
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
2
StopGradient

input"T
output"T"	
Ttype
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.13.02v2.13.0-rc2-7-g1cb1a030a628��
s
encoding/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameencoding/bias
l
!encoding/bias/Read/ReadVariableOpReadVariableOpencoding/bias*
_output_shapes	
:�*
dtype0
{
encoding/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*�* 
shared_nameencoding/kernel
t
#encoding/kernel/Read/ReadVariableOpReadVariableOpencoding/kernel*
_output_shapes
:	*�*
dtype0
v
encoding/bias_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:** 
shared_nameencoding/bias_1
o
#encoding/bias_1/Read/ReadVariableOpReadVariableOpencoding/bias_1*
_output_shapes
:**
dtype0

encoding/kernel_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:	�**"
shared_nameencoding/kernel_1
x
%encoding/kernel_1/Read/ReadVariableOpReadVariableOpencoding/kernel_1*
_output_shapes
:	�**
dtype0
s
encoding/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameencoding/beta
l
!encoding/beta/Read/ReadVariableOpReadVariableOpencoding/beta*
_output_shapes	
:�*
dtype0
u
encoding/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameencoding/gamma
n
"encoding/gamma/Read/ReadVariableOpReadVariableOpencoding/gamma*
_output_shapes	
:�*
dtype0
w
encoding/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:�* 
shared_nameencoding/beta_1
p
#encoding/beta_1/Read/ReadVariableOpReadVariableOpencoding/beta_1*
_output_shapes	
:�*
dtype0
y
encoding/gamma_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:�*!
shared_nameencoding/gamma_1
r
$encoding/gamma_1/Read/ReadVariableOpReadVariableOpencoding/gamma_1*
_output_shapes	
:�*
dtype0
�
3encoding/multi_head_attention/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*D
shared_name53encoding/multi_head_attention/attention_output/bias
�
Gencoding/multi_head_attention/attention_output/bias/Read/ReadVariableOpReadVariableOp3encoding/multi_head_attention/attention_output/bias*
_output_shapes	
:�*
dtype0
�
5encoding/multi_head_attention/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: �*F
shared_name75encoding/multi_head_attention/attention_output/kernel
�
Iencoding/multi_head_attention/attention_output/kernel/Read/ReadVariableOpReadVariableOp5encoding/multi_head_attention/attention_output/kernel*#
_output_shapes
: �*
dtype0
�
(encoding/multi_head_attention/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *9
shared_name*(encoding/multi_head_attention/value/bias
�
<encoding/multi_head_attention/value/bias/Read/ReadVariableOpReadVariableOp(encoding/multi_head_attention/value/bias*
_output_shapes

: *
dtype0
�
*encoding/multi_head_attention/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:� *;
shared_name,*encoding/multi_head_attention/value/kernel
�
>encoding/multi_head_attention/value/kernel/Read/ReadVariableOpReadVariableOp*encoding/multi_head_attention/value/kernel*#
_output_shapes
:� *
dtype0
�
&encoding/multi_head_attention/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *7
shared_name(&encoding/multi_head_attention/key/bias
�
:encoding/multi_head_attention/key/bias/Read/ReadVariableOpReadVariableOp&encoding/multi_head_attention/key/bias*
_output_shapes

: *
dtype0
�
(encoding/multi_head_attention/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:� *9
shared_name*(encoding/multi_head_attention/key/kernel
�
<encoding/multi_head_attention/key/kernel/Read/ReadVariableOpReadVariableOp(encoding/multi_head_attention/key/kernel*#
_output_shapes
:� *
dtype0
�
(encoding/multi_head_attention/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *9
shared_name*(encoding/multi_head_attention/query/bias
�
<encoding/multi_head_attention/query/bias/Read/ReadVariableOpReadVariableOp(encoding/multi_head_attention/query/bias*
_output_shapes

: *
dtype0
�
*encoding/multi_head_attention/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:� *;
shared_name,*encoding/multi_head_attention/query/kernel
�
>encoding/multi_head_attention/query/kernel/Read/ReadVariableOpReadVariableOp*encoding/multi_head_attention/query/kernel*#
_output_shapes
:� *
dtype0
�
input_embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	U�*+
shared_nameinput_embedding/embeddings
�
.input_embedding/embeddings/Read/ReadVariableOpReadVariableOpinput_embedding/embeddings*
_output_shapes
:	U�*
dtype0
�
input_embedding/embeddings_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:
�'�*-
shared_nameinput_embedding/embeddings_1
�
0input_embedding/embeddings_1/Read/ReadVariableOpReadVariableOpinput_embedding/embeddings_1* 
_output_shapes
:
�'�*
dtype0
t
z_log_var/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namez_log_var/bias
m
"z_log_var/bias/Read/ReadVariableOpReadVariableOpz_log_var/bias*
_output_shapes
:**
dtype0
}
z_log_var/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�**!
shared_namez_log_var/kernel
v
$z_log_var/kernel/Read/ReadVariableOpReadVariableOpz_log_var/kernel*
_output_shapes
:	�**
dtype0
n
z_mean/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namez_mean/bias
g
z_mean/bias/Read/ReadVariableOpReadVariableOpz_mean/bias*
_output_shapes
:**
dtype0
w
z_mean/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�**
shared_namez_mean/kernel
p
!z_mean/kernel/Read/ReadVariableOpReadVariableOpz_mean/kernel*
_output_shapes
:	�**
dtype0
�
serving_default_input_word_idsPlaceholder*'
_output_shapes
:���������U*
dtype0*
shape:���������U
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_word_idsinput_embedding/embeddings_1input_embedding/embeddings*encoding/multi_head_attention/query/kernel(encoding/multi_head_attention/query/bias(encoding/multi_head_attention/key/kernel&encoding/multi_head_attention/key/bias*encoding/multi_head_attention/value/kernel(encoding/multi_head_attention/value/bias5encoding/multi_head_attention/attention_output/kernel3encoding/multi_head_attention/attention_output/biasencoding/gamma_1encoding/beta_1encoding/kernel_1encoding/bias_1encoding/kernelencoding/biasencoding/gammaencoding/betaz_mean/kernelz_mean/biasz_log_var/kernelz_log_var/bias*"
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������*:���������*:���������**8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference_signature_wrapper_16292

NoOpNoOp
�u
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�t
value�tB�t B�t
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
		variables

trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
token_embedding
position_embedding*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_self_attention_layer
 _self_attention_layernorm
!_self_attention_dropout
"_feedforward_layernorm
##_feedforward_intermediate_dense
$_feedforward_output_dense
%_feedforward_dropout*

&	keras_api* 

'	keras_api* 
�
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses

.kernel
/bias*
�
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses

6kernel
7bias*
�
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses* 
�
>0
?1
@2
A3
B4
C5
D6
E7
F8
G9
H10
I11
J12
K13
L14
M15
N16
O17
.18
/19
620
721*
�
>0
?1
@2
A3
B4
C5
D6
E7
F8
G9
H10
I11
J12
K13
L14
M15
N16
O17
.18
/19
620
721*
* 
�
Pnon_trainable_variables

Qlayers
Rmetrics
Slayer_regularization_losses
Tlayer_metrics
		variables

trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Utrace_0
Vtrace_1* 

Wtrace_0
Xtrace_1* 
* 

Yserving_default* 

>0
?1*

>0
?1*
* 
�
Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

_trace_0* 

`trace_0* 
�
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses
>
embeddings*
�
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses
?
embeddings
?position_embeddings*
z
@0
A1
B2
C3
D4
E5
F6
G7
H8
I9
J10
K11
L12
M13
N14
O15*
z
@0
A1
B2
C3
D4
E5
F6
G7
H8
I9
J10
K11
L12
M13
N14
O15*
* 
�
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

rtrace_0
strace_1* 

ttrace_0
utrace_1* 
�
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses
|_query_dense
}
_key_dense
~_value_dense
_softmax
�_dropout_layer
�_output_dense*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
	Hgamma
Ibeta*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
	Jgamma
Kbeta*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

Lkernel
Mbias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

Nkernel
Obias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 
* 
* 

.0
/1*

.0
/1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
]W
VARIABLE_VALUEz_mean/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEz_mean/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

60
71*

60
71*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEz_log_var/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEz_log_var/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
\V
VARIABLE_VALUEinput_embedding/embeddings_1&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEinput_embedding/embeddings&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUE*encoding/multi_head_attention/query/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUE(encoding/multi_head_attention/query/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUE(encoding/multi_head_attention/key/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE&encoding/multi_head_attention/key/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUE*encoding/multi_head_attention/value/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUE(encoding/multi_head_attention/value/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE5encoding/multi_head_attention/attention_output/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE3encoding/multi_head_attention/attention_output/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEencoding/gamma_1'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEencoding/beta_1'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEencoding/gamma'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEencoding/beta'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEencoding/kernel_1'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEencoding/bias_1'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEencoding/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEencoding/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
* 
<
0
1
2
3
4
5
6
7*
* 
* 
* 
* 
* 
* 
* 
* 
* 

0
1*
* 
* 
* 
* 
* 

>0*

>0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses*
* 
* 

?0*

?0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses*
* 
* 
* 
5
0
 1
!2
"3
#4
$5
%6*
* 
* 
* 
* 
* 
* 
* 
<
@0
A1
B2
C3
D4
E5
F6
G7*
<
@0
A1
B2
C3
D4
E5
F6
G7*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses*
* 
* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

@kernel
Abias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

Bkernel
Cbias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

Dkernel
Ebias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

Fkernel
Gbias*

H0
I1*

H0
I1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 

J0
K1*

J0
K1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 

L0
M1*

L0
M1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 

N0
O1*

N0
O1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
0
|0
}1
~2
3
�4
�5*
* 
* 
* 

@0
A1*

@0
A1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 

B0
C1*

B0
C1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 

D0
E1*

D0
E1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 

F0
G1*

F0
G1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamez_mean/kernelz_mean/biasz_log_var/kernelz_log_var/biasinput_embedding/embeddings_1input_embedding/embeddings*encoding/multi_head_attention/query/kernel(encoding/multi_head_attention/query/bias(encoding/multi_head_attention/key/kernel&encoding/multi_head_attention/key/bias*encoding/multi_head_attention/value/kernel(encoding/multi_head_attention/value/bias5encoding/multi_head_attention/attention_output/kernel3encoding/multi_head_attention/attention_output/biasencoding/gamma_1encoding/beta_1encoding/gammaencoding/betaencoding/kernel_1encoding/bias_1encoding/kernelencoding/biasConst*#
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *'
f"R 
__inference__traced_save_16875
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamez_mean/kernelz_mean/biasz_log_var/kernelz_log_var/biasinput_embedding/embeddings_1input_embedding/embeddings*encoding/multi_head_attention/query/kernel(encoding/multi_head_attention/query/bias(encoding/multi_head_attention/key/kernel&encoding/multi_head_attention/key/bias*encoding/multi_head_attention/value/kernel(encoding/multi_head_attention/value/bias5encoding/multi_head_attention/attention_output/kernel3encoding/multi_head_attention/attention_output/biasencoding/gamma_1encoding/beta_1encoding/gammaencoding/betaencoding/kernel_1encoding/bias_1encoding/kernelencoding/bias*"
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__traced_restore_16950��
�
�
(__inference_encoding_layer_call_fn_16364

inputs
unknown:� 
	unknown_0:  
	unknown_1:� 
	unknown_2:  
	unknown_3:� 
	unknown_4:  
	unknown_5: �
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:	�*

unknown_10:*

unknown_11:	*�

unknown_12:	�

unknown_13:	�

unknown_14:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������U�*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_encoding_layer_call_and_return_conditional_losses_15737t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������U�<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:���������U�: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name16360:%!

_user_specified_name16358:%!

_user_specified_name16356:%!

_user_specified_name16354:%!

_user_specified_name16352:%!

_user_specified_name16350:%
!

_user_specified_name16348:%	!

_user_specified_name16346:%!

_user_specified_name16344:%!

_user_specified_name16342:%!

_user_specified_name16340:%!

_user_specified_name16338:%!

_user_specified_name16336:%!

_user_specified_name16334:%!

_user_specified_name16332:%!

_user_specified_name16330:T P
,
_output_shapes
:���������U�
 
_user_specified_nameinputs
�
i
<__inference_z_layer_call_and_return_conditional_losses_15828

inputs
inputs_1
identity�I
ShapeShapeinputs*
T0*
_output_shapes
::��W
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*'
_output_shapes
:���������**
dtype0*
seed2��;*
seed���)�
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:���������*|
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:���������*J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?V
mulMulmul/x:output:0inputs_1*
T0*'
_output_shapes
:���������*E
ExpExpmul:z:0*
T0*'
_output_shapes
:���������*Z
mul_1MulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:���������*Q
addAddV2inputs	mul_1:z:0*
T0*'
_output_shapes
:���������*L
mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �BY
mul_2Muladd:z:0mul_2/y:output:0*
T0*'
_output_shapes
:���������*K
RoundRound	mul_2:z:0*
T0*'
_output_shapes
:���������*X
CastCast	Round:y:0*

DstT0*

SrcT0*'
_output_shapes
:���������*P
IdentityIdentityCast:y:0*
T0*'
_output_shapes
:���������*"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������*:���������*:OK
'
_output_shapes
:���������*
 
_user_specified_nameinputs:O K
'
_output_shapes
:���������*
 
_user_specified_nameinputs
�
�
/__inference_input_embedding_layer_call_fn_16301

inputs
unknown:
�'�
	unknown_0:	U�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������U�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_input_embedding_layer_call_and_return_conditional_losses_15603t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������U�<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������U: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name16297:%!

_user_specified_name16295:O K
'
_output_shapes
:���������U
 
_user_specified_nameinputs
Ŵ
�
__inference__traced_save_16875
file_prefix7
$read_disablecopyonread_z_mean_kernel:	�*2
$read_1_disablecopyonread_z_mean_bias:*<
)read_2_disablecopyonread_z_log_var_kernel:	�*5
'read_3_disablecopyonread_z_log_var_bias:*I
5read_4_disablecopyonread_input_embedding_embeddings_1:
�'�F
3read_5_disablecopyonread_input_embedding_embeddings:	U�Z
Cread_6_disablecopyonread_encoding_multi_head_attention_query_kernel:� S
Aread_7_disablecopyonread_encoding_multi_head_attention_query_bias: X
Aread_8_disablecopyonread_encoding_multi_head_attention_key_kernel:� Q
?read_9_disablecopyonread_encoding_multi_head_attention_key_bias: [
Dread_10_disablecopyonread_encoding_multi_head_attention_value_kernel:� T
Bread_11_disablecopyonread_encoding_multi_head_attention_value_bias: f
Oread_12_disablecopyonread_encoding_multi_head_attention_attention_output_kernel: �\
Mread_13_disablecopyonread_encoding_multi_head_attention_attention_output_bias:	�9
*read_14_disablecopyonread_encoding_gamma_1:	�8
)read_15_disablecopyonread_encoding_beta_1:	�7
(read_16_disablecopyonread_encoding_gamma:	�6
'read_17_disablecopyonread_encoding_beta:	�>
+read_18_disablecopyonread_encoding_kernel_1:	�*7
)read_19_disablecopyonread_encoding_bias_1:*<
)read_20_disablecopyonread_encoding_kernel:	*�6
'read_21_disablecopyonread_encoding_bias:	�
savev2_const
identity_45��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: v
Read/DisableCopyOnReadDisableCopyOnRead$read_disablecopyonread_z_mean_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp$read_disablecopyonread_z_mean_kernel^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�**
dtype0j
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�*b

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes
:	�*x
Read_1/DisableCopyOnReadDisableCopyOnRead$read_1_disablecopyonread_z_mean_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp$read_1_disablecopyonread_z_mean_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:**
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:*_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:*}
Read_2/DisableCopyOnReadDisableCopyOnRead)read_2_disablecopyonread_z_log_var_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp)read_2_disablecopyonread_z_log_var_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�**
dtype0n

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�*d

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
:	�*{
Read_3/DisableCopyOnReadDisableCopyOnRead'read_3_disablecopyonread_z_log_var_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp'read_3_disablecopyonread_z_log_var_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:**
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:*_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:*�
Read_4/DisableCopyOnReadDisableCopyOnRead5read_4_disablecopyonread_input_embedding_embeddings_1"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp5read_4_disablecopyonread_input_embedding_embeddings_1^Read_4/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
�'�*
dtype0o

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
�'�e

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0* 
_output_shapes
:
�'��
Read_5/DisableCopyOnReadDisableCopyOnRead3read_5_disablecopyonread_input_embedding_embeddings"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp3read_5_disablecopyonread_input_embedding_embeddings^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	U�*
dtype0o
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	U�f
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:	U��
Read_6/DisableCopyOnReadDisableCopyOnReadCread_6_disablecopyonread_encoding_multi_head_attention_query_kernel"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOpCread_6_disablecopyonread_encoding_multi_head_attention_query_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*#
_output_shapes
:� *
dtype0s
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*#
_output_shapes
:� j
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*#
_output_shapes
:� �
Read_7/DisableCopyOnReadDisableCopyOnReadAread_7_disablecopyonread_encoding_multi_head_attention_query_bias"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOpAread_7_disablecopyonread_encoding_multi_head_attention_query_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0n
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: e
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes

: �
Read_8/DisableCopyOnReadDisableCopyOnReadAread_8_disablecopyonread_encoding_multi_head_attention_key_kernel"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOpAread_8_disablecopyonread_encoding_multi_head_attention_key_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*#
_output_shapes
:� *
dtype0s
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*#
_output_shapes
:� j
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*#
_output_shapes
:� �
Read_9/DisableCopyOnReadDisableCopyOnRead?read_9_disablecopyonread_encoding_multi_head_attention_key_bias"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp?read_9_disablecopyonread_encoding_multi_head_attention_key_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0n
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: e
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes

: �
Read_10/DisableCopyOnReadDisableCopyOnReadDread_10_disablecopyonread_encoding_multi_head_attention_value_kernel"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOpDread_10_disablecopyonread_encoding_multi_head_attention_value_kernel^Read_10/DisableCopyOnRead"/device:CPU:0*#
_output_shapes
:� *
dtype0t
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*#
_output_shapes
:� j
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*#
_output_shapes
:� �
Read_11/DisableCopyOnReadDisableCopyOnReadBread_11_disablecopyonread_encoding_multi_head_attention_value_bias"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOpBread_11_disablecopyonread_encoding_multi_head_attention_value_bias^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0o
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: e
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes

: �
Read_12/DisableCopyOnReadDisableCopyOnReadOread_12_disablecopyonread_encoding_multi_head_attention_attention_output_kernel"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOpOread_12_disablecopyonread_encoding_multi_head_attention_attention_output_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*#
_output_shapes
: �*
dtype0t
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*#
_output_shapes
: �j
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*#
_output_shapes
: ��
Read_13/DisableCopyOnReadDisableCopyOnReadMread_13_disablecopyonread_encoding_multi_head_attention_attention_output_bias"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOpMread_13_disablecopyonread_encoding_multi_head_attention_attention_output_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes	
:�
Read_14/DisableCopyOnReadDisableCopyOnRead*read_14_disablecopyonread_encoding_gamma_1"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp*read_14_disablecopyonread_encoding_gamma_1^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes	
:�~
Read_15/DisableCopyOnReadDisableCopyOnRead)read_15_disablecopyonread_encoding_beta_1"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp)read_15_disablecopyonread_encoding_beta_1^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes	
:�}
Read_16/DisableCopyOnReadDisableCopyOnRead(read_16_disablecopyonread_encoding_gamma"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp(read_16_disablecopyonread_encoding_gamma^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes	
:�|
Read_17/DisableCopyOnReadDisableCopyOnRead'read_17_disablecopyonread_encoding_beta"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp'read_17_disablecopyonread_encoding_beta^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_18/DisableCopyOnReadDisableCopyOnRead+read_18_disablecopyonread_encoding_kernel_1"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp+read_18_disablecopyonread_encoding_kernel_1^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�**
dtype0p
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�*f
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
:	�*~
Read_19/DisableCopyOnReadDisableCopyOnRead)read_19_disablecopyonread_encoding_bias_1"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp)read_19_disablecopyonread_encoding_bias_1^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:**
dtype0k
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:*a
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
:*~
Read_20/DisableCopyOnReadDisableCopyOnRead)read_20_disablecopyonread_encoding_kernel"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp)read_20_disablecopyonread_encoding_kernel^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*�*
dtype0p
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	*�f
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes
:	*�|
Read_21/DisableCopyOnReadDisableCopyOnRead'read_21_disablecopyonread_encoding_bias"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp'read_21_disablecopyonread_encoding_bias^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *%
dtypes
2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_44Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_45IdentityIdentity_44:output:0^NoOp*
T0*
_output_shapes
: �	
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_45Identity_45:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0: : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:=9

_output_shapes
: 

_user_specified_nameConst:-)
'
_user_specified_nameencoding/bias:/+
)
_user_specified_nameencoding/kernel:/+
)
_user_specified_nameencoding/bias_1:1-
+
_user_specified_nameencoding/kernel_1:-)
'
_user_specified_nameencoding/beta:.*
(
_user_specified_nameencoding/gamma:/+
)
_user_specified_nameencoding/beta_1:0,
*
_user_specified_nameencoding/gamma_1:SO
M
_user_specified_name53encoding/multi_head_attention/attention_output/bias:UQ
O
_user_specified_name75encoding/multi_head_attention/attention_output/kernel:HD
B
_user_specified_name*(encoding/multi_head_attention/value/bias:JF
D
_user_specified_name,*encoding/multi_head_attention/value/kernel:F
B
@
_user_specified_name(&encoding/multi_head_attention/key/bias:H	D
B
_user_specified_name*(encoding/multi_head_attention/key/kernel:HD
B
_user_specified_name*(encoding/multi_head_attention/query/bias:JF
D
_user_specified_name,*encoding/multi_head_attention/query/kernel::6
4
_user_specified_nameinput_embedding/embeddings:<8
6
_user_specified_nameinput_embedding/embeddings_1:.*
(
_user_specified_namez_log_var/bias:0,
*
_user_specified_namez_log_var/kernel:+'
%
_user_specified_namez_mean/bias:-)
'
_user_specified_namez_mean/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�k
�
!__inference__traced_restore_16950
file_prefix1
assignvariableop_z_mean_kernel:	�*,
assignvariableop_1_z_mean_bias:*6
#assignvariableop_2_z_log_var_kernel:	�*/
!assignvariableop_3_z_log_var_bias:*C
/assignvariableop_4_input_embedding_embeddings_1:
�'�@
-assignvariableop_5_input_embedding_embeddings:	U�T
=assignvariableop_6_encoding_multi_head_attention_query_kernel:� M
;assignvariableop_7_encoding_multi_head_attention_query_bias: R
;assignvariableop_8_encoding_multi_head_attention_key_kernel:� K
9assignvariableop_9_encoding_multi_head_attention_key_bias: U
>assignvariableop_10_encoding_multi_head_attention_value_kernel:� N
<assignvariableop_11_encoding_multi_head_attention_value_bias: `
Iassignvariableop_12_encoding_multi_head_attention_attention_output_kernel: �V
Gassignvariableop_13_encoding_multi_head_attention_attention_output_bias:	�3
$assignvariableop_14_encoding_gamma_1:	�2
#assignvariableop_15_encoding_beta_1:	�1
"assignvariableop_16_encoding_gamma:	�0
!assignvariableop_17_encoding_beta:	�8
%assignvariableop_18_encoding_kernel_1:	�*1
#assignvariableop_19_encoding_bias_1:*6
#assignvariableop_20_encoding_kernel:	*�0
!assignvariableop_21_encoding_bias:	�
identity_23��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*p
_output_shapes^
\:::::::::::::::::::::::*%
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_z_mean_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_z_mean_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp#assignvariableop_2_z_log_var_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp!assignvariableop_3_z_log_var_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp/assignvariableop_4_input_embedding_embeddings_1Identity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp-assignvariableop_5_input_embedding_embeddingsIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp=assignvariableop_6_encoding_multi_head_attention_query_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp;assignvariableop_7_encoding_multi_head_attention_query_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp;assignvariableop_8_encoding_multi_head_attention_key_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp9assignvariableop_9_encoding_multi_head_attention_key_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp>assignvariableop_10_encoding_multi_head_attention_value_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp<assignvariableop_11_encoding_multi_head_attention_value_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpIassignvariableop_12_encoding_multi_head_attention_attention_output_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpGassignvariableop_13_encoding_multi_head_attention_attention_output_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp$assignvariableop_14_encoding_gamma_1Identity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp#assignvariableop_15_encoding_beta_1Identity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp"assignvariableop_16_encoding_gammaIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp!assignvariableop_17_encoding_betaIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp%assignvariableop_18_encoding_kernel_1Identity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp#assignvariableop_19_encoding_bias_1Identity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp#assignvariableop_20_encoding_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp!assignvariableop_21_encoding_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_22Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_23IdentityIdentity_22:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_23Identity_23:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : : : : : : : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:-)
'
_user_specified_nameencoding/bias:/+
)
_user_specified_nameencoding/kernel:/+
)
_user_specified_nameencoding/bias_1:1-
+
_user_specified_nameencoding/kernel_1:-)
'
_user_specified_nameencoding/beta:.*
(
_user_specified_nameencoding/gamma:/+
)
_user_specified_nameencoding/beta_1:0,
*
_user_specified_nameencoding/gamma_1:SO
M
_user_specified_name53encoding/multi_head_attention/attention_output/bias:UQ
O
_user_specified_name75encoding/multi_head_attention/attention_output/kernel:HD
B
_user_specified_name*(encoding/multi_head_attention/value/bias:JF
D
_user_specified_name,*encoding/multi_head_attention/value/kernel:F
B
@
_user_specified_name(&encoding/multi_head_attention/key/bias:H	D
B
_user_specified_name*(encoding/multi_head_attention/key/kernel:HD
B
_user_specified_name*(encoding/multi_head_attention/query/bias:JF
D
_user_specified_name,*encoding/multi_head_attention/query/kernel::6
4
_user_specified_nameinput_embedding/embeddings:<8
6
_user_specified_nameinput_embedding/embeddings_1:.*
(
_user_specified_namez_log_var/bias:0,
*
_user_specified_namez_log_var/kernel:+'
%
_user_specified_namez_mean/bias:-)
'
_user_specified_namez_mean/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
��
�
C__inference_encoding_layer_call_and_return_conditional_losses_16655

inputsW
@multi_head_attention_query_einsum_einsum_readvariableop_resource:� H
6multi_head_attention_query_add_readvariableop_resource: U
>multi_head_attention_key_einsum_einsum_readvariableop_resource:� F
4multi_head_attention_key_add_readvariableop_resource: W
@multi_head_attention_value_einsum_einsum_readvariableop_resource:� H
6multi_head_attention_value_add_readvariableop_resource: b
Kmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource: �P
Amulti_head_attention_attention_output_add_readvariableop_resource:	�H
9layer_normalization_batchnorm_mul_readvariableop_resource:	�D
5layer_normalization_batchnorm_readvariableop_resource:	�:
'dense_tensordot_readvariableop_resource:	�*3
%dense_biasadd_readvariableop_resource:*<
)dense_1_tensordot_readvariableop_resource:	*�6
'dense_1_biasadd_readvariableop_resource:	�J
;layer_normalization_1_batchnorm_mul_readvariableop_resource:	�F
7layer_normalization_1_batchnorm_readvariableop_resource:	�
identity��dense/BiasAdd/ReadVariableOp�dense/Tensordot/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp� dense_1/Tensordot/ReadVariableOp�,layer_normalization/batchnorm/ReadVariableOp�0layer_normalization/batchnorm/mul/ReadVariableOp�.layer_normalization_1/batchnorm/ReadVariableOp�2layer_normalization_1/batchnorm/mul/ReadVariableOp�8multi_head_attention/attention_output/add/ReadVariableOp�Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp�+multi_head_attention/key/add/ReadVariableOp�5multi_head_attention/key/einsum/Einsum/ReadVariableOp�-multi_head_attention/query/add/ReadVariableOp�7multi_head_attention/query/einsum/Einsum/ReadVariableOp�-multi_head_attention/value/add/ReadVariableOp�7multi_head_attention/value/einsum/Einsum/ReadVariableOp�
7multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_query_einsum_einsum_readvariableop_resource*#
_output_shapes
:� *
dtype0�
(multi_head_attention/query/einsum/EinsumEinsuminputs?multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������U *
equationabc,cde->abde�
-multi_head_attention/query/add/ReadVariableOpReadVariableOp6multi_head_attention_query_add_readvariableop_resource*
_output_shapes

: *
dtype0�
multi_head_attention/query/addAddV21multi_head_attention/query/einsum/Einsum:output:05multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������U �
5multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp>multi_head_attention_key_einsum_einsum_readvariableop_resource*#
_output_shapes
:� *
dtype0�
&multi_head_attention/key/einsum/EinsumEinsuminputs=multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������U *
equationabc,cde->abde�
+multi_head_attention/key/add/ReadVariableOpReadVariableOp4multi_head_attention_key_add_readvariableop_resource*
_output_shapes

: *
dtype0�
multi_head_attention/key/addAddV2/multi_head_attention/key/einsum/Einsum:output:03multi_head_attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������U �
7multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_value_einsum_einsum_readvariableop_resource*#
_output_shapes
:� *
dtype0�
(multi_head_attention/value/einsum/EinsumEinsuminputs?multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������U *
equationabc,cde->abde�
-multi_head_attention/value/add/ReadVariableOpReadVariableOp6multi_head_attention_value_add_readvariableop_resource*
_output_shapes

: *
dtype0�
multi_head_attention/value/addAddV21multi_head_attention/value/einsum/Einsum:output:05multi_head_attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������U _
multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *�5>�
multi_head_attention/MulMul"multi_head_attention/query/add:z:0#multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:���������U �
"multi_head_attention/einsum/EinsumEinsum multi_head_attention/key/add:z:0multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:���������UU*
equationaecd,abcd->acbe�
$multi_head_attention/softmax/SoftmaxSoftmax+multi_head_attention/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������UU�
%multi_head_attention/dropout/IdentityIdentity.multi_head_attention/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:���������UU�
$multi_head_attention/einsum_1/EinsumEinsum.multi_head_attention/dropout/Identity:output:0"multi_head_attention/value/add:z:0*
N*
T0*/
_output_shapes
:���������U *
equationacbe,aecd->abcd�
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpKmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource*#
_output_shapes
: �*
dtype0�
3multi_head_attention/attention_output/einsum/EinsumEinsum-multi_head_attention/einsum_1/Einsum:output:0Jmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:���������U�*
equationabcd,cde->abe�
8multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpAmulti_head_attention_attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
)multi_head_attention/attention_output/addAddV2<multi_head_attention/attention_output/einsum/Einsum:output:0@multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������U��
dropout_1/IdentityIdentity-multi_head_attention/attention_output/add:z:0*
T0*,
_output_shapes
:���������U�h
addAddV2dropout_1/Identity:output:0inputs*
T0*,
_output_shapes
:���������U�|
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
 layer_normalization/moments/meanMeanadd:z:0;layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������U*
	keep_dims(�
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:���������U�
-layer_normalization/moments/SquaredDifferenceSquaredDifferenceadd:z:01layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:���������U��
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������U*
	keep_dims(h
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������U�
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:���������U�
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������U��
#layer_normalization/batchnorm/mul_1Muladd:z:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������U��
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������U��
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:���������U��
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������U��
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes
:	�**
dtype0^
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:e
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       z
dense/Tensordot/ShapeShape'layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
::��_
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:_
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: a
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ]
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
dense/Tensordot/transpose	Transpose'layer_normalization/batchnorm/add_1:z:0dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������U��
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������*a
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:*_
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������U*~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:**
dtype0�
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������U*`

dense/ReluReludense/BiasAdd:output:0*
T0*+
_output_shapes
:���������U*�
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes
:	*�*
dtype0`
dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       m
dense_1/Tensordot/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
::��a
dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_1/Tensordot/GatherV2GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/free:output:0(dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_1/Tensordot/GatherV2_1GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/axes:output:0*dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
dense_1/Tensordot/ProdProd#dense_1/Tensordot/GatherV2:output:0 dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense_1/Tensordot/Prod_1Prod%dense_1/Tensordot/GatherV2_1:output:0"dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_1/Tensordot/concatConcatV2dense_1/Tensordot/free:output:0dense_1/Tensordot/axes:output:0&dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
dense_1/Tensordot/stackPackdense_1/Tensordot/Prod:output:0!dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
dense_1/Tensordot/transpose	Transposedense/Relu:activations:0!dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������U*�
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�a
dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_1/Tensordot/concat_1ConcatV2#dense_1/Tensordot/GatherV2:output:0"dense_1/Tensordot/Const_2:output:0(dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0#dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������U��
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������U�o
dropout_2/IdentityIdentitydense_1/BiasAdd:output:0*
T0*,
_output_shapes
:���������U��
add_1AddV2dropout_2/Identity:output:0'layer_normalization/batchnorm/add_1:z:0*
T0*,
_output_shapes
:���������U�~
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
"layer_normalization_1/moments/meanMean	add_1:z:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������U*
	keep_dims(�
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:���������U�
/layer_normalization_1/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:���������U��
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������U*
	keep_dims(j
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������U�
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:���������U�
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������U��
%layer_normalization_1/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������U��
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������U��
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:���������U��
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������U�}
IdentityIdentity)layer_normalization_1/batchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:���������U��
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp9^multi_head_attention/attention_output/add/ReadVariableOpC^multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp,^multi_head_attention/key/add/ReadVariableOp6^multi_head_attention/key/einsum/Einsum/ReadVariableOp.^multi_head_attention/query/add/ReadVariableOp8^multi_head_attention/query/einsum/Einsum/ReadVariableOp.^multi_head_attention/value/add/ReadVariableOp8^multi_head_attention/value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:���������U�: : : : : : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp2\
,layer_normalization/batchnorm/ReadVariableOp,layer_normalization/batchnorm/ReadVariableOp2d
0layer_normalization/batchnorm/mul/ReadVariableOp0layer_normalization/batchnorm/mul/ReadVariableOp2`
.layer_normalization_1/batchnorm/ReadVariableOp.layer_normalization_1/batchnorm/ReadVariableOp2h
2layer_normalization_1/batchnorm/mul/ReadVariableOp2layer_normalization_1/batchnorm/mul/ReadVariableOp2t
8multi_head_attention/attention_output/add/ReadVariableOp8multi_head_attention/attention_output/add/ReadVariableOp2�
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpBmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp2Z
+multi_head_attention/key/add/ReadVariableOp+multi_head_attention/key/add/ReadVariableOp2n
5multi_head_attention/key/einsum/Einsum/ReadVariableOp5multi_head_attention/key/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/query/add/ReadVariableOp-multi_head_attention/query/add/ReadVariableOp2r
7multi_head_attention/query/einsum/Einsum/ReadVariableOp7multi_head_attention/query/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/value/add/ReadVariableOp-multi_head_attention/value/add/ReadVariableOp2r
7multi_head_attention/value/einsum/Einsum/ReadVariableOp7multi_head_attention/value/einsum/Einsum/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:T P
,
_output_shapes
:���������U�
 
_user_specified_nameinputs
�
�
J__inference_input_embedding_layer_call_and_return_conditional_losses_15603

inputs;
'token_embedding8_embedding_lookup_15580:
�'�D
1position_embedding8_slice_readvariableop_resource:	U�
identity��(position_embedding8/Slice/ReadVariableOp�!token_embedding8/embedding_lookup�
!token_embedding8/embedding_lookupResourceGather'token_embedding8_embedding_lookup_15580inputs*
Tindices0*:
_class0
.,loc:@token_embedding8/embedding_lookup/15580*,
_output_shapes
:���������U�*
dtype0�
*token_embedding8/embedding_lookup/IdentityIdentity*token_embedding8/embedding_lookup:output:0*
T0*,
_output_shapes
:���������U�]
token_embedding8/NotEqual/yConst*
_output_shapes
: *
dtype0*
value	B : �
token_embedding8/NotEqualNotEqualinputs$token_embedding8/NotEqual/y:output:0*
T0*'
_output_shapes
:���������U�
position_embedding8/ShapeShape3token_embedding8/embedding_lookup/Identity:output:0*
T0*
_output_shapes
::��q
'position_embedding8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)position_embedding8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)position_embedding8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
!position_embedding8/strided_sliceStridedSlice"position_embedding8/Shape:output:00position_embedding8/strided_slice/stack:output:02position_embedding8/strided_slice/stack_1:output:02position_embedding8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
(position_embedding8/Slice/ReadVariableOpReadVariableOp1position_embedding8_slice_readvariableop_resource*
_output_shapes
:	U�*
dtype0p
position_embedding8/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB"        o
position_embedding8/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB"U   �   �
position_embedding8/SliceSlice0position_embedding8/Slice/ReadVariableOp:value:0(position_embedding8/Slice/begin:output:0'position_embedding8/Slice/size:output:0*
Index0*
T0*
_output_shapes
:	U�^
position_embedding8/packed/1Const*
_output_shapes
: *
dtype0*
value	B :U_
position_embedding8/packed/2Const*
_output_shapes
: *
dtype0*
value
B :��
position_embedding8/packedPack*position_embedding8/strided_slice:output:0%position_embedding8/packed/1:output:0%position_embedding8/packed/2:output:0*
N*
T0*
_output_shapes
:Z
position_embedding8/RankConst*
_output_shapes
: *
dtype0*
value	B :�
position_embedding8/BroadcastToBroadcastTo"position_embedding8/Slice:output:0#position_embedding8/packed:output:0*
T0*,
_output_shapes
:���������U��
addAddV23token_embedding8/embedding_lookup/Identity:output:0(position_embedding8/BroadcastTo:output:0*
T0*,
_output_shapes
:���������U�[
IdentityIdentityadd:z:0^NoOp*
T0*,
_output_shapes
:���������U�q
NoOpNoOp)^position_embedding8/Slice/ReadVariableOp"^token_embedding8/embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������U: : 2T
(position_embedding8/Slice/ReadVariableOp(position_embedding8/Slice/ReadVariableOp2F
!token_embedding8/embedding_lookup!token_embedding8/embedding_lookup:($
"
_user_specified_name
resource:%!

_user_specified_name15580:O K
'
_output_shapes
:���������U
 
_user_specified_nameinputs
�	
�
A__inference_z_mean_layer_call_and_return_conditional_losses_16674

inputs1
matmul_readvariableop_resource:	�*-
biasadd_readvariableop_resource:*
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�**
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������*r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:**
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������*_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������*S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
&__inference_z_mean_layer_call_fn_16664

inputs
unknown:	�*
	unknown_0:*
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_z_mean_layer_call_and_return_conditional_losses_15788o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������*<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name16660:%!

_user_specified_name16658:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
'__inference_encoder_layer_call_fn_16131
input_word_ids
unknown:
�'�
	unknown_0:	U� 
	unknown_1:� 
	unknown_2:  
	unknown_3:� 
	unknown_4:  
	unknown_5:� 
	unknown_6:  
	unknown_7: �
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:	�*

unknown_12:*

unknown_13:	*�

unknown_14:	�

unknown_15:	�

unknown_16:	�

unknown_17:	�*

unknown_18:*

unknown_19:	�*

unknown_20:*
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_word_idsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������*:���������*:���������**8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_16025o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������*q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������*q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:���������*<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������U: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name16123:%!

_user_specified_name16121:%!

_user_specified_name16119:%!

_user_specified_name16117:%!

_user_specified_name16115:%!

_user_specified_name16113:%!

_user_specified_name16111:%!

_user_specified_name16109:%!

_user_specified_name16107:%!

_user_specified_name16105:%!

_user_specified_name16103:%!

_user_specified_name16101:%
!

_user_specified_name16099:%	!

_user_specified_name16097:%!

_user_specified_name16095:%!

_user_specified_name16093:%!

_user_specified_name16091:%!

_user_specified_name16089:%!

_user_specified_name16087:%!

_user_specified_name16085:%!

_user_specified_name16083:%!

_user_specified_name16081:W S
'
_output_shapes
:���������U
(
_user_specified_nameinput_word_ids
�	
�
D__inference_z_log_var_layer_call_and_return_conditional_losses_15803

inputs1
matmul_readvariableop_resource:	�*-
biasadd_readvariableop_resource:*
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�**
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������*r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:**
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������*_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������*S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
(__inference_encoding_layer_call_fn_16401

inputs
unknown:� 
	unknown_0:  
	unknown_1:� 
	unknown_2:  
	unknown_3:� 
	unknown_4:  
	unknown_5: �
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:	�*

unknown_10:*

unknown_11:	*�

unknown_12:	�

unknown_13:	�

unknown_14:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������U�*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_encoding_layer_call_and_return_conditional_losses_15969t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������U�<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:���������U�: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name16397:%!

_user_specified_name16395:%!

_user_specified_name16393:%!

_user_specified_name16391:%!

_user_specified_name16389:%!

_user_specified_name16387:%
!

_user_specified_name16385:%	!

_user_specified_name16383:%!

_user_specified_name16381:%!

_user_specified_name16379:%!

_user_specified_name16377:%!

_user_specified_name16375:%!

_user_specified_name16373:%!

_user_specified_name16371:%!

_user_specified_name16369:%!

_user_specified_name16367:T P
,
_output_shapes
:���������U�
 
_user_specified_nameinputs
��
�
C__inference_encoding_layer_call_and_return_conditional_losses_15737

inputsW
@multi_head_attention_query_einsum_einsum_readvariableop_resource:� H
6multi_head_attention_query_add_readvariableop_resource: U
>multi_head_attention_key_einsum_einsum_readvariableop_resource:� F
4multi_head_attention_key_add_readvariableop_resource: W
@multi_head_attention_value_einsum_einsum_readvariableop_resource:� H
6multi_head_attention_value_add_readvariableop_resource: b
Kmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource: �P
Amulti_head_attention_attention_output_add_readvariableop_resource:	�H
9layer_normalization_batchnorm_mul_readvariableop_resource:	�D
5layer_normalization_batchnorm_readvariableop_resource:	�:
'dense_tensordot_readvariableop_resource:	�*3
%dense_biasadd_readvariableop_resource:*<
)dense_1_tensordot_readvariableop_resource:	*�6
'dense_1_biasadd_readvariableop_resource:	�J
;layer_normalization_1_batchnorm_mul_readvariableop_resource:	�F
7layer_normalization_1_batchnorm_readvariableop_resource:	�
identity��dense/BiasAdd/ReadVariableOp�dense/Tensordot/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp� dense_1/Tensordot/ReadVariableOp�,layer_normalization/batchnorm/ReadVariableOp�0layer_normalization/batchnorm/mul/ReadVariableOp�.layer_normalization_1/batchnorm/ReadVariableOp�2layer_normalization_1/batchnorm/mul/ReadVariableOp�8multi_head_attention/attention_output/add/ReadVariableOp�Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp�+multi_head_attention/key/add/ReadVariableOp�5multi_head_attention/key/einsum/Einsum/ReadVariableOp�-multi_head_attention/query/add/ReadVariableOp�7multi_head_attention/query/einsum/Einsum/ReadVariableOp�-multi_head_attention/value/add/ReadVariableOp�7multi_head_attention/value/einsum/Einsum/ReadVariableOp�
7multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_query_einsum_einsum_readvariableop_resource*#
_output_shapes
:� *
dtype0�
(multi_head_attention/query/einsum/EinsumEinsuminputs?multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������U *
equationabc,cde->abde�
-multi_head_attention/query/add/ReadVariableOpReadVariableOp6multi_head_attention_query_add_readvariableop_resource*
_output_shapes

: *
dtype0�
multi_head_attention/query/addAddV21multi_head_attention/query/einsum/Einsum:output:05multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������U �
5multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp>multi_head_attention_key_einsum_einsum_readvariableop_resource*#
_output_shapes
:� *
dtype0�
&multi_head_attention/key/einsum/EinsumEinsuminputs=multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������U *
equationabc,cde->abde�
+multi_head_attention/key/add/ReadVariableOpReadVariableOp4multi_head_attention_key_add_readvariableop_resource*
_output_shapes

: *
dtype0�
multi_head_attention/key/addAddV2/multi_head_attention/key/einsum/Einsum:output:03multi_head_attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������U �
7multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_value_einsum_einsum_readvariableop_resource*#
_output_shapes
:� *
dtype0�
(multi_head_attention/value/einsum/EinsumEinsuminputs?multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������U *
equationabc,cde->abde�
-multi_head_attention/value/add/ReadVariableOpReadVariableOp6multi_head_attention_value_add_readvariableop_resource*
_output_shapes

: *
dtype0�
multi_head_attention/value/addAddV21multi_head_attention/value/einsum/Einsum:output:05multi_head_attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������U _
multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *�5>�
multi_head_attention/MulMul"multi_head_attention/query/add:z:0#multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:���������U �
"multi_head_attention/einsum/EinsumEinsum multi_head_attention/key/add:z:0multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:���������UU*
equationaecd,abcd->acbe�
$multi_head_attention/softmax/SoftmaxSoftmax+multi_head_attention/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������UU�
%multi_head_attention/dropout/IdentityIdentity.multi_head_attention/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:���������UU�
$multi_head_attention/einsum_1/EinsumEinsum.multi_head_attention/dropout/Identity:output:0"multi_head_attention/value/add:z:0*
N*
T0*/
_output_shapes
:���������U *
equationacbe,aecd->abcd�
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpKmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource*#
_output_shapes
: �*
dtype0�
3multi_head_attention/attention_output/einsum/EinsumEinsum-multi_head_attention/einsum_1/Einsum:output:0Jmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:���������U�*
equationabcd,cde->abe�
8multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpAmulti_head_attention_attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
)multi_head_attention/attention_output/addAddV2<multi_head_attention/attention_output/einsum/Einsum:output:0@multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������U��
dropout_1/IdentityIdentity-multi_head_attention/attention_output/add:z:0*
T0*,
_output_shapes
:���������U�h
addAddV2dropout_1/Identity:output:0inputs*
T0*,
_output_shapes
:���������U�|
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
 layer_normalization/moments/meanMeanadd:z:0;layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������U*
	keep_dims(�
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:���������U�
-layer_normalization/moments/SquaredDifferenceSquaredDifferenceadd:z:01layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:���������U��
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������U*
	keep_dims(h
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������U�
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:���������U�
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������U��
#layer_normalization/batchnorm/mul_1Muladd:z:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������U��
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������U��
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:���������U��
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������U��
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes
:	�**
dtype0^
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:e
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       z
dense/Tensordot/ShapeShape'layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
::��_
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:_
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: a
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ]
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
dense/Tensordot/transpose	Transpose'layer_normalization/batchnorm/add_1:z:0dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������U��
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������*a
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:*_
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������U*~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:**
dtype0�
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������U*`

dense/ReluReludense/BiasAdd:output:0*
T0*+
_output_shapes
:���������U*�
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes
:	*�*
dtype0`
dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       m
dense_1/Tensordot/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
::��a
dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_1/Tensordot/GatherV2GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/free:output:0(dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_1/Tensordot/GatherV2_1GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/axes:output:0*dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
dense_1/Tensordot/ProdProd#dense_1/Tensordot/GatherV2:output:0 dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense_1/Tensordot/Prod_1Prod%dense_1/Tensordot/GatherV2_1:output:0"dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_1/Tensordot/concatConcatV2dense_1/Tensordot/free:output:0dense_1/Tensordot/axes:output:0&dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
dense_1/Tensordot/stackPackdense_1/Tensordot/Prod:output:0!dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
dense_1/Tensordot/transpose	Transposedense/Relu:activations:0!dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������U*�
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�a
dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_1/Tensordot/concat_1ConcatV2#dense_1/Tensordot/GatherV2:output:0"dense_1/Tensordot/Const_2:output:0(dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0#dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������U��
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������U�o
dropout_2/IdentityIdentitydense_1/BiasAdd:output:0*
T0*,
_output_shapes
:���������U��
add_1AddV2dropout_2/Identity:output:0'layer_normalization/batchnorm/add_1:z:0*
T0*,
_output_shapes
:���������U�~
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
"layer_normalization_1/moments/meanMean	add_1:z:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������U*
	keep_dims(�
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:���������U�
/layer_normalization_1/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:���������U��
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������U*
	keep_dims(j
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������U�
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:���������U�
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������U��
%layer_normalization_1/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������U��
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������U��
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:���������U��
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������U�}
IdentityIdentity)layer_normalization_1/batchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:���������U��
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp9^multi_head_attention/attention_output/add/ReadVariableOpC^multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp,^multi_head_attention/key/add/ReadVariableOp6^multi_head_attention/key/einsum/Einsum/ReadVariableOp.^multi_head_attention/query/add/ReadVariableOp8^multi_head_attention/query/einsum/Einsum/ReadVariableOp.^multi_head_attention/value/add/ReadVariableOp8^multi_head_attention/value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:���������U�: : : : : : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp2\
,layer_normalization/batchnorm/ReadVariableOp,layer_normalization/batchnorm/ReadVariableOp2d
0layer_normalization/batchnorm/mul/ReadVariableOp0layer_normalization/batchnorm/mul/ReadVariableOp2`
.layer_normalization_1/batchnorm/ReadVariableOp.layer_normalization_1/batchnorm/ReadVariableOp2h
2layer_normalization_1/batchnorm/mul/ReadVariableOp2layer_normalization_1/batchnorm/mul/ReadVariableOp2t
8multi_head_attention/attention_output/add/ReadVariableOp8multi_head_attention/attention_output/add/ReadVariableOp2�
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpBmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp2Z
+multi_head_attention/key/add/ReadVariableOp+multi_head_attention/key/add/ReadVariableOp2n
5multi_head_attention/key/einsum/Einsum/ReadVariableOp5multi_head_attention/key/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/query/add/ReadVariableOp-multi_head_attention/query/add/ReadVariableOp2r
7multi_head_attention/query/einsum/Einsum/ReadVariableOp7multi_head_attention/query/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/value/add/ReadVariableOp-multi_head_attention/value/add/ReadVariableOp2r
7multi_head_attention/value/einsum/Einsum/ReadVariableOp7multi_head_attention/value/einsum/Einsum/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:T P
,
_output_shapes
:���������U�
 
_user_specified_nameinputs
�
�
J__inference_input_embedding_layer_call_and_return_conditional_losses_16327

inputs;
'token_embedding8_embedding_lookup_16304:
�'�D
1position_embedding8_slice_readvariableop_resource:	U�
identity��(position_embedding8/Slice/ReadVariableOp�!token_embedding8/embedding_lookup�
!token_embedding8/embedding_lookupResourceGather'token_embedding8_embedding_lookup_16304inputs*
Tindices0*:
_class0
.,loc:@token_embedding8/embedding_lookup/16304*,
_output_shapes
:���������U�*
dtype0�
*token_embedding8/embedding_lookup/IdentityIdentity*token_embedding8/embedding_lookup:output:0*
T0*,
_output_shapes
:���������U�]
token_embedding8/NotEqual/yConst*
_output_shapes
: *
dtype0*
value	B : �
token_embedding8/NotEqualNotEqualinputs$token_embedding8/NotEqual/y:output:0*
T0*'
_output_shapes
:���������U�
position_embedding8/ShapeShape3token_embedding8/embedding_lookup/Identity:output:0*
T0*
_output_shapes
::��q
'position_embedding8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)position_embedding8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)position_embedding8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
!position_embedding8/strided_sliceStridedSlice"position_embedding8/Shape:output:00position_embedding8/strided_slice/stack:output:02position_embedding8/strided_slice/stack_1:output:02position_embedding8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
(position_embedding8/Slice/ReadVariableOpReadVariableOp1position_embedding8_slice_readvariableop_resource*
_output_shapes
:	U�*
dtype0p
position_embedding8/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB"        o
position_embedding8/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB"U   �   �
position_embedding8/SliceSlice0position_embedding8/Slice/ReadVariableOp:value:0(position_embedding8/Slice/begin:output:0'position_embedding8/Slice/size:output:0*
Index0*
T0*
_output_shapes
:	U�^
position_embedding8/packed/1Const*
_output_shapes
: *
dtype0*
value	B :U_
position_embedding8/packed/2Const*
_output_shapes
: *
dtype0*
value
B :��
position_embedding8/packedPack*position_embedding8/strided_slice:output:0%position_embedding8/packed/1:output:0%position_embedding8/packed/2:output:0*
N*
T0*
_output_shapes
:Z
position_embedding8/RankConst*
_output_shapes
: *
dtype0*
value	B :�
position_embedding8/BroadcastToBroadcastTo"position_embedding8/Slice:output:0#position_embedding8/packed:output:0*
T0*,
_output_shapes
:���������U��
addAddV23token_embedding8/embedding_lookup/Identity:output:0(position_embedding8/BroadcastTo:output:0*
T0*,
_output_shapes
:���������U�[
IdentityIdentityadd:z:0^NoOp*
T0*,
_output_shapes
:���������U�q
NoOpNoOp)^position_embedding8/Slice/ReadVariableOp"^token_embedding8/embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������U: : 2T
(position_embedding8/Slice/ReadVariableOp(position_embedding8/Slice/ReadVariableOp2F
!token_embedding8/embedding_lookup!token_embedding8/embedding_lookup:($
"
_user_specified_name
resource:%!

_user_specified_name16304:O K
'
_output_shapes
:���������U
 
_user_specified_nameinputs
�
j
!__inference_z_layer_call_fn_16699
inputs_0
inputs_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������** 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *E
f@R>
<__inference_z_layer_call_and_return_conditional_losses_15828o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������*<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������*:���������*22
StatefulPartitionedCallStatefulPartitionedCall:QM
'
_output_shapes
:���������*
"
_user_specified_name
inputs_1:Q M
'
_output_shapes
:���������*
"
_user_specified_name
inputs_0
�9
�
B__inference_encoder_layer_call_and_return_conditional_losses_15833
input_word_ids)
input_embedding_15604:
�'�(
input_embedding_15606:	U�%
encoding_15738:�  
encoding_15740: %
encoding_15742:�  
encoding_15744: %
encoding_15746:�  
encoding_15748: %
encoding_15750: �
encoding_15752:	�
encoding_15754:	�
encoding_15756:	�!
encoding_15758:	�*
encoding_15760:*!
encoding_15762:	*�
encoding_15764:	�
encoding_15766:	�
encoding_15768:	�
z_mean_15789:	�*
z_mean_15791:*"
z_log_var_15804:	�*
z_log_var_15806:*
identity

identity_1

identity_2�� encoding/StatefulPartitionedCall�'input_embedding/StatefulPartitionedCall�z/StatefulPartitionedCall�!z_log_var/StatefulPartitionedCall�z_mean/StatefulPartitionedCall�
'input_embedding/StatefulPartitionedCallStatefulPartitionedCallinput_word_idsinput_embedding_15604input_embedding_15606*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������U�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_input_embedding_layer_call_and_return_conditional_losses_15603\
input_embedding/NotEqual/yConst*
_output_shapes
: *
dtype0*
value	B : �
input_embedding/NotEqualNotEqualinput_word_ids#input_embedding/NotEqual/y:output:0*
T0*'
_output_shapes
:���������U�
 encoding/StatefulPartitionedCallStatefulPartitionedCall0input_embedding/StatefulPartitionedCall:output:0encoding_15738encoding_15740encoding_15742encoding_15744encoding_15746encoding_15748encoding_15750encoding_15752encoding_15754encoding_15756encoding_15758encoding_15760encoding_15762encoding_15764encoding_15766encoding_15768*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������U�*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_encoding_layer_call_and_return_conditional_losses_15737�
.tf.__operators__.getitem_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            �
0tf.__operators__.getitem_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           �
0tf.__operators__.getitem_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
(tf.__operators__.getitem_7/strided_sliceStridedSlice)encoding/StatefulPartitionedCall:output:07tf.__operators__.getitem_7/strided_slice/stack:output:09tf.__operators__.getitem_7/strided_slice/stack_1:output:09tf.__operators__.getitem_7/strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*

begin_mask*
end_mask*
shrink_axis_mask�
.tf.__operators__.getitem_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            �
0tf.__operators__.getitem_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           �
0tf.__operators__.getitem_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
(tf.__operators__.getitem_6/strided_sliceStridedSlice)encoding/StatefulPartitionedCall:output:07tf.__operators__.getitem_6/strided_slice/stack:output:09tf.__operators__.getitem_6/strided_slice/stack_1:output:09tf.__operators__.getitem_6/strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*

begin_mask*
end_mask*
shrink_axis_mask�
z_mean/StatefulPartitionedCallStatefulPartitionedCall1tf.__operators__.getitem_6/strided_slice:output:0z_mean_15789z_mean_15791*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_z_mean_layer_call_and_return_conditional_losses_15788�
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall1tf.__operators__.getitem_7/strided_slice:output:0z_log_var_15804z_log_var_15806*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_z_log_var_layer_call_and_return_conditional_losses_15803�
z/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������** 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *E
f@R>
<__inference_z_layer_call_and_return_conditional_losses_15828v
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������*{

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������*s

Identity_2Identity"z/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������*�
NoOpNoOp!^encoding/StatefulPartitionedCall(^input_embedding/StatefulPartitionedCall^z/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������U: : : : : : : : : : : : : : : : : : : : : : 2D
 encoding/StatefulPartitionedCall encoding/StatefulPartitionedCall2R
'input_embedding/StatefulPartitionedCall'input_embedding/StatefulPartitionedCall26
z/StatefulPartitionedCallz/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:%!

_user_specified_name15806:%!

_user_specified_name15804:%!

_user_specified_name15791:%!

_user_specified_name15789:%!

_user_specified_name15768:%!

_user_specified_name15766:%!

_user_specified_name15764:%!

_user_specified_name15762:%!

_user_specified_name15760:%!

_user_specified_name15758:%!

_user_specified_name15756:%!

_user_specified_name15754:%
!

_user_specified_name15752:%	!

_user_specified_name15750:%!

_user_specified_name15748:%!

_user_specified_name15746:%!

_user_specified_name15744:%!

_user_specified_name15742:%!

_user_specified_name15740:%!

_user_specified_name15738:%!

_user_specified_name15606:%!

_user_specified_name15604:W S
'
_output_shapes
:���������U
(
_user_specified_nameinput_word_ids
�
�
#__inference_signature_wrapper_16292
input_word_ids
unknown:
�'�
	unknown_0:	U� 
	unknown_1:� 
	unknown_2:  
	unknown_3:� 
	unknown_4:  
	unknown_5:� 
	unknown_6:  
	unknown_7: �
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:	�*

unknown_12:*

unknown_13:	*�

unknown_14:	�

unknown_15:	�

unknown_16:	�

unknown_17:	�*

unknown_18:*

unknown_19:	�*

unknown_20:*
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_word_idsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������*:���������*:���������**8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__wrapped_model_15575o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������*q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������*q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:���������*<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������U: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name16284:%!

_user_specified_name16282:%!

_user_specified_name16280:%!

_user_specified_name16278:%!

_user_specified_name16276:%!

_user_specified_name16274:%!

_user_specified_name16272:%!

_user_specified_name16270:%!

_user_specified_name16268:%!

_user_specified_name16266:%!

_user_specified_name16264:%!

_user_specified_name16262:%
!

_user_specified_name16260:%	!

_user_specified_name16258:%!

_user_specified_name16256:%!

_user_specified_name16254:%!

_user_specified_name16252:%!

_user_specified_name16250:%!

_user_specified_name16248:%!

_user_specified_name16246:%!

_user_specified_name16244:%!

_user_specified_name16242:W S
'
_output_shapes
:���������U
(
_user_specified_nameinput_word_ids
��
�
C__inference_encoding_layer_call_and_return_conditional_losses_16528

inputsW
@multi_head_attention_query_einsum_einsum_readvariableop_resource:� H
6multi_head_attention_query_add_readvariableop_resource: U
>multi_head_attention_key_einsum_einsum_readvariableop_resource:� F
4multi_head_attention_key_add_readvariableop_resource: W
@multi_head_attention_value_einsum_einsum_readvariableop_resource:� H
6multi_head_attention_value_add_readvariableop_resource: b
Kmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource: �P
Amulti_head_attention_attention_output_add_readvariableop_resource:	�H
9layer_normalization_batchnorm_mul_readvariableop_resource:	�D
5layer_normalization_batchnorm_readvariableop_resource:	�:
'dense_tensordot_readvariableop_resource:	�*3
%dense_biasadd_readvariableop_resource:*<
)dense_1_tensordot_readvariableop_resource:	*�6
'dense_1_biasadd_readvariableop_resource:	�J
;layer_normalization_1_batchnorm_mul_readvariableop_resource:	�F
7layer_normalization_1_batchnorm_readvariableop_resource:	�
identity��dense/BiasAdd/ReadVariableOp�dense/Tensordot/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp� dense_1/Tensordot/ReadVariableOp�,layer_normalization/batchnorm/ReadVariableOp�0layer_normalization/batchnorm/mul/ReadVariableOp�.layer_normalization_1/batchnorm/ReadVariableOp�2layer_normalization_1/batchnorm/mul/ReadVariableOp�8multi_head_attention/attention_output/add/ReadVariableOp�Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp�+multi_head_attention/key/add/ReadVariableOp�5multi_head_attention/key/einsum/Einsum/ReadVariableOp�-multi_head_attention/query/add/ReadVariableOp�7multi_head_attention/query/einsum/Einsum/ReadVariableOp�-multi_head_attention/value/add/ReadVariableOp�7multi_head_attention/value/einsum/Einsum/ReadVariableOp�
7multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_query_einsum_einsum_readvariableop_resource*#
_output_shapes
:� *
dtype0�
(multi_head_attention/query/einsum/EinsumEinsuminputs?multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������U *
equationabc,cde->abde�
-multi_head_attention/query/add/ReadVariableOpReadVariableOp6multi_head_attention_query_add_readvariableop_resource*
_output_shapes

: *
dtype0�
multi_head_attention/query/addAddV21multi_head_attention/query/einsum/Einsum:output:05multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������U �
5multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp>multi_head_attention_key_einsum_einsum_readvariableop_resource*#
_output_shapes
:� *
dtype0�
&multi_head_attention/key/einsum/EinsumEinsuminputs=multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������U *
equationabc,cde->abde�
+multi_head_attention/key/add/ReadVariableOpReadVariableOp4multi_head_attention_key_add_readvariableop_resource*
_output_shapes

: *
dtype0�
multi_head_attention/key/addAddV2/multi_head_attention/key/einsum/Einsum:output:03multi_head_attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������U �
7multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_value_einsum_einsum_readvariableop_resource*#
_output_shapes
:� *
dtype0�
(multi_head_attention/value/einsum/EinsumEinsuminputs?multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������U *
equationabc,cde->abde�
-multi_head_attention/value/add/ReadVariableOpReadVariableOp6multi_head_attention_value_add_readvariableop_resource*
_output_shapes

: *
dtype0�
multi_head_attention/value/addAddV21multi_head_attention/value/einsum/Einsum:output:05multi_head_attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������U _
multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *�5>�
multi_head_attention/MulMul"multi_head_attention/query/add:z:0#multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:���������U �
"multi_head_attention/einsum/EinsumEinsum multi_head_attention/key/add:z:0multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:���������UU*
equationaecd,abcd->acbe�
$multi_head_attention/softmax/SoftmaxSoftmax+multi_head_attention/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������UU�
%multi_head_attention/dropout/IdentityIdentity.multi_head_attention/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:���������UU�
$multi_head_attention/einsum_1/EinsumEinsum.multi_head_attention/dropout/Identity:output:0"multi_head_attention/value/add:z:0*
N*
T0*/
_output_shapes
:���������U *
equationacbe,aecd->abcd�
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpKmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource*#
_output_shapes
: �*
dtype0�
3multi_head_attention/attention_output/einsum/EinsumEinsum-multi_head_attention/einsum_1/Einsum:output:0Jmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:���������U�*
equationabcd,cde->abe�
8multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpAmulti_head_attention_attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
)multi_head_attention/attention_output/addAddV2<multi_head_attention/attention_output/einsum/Einsum:output:0@multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������U��
dropout_1/IdentityIdentity-multi_head_attention/attention_output/add:z:0*
T0*,
_output_shapes
:���������U�h
addAddV2dropout_1/Identity:output:0inputs*
T0*,
_output_shapes
:���������U�|
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
 layer_normalization/moments/meanMeanadd:z:0;layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������U*
	keep_dims(�
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:���������U�
-layer_normalization/moments/SquaredDifferenceSquaredDifferenceadd:z:01layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:���������U��
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������U*
	keep_dims(h
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������U�
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:���������U�
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������U��
#layer_normalization/batchnorm/mul_1Muladd:z:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������U��
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������U��
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:���������U��
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������U��
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes
:	�**
dtype0^
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:e
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       z
dense/Tensordot/ShapeShape'layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
::��_
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:_
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: a
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ]
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
dense/Tensordot/transpose	Transpose'layer_normalization/batchnorm/add_1:z:0dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������U��
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������*a
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:*_
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������U*~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:**
dtype0�
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������U*`

dense/ReluReludense/BiasAdd:output:0*
T0*+
_output_shapes
:���������U*�
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes
:	*�*
dtype0`
dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       m
dense_1/Tensordot/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
::��a
dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_1/Tensordot/GatherV2GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/free:output:0(dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_1/Tensordot/GatherV2_1GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/axes:output:0*dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
dense_1/Tensordot/ProdProd#dense_1/Tensordot/GatherV2:output:0 dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense_1/Tensordot/Prod_1Prod%dense_1/Tensordot/GatherV2_1:output:0"dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_1/Tensordot/concatConcatV2dense_1/Tensordot/free:output:0dense_1/Tensordot/axes:output:0&dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
dense_1/Tensordot/stackPackdense_1/Tensordot/Prod:output:0!dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
dense_1/Tensordot/transpose	Transposedense/Relu:activations:0!dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������U*�
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�a
dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_1/Tensordot/concat_1ConcatV2#dense_1/Tensordot/GatherV2:output:0"dense_1/Tensordot/Const_2:output:0(dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0#dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������U��
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������U�o
dropout_2/IdentityIdentitydense_1/BiasAdd:output:0*
T0*,
_output_shapes
:���������U��
add_1AddV2dropout_2/Identity:output:0'layer_normalization/batchnorm/add_1:z:0*
T0*,
_output_shapes
:���������U�~
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
"layer_normalization_1/moments/meanMean	add_1:z:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������U*
	keep_dims(�
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:���������U�
/layer_normalization_1/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:���������U��
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������U*
	keep_dims(j
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������U�
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:���������U�
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������U��
%layer_normalization_1/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������U��
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������U��
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:���������U��
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������U�}
IdentityIdentity)layer_normalization_1/batchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:���������U��
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp9^multi_head_attention/attention_output/add/ReadVariableOpC^multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp,^multi_head_attention/key/add/ReadVariableOp6^multi_head_attention/key/einsum/Einsum/ReadVariableOp.^multi_head_attention/query/add/ReadVariableOp8^multi_head_attention/query/einsum/Einsum/ReadVariableOp.^multi_head_attention/value/add/ReadVariableOp8^multi_head_attention/value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:���������U�: : : : : : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp2\
,layer_normalization/batchnorm/ReadVariableOp,layer_normalization/batchnorm/ReadVariableOp2d
0layer_normalization/batchnorm/mul/ReadVariableOp0layer_normalization/batchnorm/mul/ReadVariableOp2`
.layer_normalization_1/batchnorm/ReadVariableOp.layer_normalization_1/batchnorm/ReadVariableOp2h
2layer_normalization_1/batchnorm/mul/ReadVariableOp2layer_normalization_1/batchnorm/mul/ReadVariableOp2t
8multi_head_attention/attention_output/add/ReadVariableOp8multi_head_attention/attention_output/add/ReadVariableOp2�
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpBmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp2Z
+multi_head_attention/key/add/ReadVariableOp+multi_head_attention/key/add/ReadVariableOp2n
5multi_head_attention/key/einsum/Einsum/ReadVariableOp5multi_head_attention/key/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/query/add/ReadVariableOp-multi_head_attention/query/add/ReadVariableOp2r
7multi_head_attention/query/einsum/Einsum/ReadVariableOp7multi_head_attention/query/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/value/add/ReadVariableOp-multi_head_attention/value/add/ReadVariableOp2r
7multi_head_attention/value/einsum/Einsum/ReadVariableOp7multi_head_attention/value/einsum/Einsum/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:T P
,
_output_shapes
:���������U�
 
_user_specified_nameinputs
�
k
<__inference_z_layer_call_and_return_conditional_losses_16719
inputs_0
inputs_1
identity�K
ShapeShapeinputs_0*
T0*
_output_shapes
::��W
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*'
_output_shapes
:���������**
dtype0*
seed2���*
seed���)�
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:���������*|
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:���������*J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?V
mulMulmul/x:output:0inputs_1*
T0*'
_output_shapes
:���������*E
ExpExpmul:z:0*
T0*'
_output_shapes
:���������*Z
mul_1MulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:���������*S
addAddV2inputs_0	mul_1:z:0*
T0*'
_output_shapes
:���������*L
mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �BY
mul_2Muladd:z:0mul_2/y:output:0*
T0*'
_output_shapes
:���������*K
RoundRound	mul_2:z:0*
T0*'
_output_shapes
:���������*X
CastCast	Round:y:0*

DstT0*

SrcT0*'
_output_shapes
:���������*P
IdentityIdentityCast:y:0*
T0*'
_output_shapes
:���������*"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������*:���������*:QM
'
_output_shapes
:���������*
"
_user_specified_name
inputs_1:Q M
'
_output_shapes
:���������*
"
_user_specified_name
inputs_0
�	
�
D__inference_z_log_var_layer_call_and_return_conditional_losses_16693

inputs1
matmul_readvariableop_resource:	�*-
biasadd_readvariableop_resource:*
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�**
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������*r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:**
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������*_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������*S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
��
�
C__inference_encoding_layer_call_and_return_conditional_losses_15969

inputsW
@multi_head_attention_query_einsum_einsum_readvariableop_resource:� H
6multi_head_attention_query_add_readvariableop_resource: U
>multi_head_attention_key_einsum_einsum_readvariableop_resource:� F
4multi_head_attention_key_add_readvariableop_resource: W
@multi_head_attention_value_einsum_einsum_readvariableop_resource:� H
6multi_head_attention_value_add_readvariableop_resource: b
Kmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource: �P
Amulti_head_attention_attention_output_add_readvariableop_resource:	�H
9layer_normalization_batchnorm_mul_readvariableop_resource:	�D
5layer_normalization_batchnorm_readvariableop_resource:	�:
'dense_tensordot_readvariableop_resource:	�*3
%dense_biasadd_readvariableop_resource:*<
)dense_1_tensordot_readvariableop_resource:	*�6
'dense_1_biasadd_readvariableop_resource:	�J
;layer_normalization_1_batchnorm_mul_readvariableop_resource:	�F
7layer_normalization_1_batchnorm_readvariableop_resource:	�
identity��dense/BiasAdd/ReadVariableOp�dense/Tensordot/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp� dense_1/Tensordot/ReadVariableOp�,layer_normalization/batchnorm/ReadVariableOp�0layer_normalization/batchnorm/mul/ReadVariableOp�.layer_normalization_1/batchnorm/ReadVariableOp�2layer_normalization_1/batchnorm/mul/ReadVariableOp�8multi_head_attention/attention_output/add/ReadVariableOp�Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp�+multi_head_attention/key/add/ReadVariableOp�5multi_head_attention/key/einsum/Einsum/ReadVariableOp�-multi_head_attention/query/add/ReadVariableOp�7multi_head_attention/query/einsum/Einsum/ReadVariableOp�-multi_head_attention/value/add/ReadVariableOp�7multi_head_attention/value/einsum/Einsum/ReadVariableOp�
7multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_query_einsum_einsum_readvariableop_resource*#
_output_shapes
:� *
dtype0�
(multi_head_attention/query/einsum/EinsumEinsuminputs?multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������U *
equationabc,cde->abde�
-multi_head_attention/query/add/ReadVariableOpReadVariableOp6multi_head_attention_query_add_readvariableop_resource*
_output_shapes

: *
dtype0�
multi_head_attention/query/addAddV21multi_head_attention/query/einsum/Einsum:output:05multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������U �
5multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp>multi_head_attention_key_einsum_einsum_readvariableop_resource*#
_output_shapes
:� *
dtype0�
&multi_head_attention/key/einsum/EinsumEinsuminputs=multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������U *
equationabc,cde->abde�
+multi_head_attention/key/add/ReadVariableOpReadVariableOp4multi_head_attention_key_add_readvariableop_resource*
_output_shapes

: *
dtype0�
multi_head_attention/key/addAddV2/multi_head_attention/key/einsum/Einsum:output:03multi_head_attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������U �
7multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_value_einsum_einsum_readvariableop_resource*#
_output_shapes
:� *
dtype0�
(multi_head_attention/value/einsum/EinsumEinsuminputs?multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������U *
equationabc,cde->abde�
-multi_head_attention/value/add/ReadVariableOpReadVariableOp6multi_head_attention_value_add_readvariableop_resource*
_output_shapes

: *
dtype0�
multi_head_attention/value/addAddV21multi_head_attention/value/einsum/Einsum:output:05multi_head_attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������U _
multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *�5>�
multi_head_attention/MulMul"multi_head_attention/query/add:z:0#multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:���������U �
"multi_head_attention/einsum/EinsumEinsum multi_head_attention/key/add:z:0multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:���������UU*
equationaecd,abcd->acbe�
$multi_head_attention/softmax/SoftmaxSoftmax+multi_head_attention/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������UU�
%multi_head_attention/dropout/IdentityIdentity.multi_head_attention/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:���������UU�
$multi_head_attention/einsum_1/EinsumEinsum.multi_head_attention/dropout/Identity:output:0"multi_head_attention/value/add:z:0*
N*
T0*/
_output_shapes
:���������U *
equationacbe,aecd->abcd�
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpKmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource*#
_output_shapes
: �*
dtype0�
3multi_head_attention/attention_output/einsum/EinsumEinsum-multi_head_attention/einsum_1/Einsum:output:0Jmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:���������U�*
equationabcd,cde->abe�
8multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpAmulti_head_attention_attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
)multi_head_attention/attention_output/addAddV2<multi_head_attention/attention_output/einsum/Einsum:output:0@multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������U��
dropout_1/IdentityIdentity-multi_head_attention/attention_output/add:z:0*
T0*,
_output_shapes
:���������U�h
addAddV2dropout_1/Identity:output:0inputs*
T0*,
_output_shapes
:���������U�|
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
 layer_normalization/moments/meanMeanadd:z:0;layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������U*
	keep_dims(�
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:���������U�
-layer_normalization/moments/SquaredDifferenceSquaredDifferenceadd:z:01layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:���������U��
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������U*
	keep_dims(h
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������U�
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:���������U�
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������U��
#layer_normalization/batchnorm/mul_1Muladd:z:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������U��
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������U��
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:���������U��
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������U��
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes
:	�**
dtype0^
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:e
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       z
dense/Tensordot/ShapeShape'layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
::��_
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:_
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: a
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ]
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
dense/Tensordot/transpose	Transpose'layer_normalization/batchnorm/add_1:z:0dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������U��
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������*a
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:*_
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������U*~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:**
dtype0�
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������U*`

dense/ReluReludense/BiasAdd:output:0*
T0*+
_output_shapes
:���������U*�
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes
:	*�*
dtype0`
dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       m
dense_1/Tensordot/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
::��a
dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_1/Tensordot/GatherV2GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/free:output:0(dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_1/Tensordot/GatherV2_1GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/axes:output:0*dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
dense_1/Tensordot/ProdProd#dense_1/Tensordot/GatherV2:output:0 dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense_1/Tensordot/Prod_1Prod%dense_1/Tensordot/GatherV2_1:output:0"dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_1/Tensordot/concatConcatV2dense_1/Tensordot/free:output:0dense_1/Tensordot/axes:output:0&dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
dense_1/Tensordot/stackPackdense_1/Tensordot/Prod:output:0!dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
dense_1/Tensordot/transpose	Transposedense/Relu:activations:0!dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������U*�
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�a
dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_1/Tensordot/concat_1ConcatV2#dense_1/Tensordot/GatherV2:output:0"dense_1/Tensordot/Const_2:output:0(dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0#dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������U��
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������U�o
dropout_2/IdentityIdentitydense_1/BiasAdd:output:0*
T0*,
_output_shapes
:���������U��
add_1AddV2dropout_2/Identity:output:0'layer_normalization/batchnorm/add_1:z:0*
T0*,
_output_shapes
:���������U�~
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
"layer_normalization_1/moments/meanMean	add_1:z:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������U*
	keep_dims(�
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:���������U�
/layer_normalization_1/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:���������U��
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������U*
	keep_dims(j
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������U�
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:���������U�
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������U��
%layer_normalization_1/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������U��
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������U��
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:���������U��
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������U�}
IdentityIdentity)layer_normalization_1/batchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:���������U��
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp9^multi_head_attention/attention_output/add/ReadVariableOpC^multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp,^multi_head_attention/key/add/ReadVariableOp6^multi_head_attention/key/einsum/Einsum/ReadVariableOp.^multi_head_attention/query/add/ReadVariableOp8^multi_head_attention/query/einsum/Einsum/ReadVariableOp.^multi_head_attention/value/add/ReadVariableOp8^multi_head_attention/value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:���������U�: : : : : : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp2\
,layer_normalization/batchnorm/ReadVariableOp,layer_normalization/batchnorm/ReadVariableOp2d
0layer_normalization/batchnorm/mul/ReadVariableOp0layer_normalization/batchnorm/mul/ReadVariableOp2`
.layer_normalization_1/batchnorm/ReadVariableOp.layer_normalization_1/batchnorm/ReadVariableOp2h
2layer_normalization_1/batchnorm/mul/ReadVariableOp2layer_normalization_1/batchnorm/mul/ReadVariableOp2t
8multi_head_attention/attention_output/add/ReadVariableOp8multi_head_attention/attention_output/add/ReadVariableOp2�
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpBmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp2Z
+multi_head_attention/key/add/ReadVariableOp+multi_head_attention/key/add/ReadVariableOp2n
5multi_head_attention/key/einsum/Einsum/ReadVariableOp5multi_head_attention/key/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/query/add/ReadVariableOp-multi_head_attention/query/add/ReadVariableOp2r
7multi_head_attention/query/einsum/Einsum/ReadVariableOp7multi_head_attention/query/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/value/add/ReadVariableOp-multi_head_attention/value/add/ReadVariableOp2r
7multi_head_attention/value/einsum/Einsum/ReadVariableOp7multi_head_attention/value/einsum/Einsum/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:T P
,
_output_shapes
:���������U�
 
_user_specified_nameinputs
�	
�
A__inference_z_mean_layer_call_and_return_conditional_losses_15788

inputs1
matmul_readvariableop_resource:	�*-
biasadd_readvariableop_resource:*
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�**
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������*r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:**
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������*_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������*S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
'__inference_encoder_layer_call_fn_16078
input_word_ids
unknown:
�'�
	unknown_0:	U� 
	unknown_1:� 
	unknown_2:  
	unknown_3:� 
	unknown_4:  
	unknown_5:� 
	unknown_6:  
	unknown_7: �
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:	�*

unknown_12:*

unknown_13:	*�

unknown_14:	�

unknown_15:	�

unknown_16:	�

unknown_17:	�*

unknown_18:*

unknown_19:	�*

unknown_20:*
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_word_idsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������*:���������*:���������**8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_15833o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������*q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������*q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:���������*<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������U: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name16070:%!

_user_specified_name16068:%!

_user_specified_name16066:%!

_user_specified_name16064:%!

_user_specified_name16062:%!

_user_specified_name16060:%!

_user_specified_name16058:%!

_user_specified_name16056:%!

_user_specified_name16054:%!

_user_specified_name16052:%!

_user_specified_name16050:%!

_user_specified_name16048:%
!

_user_specified_name16046:%	!

_user_specified_name16044:%!

_user_specified_name16042:%!

_user_specified_name16040:%!

_user_specified_name16038:%!

_user_specified_name16036:%!

_user_specified_name16034:%!

_user_specified_name16032:%!

_user_specified_name16030:%!

_user_specified_name16028:W S
'
_output_shapes
:���������U
(
_user_specified_nameinput_word_ids
�9
�
B__inference_encoder_layer_call_and_return_conditional_losses_16025
input_word_ids)
input_embedding_15836:
�'�(
input_embedding_15838:	U�%
encoding_15970:�  
encoding_15972: %
encoding_15974:�  
encoding_15976: %
encoding_15978:�  
encoding_15980: %
encoding_15982: �
encoding_15984:	�
encoding_15986:	�
encoding_15988:	�!
encoding_15990:	�*
encoding_15992:*!
encoding_15994:	*�
encoding_15996:	�
encoding_15998:	�
encoding_16000:	�
z_mean_16011:	�*
z_mean_16013:*"
z_log_var_16016:	�*
z_log_var_16018:*
identity

identity_1

identity_2�� encoding/StatefulPartitionedCall�'input_embedding/StatefulPartitionedCall�z/StatefulPartitionedCall�!z_log_var/StatefulPartitionedCall�z_mean/StatefulPartitionedCall�
'input_embedding/StatefulPartitionedCallStatefulPartitionedCallinput_word_idsinput_embedding_15836input_embedding_15838*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������U�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_input_embedding_layer_call_and_return_conditional_losses_15603\
input_embedding/NotEqual/yConst*
_output_shapes
: *
dtype0*
value	B : �
input_embedding/NotEqualNotEqualinput_word_ids#input_embedding/NotEqual/y:output:0*
T0*'
_output_shapes
:���������U�
 encoding/StatefulPartitionedCallStatefulPartitionedCall0input_embedding/StatefulPartitionedCall:output:0encoding_15970encoding_15972encoding_15974encoding_15976encoding_15978encoding_15980encoding_15982encoding_15984encoding_15986encoding_15988encoding_15990encoding_15992encoding_15994encoding_15996encoding_15998encoding_16000*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������U�*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_encoding_layer_call_and_return_conditional_losses_15969�
.tf.__operators__.getitem_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            �
0tf.__operators__.getitem_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           �
0tf.__operators__.getitem_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
(tf.__operators__.getitem_7/strided_sliceStridedSlice)encoding/StatefulPartitionedCall:output:07tf.__operators__.getitem_7/strided_slice/stack:output:09tf.__operators__.getitem_7/strided_slice/stack_1:output:09tf.__operators__.getitem_7/strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*

begin_mask*
end_mask*
shrink_axis_mask�
.tf.__operators__.getitem_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            �
0tf.__operators__.getitem_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           �
0tf.__operators__.getitem_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
(tf.__operators__.getitem_6/strided_sliceStridedSlice)encoding/StatefulPartitionedCall:output:07tf.__operators__.getitem_6/strided_slice/stack:output:09tf.__operators__.getitem_6/strided_slice/stack_1:output:09tf.__operators__.getitem_6/strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*

begin_mask*
end_mask*
shrink_axis_mask�
z_mean/StatefulPartitionedCallStatefulPartitionedCall1tf.__operators__.getitem_6/strided_slice:output:0z_mean_16011z_mean_16013*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_z_mean_layer_call_and_return_conditional_losses_15788�
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall1tf.__operators__.getitem_7/strided_slice:output:0z_log_var_16016z_log_var_16018*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_z_log_var_layer_call_and_return_conditional_losses_15803�
z/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������** 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *E
f@R>
<__inference_z_layer_call_and_return_conditional_losses_15828v
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������*{

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������*s

Identity_2Identity"z/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������*�
NoOpNoOp!^encoding/StatefulPartitionedCall(^input_embedding/StatefulPartitionedCall^z/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������U: : : : : : : : : : : : : : : : : : : : : : 2D
 encoding/StatefulPartitionedCall encoding/StatefulPartitionedCall2R
'input_embedding/StatefulPartitionedCall'input_embedding/StatefulPartitionedCall26
z/StatefulPartitionedCallz/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:%!

_user_specified_name16018:%!

_user_specified_name16016:%!

_user_specified_name16013:%!

_user_specified_name16011:%!

_user_specified_name16000:%!

_user_specified_name15998:%!

_user_specified_name15996:%!

_user_specified_name15994:%!

_user_specified_name15992:%!

_user_specified_name15990:%!

_user_specified_name15988:%!

_user_specified_name15986:%
!

_user_specified_name15984:%	!

_user_specified_name15982:%!

_user_specified_name15980:%!

_user_specified_name15978:%!

_user_specified_name15976:%!

_user_specified_name15974:%!

_user_specified_name15972:%!

_user_specified_name15970:%!

_user_specified_name15838:%!

_user_specified_name15836:W S
'
_output_shapes
:���������U
(
_user_specified_nameinput_word_ids
��
�
 __inference__wrapped_model_15575
input_word_idsS
?encoder_input_embedding_token_embedding8_embedding_lookup_15363:
�'�\
Iencoder_input_embedding_position_embedding8_slice_readvariableop_resource:	U�h
Qencoder_encoding_multi_head_attention_query_einsum_einsum_readvariableop_resource:� Y
Gencoder_encoding_multi_head_attention_query_add_readvariableop_resource: f
Oencoder_encoding_multi_head_attention_key_einsum_einsum_readvariableop_resource:� W
Eencoder_encoding_multi_head_attention_key_add_readvariableop_resource: h
Qencoder_encoding_multi_head_attention_value_einsum_einsum_readvariableop_resource:� Y
Gencoder_encoding_multi_head_attention_value_add_readvariableop_resource: s
\encoder_encoding_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource: �a
Rencoder_encoding_multi_head_attention_attention_output_add_readvariableop_resource:	�Y
Jencoder_encoding_layer_normalization_batchnorm_mul_readvariableop_resource:	�U
Fencoder_encoding_layer_normalization_batchnorm_readvariableop_resource:	�K
8encoder_encoding_dense_tensordot_readvariableop_resource:	�*D
6encoder_encoding_dense_biasadd_readvariableop_resource:*M
:encoder_encoding_dense_1_tensordot_readvariableop_resource:	*�G
8encoder_encoding_dense_1_biasadd_readvariableop_resource:	�[
Lencoder_encoding_layer_normalization_1_batchnorm_mul_readvariableop_resource:	�W
Hencoder_encoding_layer_normalization_1_batchnorm_readvariableop_resource:	�@
-encoder_z_mean_matmul_readvariableop_resource:	�*<
.encoder_z_mean_biasadd_readvariableop_resource:*C
0encoder_z_log_var_matmul_readvariableop_resource:	�*?
1encoder_z_log_var_biasadd_readvariableop_resource:*
identity

identity_1

identity_2��-encoder/encoding/dense/BiasAdd/ReadVariableOp�/encoder/encoding/dense/Tensordot/ReadVariableOp�/encoder/encoding/dense_1/BiasAdd/ReadVariableOp�1encoder/encoding/dense_1/Tensordot/ReadVariableOp�=encoder/encoding/layer_normalization/batchnorm/ReadVariableOp�Aencoder/encoding/layer_normalization/batchnorm/mul/ReadVariableOp�?encoder/encoding/layer_normalization_1/batchnorm/ReadVariableOp�Cencoder/encoding/layer_normalization_1/batchnorm/mul/ReadVariableOp�Iencoder/encoding/multi_head_attention/attention_output/add/ReadVariableOp�Sencoder/encoding/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp�<encoder/encoding/multi_head_attention/key/add/ReadVariableOp�Fencoder/encoding/multi_head_attention/key/einsum/Einsum/ReadVariableOp�>encoder/encoding/multi_head_attention/query/add/ReadVariableOp�Hencoder/encoding/multi_head_attention/query/einsum/Einsum/ReadVariableOp�>encoder/encoding/multi_head_attention/value/add/ReadVariableOp�Hencoder/encoding/multi_head_attention/value/einsum/Einsum/ReadVariableOp�@encoder/input_embedding/position_embedding8/Slice/ReadVariableOp�9encoder/input_embedding/token_embedding8/embedding_lookup�(encoder/z_log_var/BiasAdd/ReadVariableOp�'encoder/z_log_var/MatMul/ReadVariableOp�%encoder/z_mean/BiasAdd/ReadVariableOp�$encoder/z_mean/MatMul/ReadVariableOp�
9encoder/input_embedding/token_embedding8/embedding_lookupResourceGather?encoder_input_embedding_token_embedding8_embedding_lookup_15363input_word_ids*
Tindices0*R
_classH
FDloc:@encoder/input_embedding/token_embedding8/embedding_lookup/15363*,
_output_shapes
:���������U�*
dtype0�
Bencoder/input_embedding/token_embedding8/embedding_lookup/IdentityIdentityBencoder/input_embedding/token_embedding8/embedding_lookup:output:0*
T0*,
_output_shapes
:���������U�u
3encoder/input_embedding/token_embedding8/NotEqual/yConst*
_output_shapes
: *
dtype0*
value	B : �
1encoder/input_embedding/token_embedding8/NotEqualNotEqualinput_word_ids<encoder/input_embedding/token_embedding8/NotEqual/y:output:0*
T0*'
_output_shapes
:���������U�
1encoder/input_embedding/position_embedding8/ShapeShapeKencoder/input_embedding/token_embedding8/embedding_lookup/Identity:output:0*
T0*
_output_shapes
::���
?encoder/input_embedding/position_embedding8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Aencoder/input_embedding/position_embedding8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Aencoder/input_embedding/position_embedding8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
9encoder/input_embedding/position_embedding8/strided_sliceStridedSlice:encoder/input_embedding/position_embedding8/Shape:output:0Hencoder/input_embedding/position_embedding8/strided_slice/stack:output:0Jencoder/input_embedding/position_embedding8/strided_slice/stack_1:output:0Jencoder/input_embedding/position_embedding8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
@encoder/input_embedding/position_embedding8/Slice/ReadVariableOpReadVariableOpIencoder_input_embedding_position_embedding8_slice_readvariableop_resource*
_output_shapes
:	U�*
dtype0�
7encoder/input_embedding/position_embedding8/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB"        �
6encoder/input_embedding/position_embedding8/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB"U   �   �
1encoder/input_embedding/position_embedding8/SliceSliceHencoder/input_embedding/position_embedding8/Slice/ReadVariableOp:value:0@encoder/input_embedding/position_embedding8/Slice/begin:output:0?encoder/input_embedding/position_embedding8/Slice/size:output:0*
Index0*
T0*
_output_shapes
:	U�v
4encoder/input_embedding/position_embedding8/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Uw
4encoder/input_embedding/position_embedding8/packed/2Const*
_output_shapes
: *
dtype0*
value
B :��
2encoder/input_embedding/position_embedding8/packedPackBencoder/input_embedding/position_embedding8/strided_slice:output:0=encoder/input_embedding/position_embedding8/packed/1:output:0=encoder/input_embedding/position_embedding8/packed/2:output:0*
N*
T0*
_output_shapes
:r
0encoder/input_embedding/position_embedding8/RankConst*
_output_shapes
: *
dtype0*
value	B :�
7encoder/input_embedding/position_embedding8/BroadcastToBroadcastTo:encoder/input_embedding/position_embedding8/Slice:output:0;encoder/input_embedding/position_embedding8/packed:output:0*
T0*,
_output_shapes
:���������U��
encoder/input_embedding/addAddV2Kencoder/input_embedding/token_embedding8/embedding_lookup/Identity:output:0@encoder/input_embedding/position_embedding8/BroadcastTo:output:0*
T0*,
_output_shapes
:���������U�d
"encoder/input_embedding/NotEqual/yConst*
_output_shapes
: *
dtype0*
value	B : �
 encoder/input_embedding/NotEqualNotEqualinput_word_ids+encoder/input_embedding/NotEqual/y:output:0*
T0*'
_output_shapes
:���������Ua
encoder/encoding/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
encoder/encoding/ExpandDims
ExpandDims$encoder/input_embedding/NotEqual:z:0(encoder/encoding/ExpandDims/dim:output:0*
T0
*+
_output_shapes
:���������U�
encoder/encoding/CastCast$encoder/encoding/ExpandDims:output:0*

DstT0*

SrcT0
*+
_output_shapes
:���������U�
9encoder/encoding/multi_head_attention/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            �
;encoder/encoding/multi_head_attention/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            �
;encoder/encoding/multi_head_attention/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
3encoder/encoding/multi_head_attention/strided_sliceStridedSlice$encoder/input_embedding/NotEqual:z:0Bencoder/encoding/multi_head_attention/strided_slice/stack:output:0Dencoder/encoding/multi_head_attention/strided_slice/stack_1:output:0Dencoder/encoding/multi_head_attention/strided_slice/stack_2:output:0*
Index0*
T0
*+
_output_shapes
:���������U*

begin_mask*
end_mask*
new_axis_mask�
;encoder/encoding/multi_head_attention/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            �
=encoder/encoding/multi_head_attention/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            �
=encoder/encoding/multi_head_attention/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
5encoder/encoding/multi_head_attention/strided_slice_1StridedSlice$encoder/input_embedding/NotEqual:z:0Dencoder/encoding/multi_head_attention/strided_slice_1/stack:output:0Fencoder/encoding/multi_head_attention/strided_slice_1/stack_1:output:0Fencoder/encoding/multi_head_attention/strided_slice_1/stack_2:output:0*
Index0*
T0
*+
_output_shapes
:���������U*

begin_mask*
end_mask*
new_axis_mask�
)encoder/encoding/multi_head_attention/and
LogicalAnd<encoder/encoding/multi_head_attention/strided_slice:output:0>encoder/encoding/multi_head_attention/strided_slice_1:output:0*+
_output_shapes
:���������UU�
;encoder/encoding/multi_head_attention/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"            �
=encoder/encoding/multi_head_attention/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            �
=encoder/encoding/multi_head_attention/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
5encoder/encoding/multi_head_attention/strided_slice_2StridedSlice$encoder/input_embedding/NotEqual:z:0Dencoder/encoding/multi_head_attention/strided_slice_2/stack:output:0Fencoder/encoding/multi_head_attention/strided_slice_2/stack_1:output:0Fencoder/encoding/multi_head_attention/strided_slice_2/stack_2:output:0*
Index0*
T0
*+
_output_shapes
:���������U*

begin_mask*
end_mask*
new_axis_mask�
+encoder/encoding/multi_head_attention/and_1
LogicalAnd-encoder/encoding/multi_head_attention/and:z:0>encoder/encoding/multi_head_attention/strided_slice_2:output:0*+
_output_shapes
:���������UU�
*encoder/encoding/multi_head_attention/CastCastencoder/encoding/Cast:y:0*

DstT0
*

SrcT0*+
_output_shapes
:���������U�
+encoder/encoding/multi_head_attention/and_2
LogicalAnd.encoder/encoding/multi_head_attention/Cast:y:0/encoder/encoding/multi_head_attention/and_1:z:0*+
_output_shapes
:���������UU�
Hencoder/encoding/multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOpQencoder_encoding_multi_head_attention_query_einsum_einsum_readvariableop_resource*#
_output_shapes
:� *
dtype0�
9encoder/encoding/multi_head_attention/query/einsum/EinsumEinsumencoder/input_embedding/add:z:0Pencoder/encoding/multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������U *
equationabc,cde->abde�
>encoder/encoding/multi_head_attention/query/add/ReadVariableOpReadVariableOpGencoder_encoding_multi_head_attention_query_add_readvariableop_resource*
_output_shapes

: *
dtype0�
/encoder/encoding/multi_head_attention/query/addAddV2Bencoder/encoding/multi_head_attention/query/einsum/Einsum:output:0Fencoder/encoding/multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������U �
Fencoder/encoding/multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOpOencoder_encoding_multi_head_attention_key_einsum_einsum_readvariableop_resource*#
_output_shapes
:� *
dtype0�
7encoder/encoding/multi_head_attention/key/einsum/EinsumEinsumencoder/input_embedding/add:z:0Nencoder/encoding/multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������U *
equationabc,cde->abde�
<encoder/encoding/multi_head_attention/key/add/ReadVariableOpReadVariableOpEencoder_encoding_multi_head_attention_key_add_readvariableop_resource*
_output_shapes

: *
dtype0�
-encoder/encoding/multi_head_attention/key/addAddV2@encoder/encoding/multi_head_attention/key/einsum/Einsum:output:0Dencoder/encoding/multi_head_attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������U �
Hencoder/encoding/multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOpQencoder_encoding_multi_head_attention_value_einsum_einsum_readvariableop_resource*#
_output_shapes
:� *
dtype0�
9encoder/encoding/multi_head_attention/value/einsum/EinsumEinsumencoder/input_embedding/add:z:0Pencoder/encoding/multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������U *
equationabc,cde->abde�
>encoder/encoding/multi_head_attention/value/add/ReadVariableOpReadVariableOpGencoder_encoding_multi_head_attention_value_add_readvariableop_resource*
_output_shapes

: *
dtype0�
/encoder/encoding/multi_head_attention/value/addAddV2Bencoder/encoding/multi_head_attention/value/einsum/Einsum:output:0Fencoder/encoding/multi_head_attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������U p
+encoder/encoding/multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *�5>�
)encoder/encoding/multi_head_attention/MulMul3encoder/encoding/multi_head_attention/query/add:z:04encoder/encoding/multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:���������U �
3encoder/encoding/multi_head_attention/einsum/EinsumEinsum1encoder/encoding/multi_head_attention/key/add:z:0-encoder/encoding/multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:���������UU*
equationaecd,abcd->acbe
4encoder/encoding/multi_head_attention/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
0encoder/encoding/multi_head_attention/ExpandDims
ExpandDims/encoder/encoding/multi_head_attention/and_2:z:0=encoder/encoding/multi_head_attention/ExpandDims/dim:output:0*
T0
*/
_output_shapes
:���������UU�
2encoder/encoding/multi_head_attention/softmax/CastCast9encoder/encoding/multi_head_attention/ExpandDims:output:0*

DstT0*

SrcT0
*/
_output_shapes
:���������UUx
3encoder/encoding/multi_head_attention/softmax/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
1encoder/encoding/multi_head_attention/softmax/subSub<encoder/encoding/multi_head_attention/softmax/sub/x:output:06encoder/encoding/multi_head_attention/softmax/Cast:y:0*
T0*/
_output_shapes
:���������UUx
3encoder/encoding/multi_head_attention/softmax/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *(kn��
1encoder/encoding/multi_head_attention/softmax/mulMul5encoder/encoding/multi_head_attention/softmax/sub:z:0<encoder/encoding/multi_head_attention/softmax/mul/y:output:0*
T0*/
_output_shapes
:���������UU�
1encoder/encoding/multi_head_attention/softmax/addAddV2<encoder/encoding/multi_head_attention/einsum/Einsum:output:05encoder/encoding/multi_head_attention/softmax/mul:z:0*
T0*/
_output_shapes
:���������UU�
5encoder/encoding/multi_head_attention/softmax/SoftmaxSoftmax5encoder/encoding/multi_head_attention/softmax/add:z:0*
T0*/
_output_shapes
:���������UU�
6encoder/encoding/multi_head_attention/dropout/IdentityIdentity?encoder/encoding/multi_head_attention/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:���������UU�
5encoder/encoding/multi_head_attention/einsum_1/EinsumEinsum?encoder/encoding/multi_head_attention/dropout/Identity:output:03encoder/encoding/multi_head_attention/value/add:z:0*
N*
T0*/
_output_shapes
:���������U *
equationacbe,aecd->abcd�
Sencoder/encoding/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOp\encoder_encoding_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource*#
_output_shapes
: �*
dtype0�
Dencoder/encoding/multi_head_attention/attention_output/einsum/EinsumEinsum>encoder/encoding/multi_head_attention/einsum_1/Einsum:output:0[encoder/encoding/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:���������U�*
equationabcd,cde->abe�
Iencoder/encoding/multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpRencoder_encoding_multi_head_attention_attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
:encoder/encoding/multi_head_attention/attention_output/addAddV2Mencoder/encoding/multi_head_attention/attention_output/einsum/Einsum:output:0Qencoder/encoding/multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������U��
#encoder/encoding/dropout_1/IdentityIdentity>encoder/encoding/multi_head_attention/attention_output/add:z:0*
T0*,
_output_shapes
:���������U��
encoder/encoding/addAddV2,encoder/encoding/dropout_1/Identity:output:0encoder/input_embedding/add:z:0*
T0*,
_output_shapes
:���������U��
Cencoder/encoding/layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
1encoder/encoding/layer_normalization/moments/meanMeanencoder/encoding/add:z:0Lencoder/encoding/layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������U*
	keep_dims(�
9encoder/encoding/layer_normalization/moments/StopGradientStopGradient:encoder/encoding/layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:���������U�
>encoder/encoding/layer_normalization/moments/SquaredDifferenceSquaredDifferenceencoder/encoding/add:z:0Bencoder/encoding/layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:���������U��
Gencoder/encoding/layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
5encoder/encoding/layer_normalization/moments/varianceMeanBencoder/encoding/layer_normalization/moments/SquaredDifference:z:0Pencoder/encoding/layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������U*
	keep_dims(y
4encoder/encoding/layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
2encoder/encoding/layer_normalization/batchnorm/addAddV2>encoder/encoding/layer_normalization/moments/variance:output:0=encoder/encoding/layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������U�
4encoder/encoding/layer_normalization/batchnorm/RsqrtRsqrt6encoder/encoding/layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:���������U�
Aencoder/encoding/layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOpJencoder_encoding_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
2encoder/encoding/layer_normalization/batchnorm/mulMul8encoder/encoding/layer_normalization/batchnorm/Rsqrt:y:0Iencoder/encoding/layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������U��
4encoder/encoding/layer_normalization/batchnorm/mul_1Mulencoder/encoding/add:z:06encoder/encoding/layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������U��
4encoder/encoding/layer_normalization/batchnorm/mul_2Mul:encoder/encoding/layer_normalization/moments/mean:output:06encoder/encoding/layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������U��
=encoder/encoding/layer_normalization/batchnorm/ReadVariableOpReadVariableOpFencoder_encoding_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
2encoder/encoding/layer_normalization/batchnorm/subSubEencoder/encoding/layer_normalization/batchnorm/ReadVariableOp:value:08encoder/encoding/layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:���������U��
4encoder/encoding/layer_normalization/batchnorm/add_1AddV28encoder/encoding/layer_normalization/batchnorm/mul_1:z:06encoder/encoding/layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������U��
/encoder/encoding/dense/Tensordot/ReadVariableOpReadVariableOp8encoder_encoding_dense_tensordot_readvariableop_resource*
_output_shapes
:	�**
dtype0o
%encoder/encoding/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:v
%encoder/encoding/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
&encoder/encoding/dense/Tensordot/ShapeShape8encoder/encoding/layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
::��p
.encoder/encoding/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
)encoder/encoding/dense/Tensordot/GatherV2GatherV2/encoder/encoding/dense/Tensordot/Shape:output:0.encoder/encoding/dense/Tensordot/free:output:07encoder/encoding/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:r
0encoder/encoding/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
+encoder/encoding/dense/Tensordot/GatherV2_1GatherV2/encoder/encoding/dense/Tensordot/Shape:output:0.encoder/encoding/dense/Tensordot/axes:output:09encoder/encoding/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:p
&encoder/encoding/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
%encoder/encoding/dense/Tensordot/ProdProd2encoder/encoding/dense/Tensordot/GatherV2:output:0/encoder/encoding/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: r
(encoder/encoding/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
'encoder/encoding/dense/Tensordot/Prod_1Prod4encoder/encoding/dense/Tensordot/GatherV2_1:output:01encoder/encoding/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: n
,encoder/encoding/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
'encoder/encoding/dense/Tensordot/concatConcatV2.encoder/encoding/dense/Tensordot/free:output:0.encoder/encoding/dense/Tensordot/axes:output:05encoder/encoding/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
&encoder/encoding/dense/Tensordot/stackPack.encoder/encoding/dense/Tensordot/Prod:output:00encoder/encoding/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
*encoder/encoding/dense/Tensordot/transpose	Transpose8encoder/encoding/layer_normalization/batchnorm/add_1:z:00encoder/encoding/dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������U��
(encoder/encoding/dense/Tensordot/ReshapeReshape.encoder/encoding/dense/Tensordot/transpose:y:0/encoder/encoding/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
'encoder/encoding/dense/Tensordot/MatMulMatMul1encoder/encoding/dense/Tensordot/Reshape:output:07encoder/encoding/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������*r
(encoder/encoding/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:*p
.encoder/encoding/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
)encoder/encoding/dense/Tensordot/concat_1ConcatV22encoder/encoding/dense/Tensordot/GatherV2:output:01encoder/encoding/dense/Tensordot/Const_2:output:07encoder/encoding/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
 encoder/encoding/dense/TensordotReshape1encoder/encoding/dense/Tensordot/MatMul:product:02encoder/encoding/dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������U*�
-encoder/encoding/dense/BiasAdd/ReadVariableOpReadVariableOp6encoder_encoding_dense_biasadd_readvariableop_resource*
_output_shapes
:**
dtype0�
encoder/encoding/dense/BiasAddBiasAdd)encoder/encoding/dense/Tensordot:output:05encoder/encoding/dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������U*�
encoder/encoding/dense/ReluRelu'encoder/encoding/dense/BiasAdd:output:0*
T0*+
_output_shapes
:���������U*�
1encoder/encoding/dense_1/Tensordot/ReadVariableOpReadVariableOp:encoder_encoding_dense_1_tensordot_readvariableop_resource*
_output_shapes
:	*�*
dtype0q
'encoder/encoding/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:x
'encoder/encoding/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
(encoder/encoding/dense_1/Tensordot/ShapeShape)encoder/encoding/dense/Relu:activations:0*
T0*
_output_shapes
::��r
0encoder/encoding/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
+encoder/encoding/dense_1/Tensordot/GatherV2GatherV21encoder/encoding/dense_1/Tensordot/Shape:output:00encoder/encoding/dense_1/Tensordot/free:output:09encoder/encoding/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:t
2encoder/encoding/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
-encoder/encoding/dense_1/Tensordot/GatherV2_1GatherV21encoder/encoding/dense_1/Tensordot/Shape:output:00encoder/encoding/dense_1/Tensordot/axes:output:0;encoder/encoding/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:r
(encoder/encoding/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
'encoder/encoding/dense_1/Tensordot/ProdProd4encoder/encoding/dense_1/Tensordot/GatherV2:output:01encoder/encoding/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: t
*encoder/encoding/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
)encoder/encoding/dense_1/Tensordot/Prod_1Prod6encoder/encoding/dense_1/Tensordot/GatherV2_1:output:03encoder/encoding/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: p
.encoder/encoding/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
)encoder/encoding/dense_1/Tensordot/concatConcatV20encoder/encoding/dense_1/Tensordot/free:output:00encoder/encoding/dense_1/Tensordot/axes:output:07encoder/encoding/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
(encoder/encoding/dense_1/Tensordot/stackPack0encoder/encoding/dense_1/Tensordot/Prod:output:02encoder/encoding/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
,encoder/encoding/dense_1/Tensordot/transpose	Transpose)encoder/encoding/dense/Relu:activations:02encoder/encoding/dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������U*�
*encoder/encoding/dense_1/Tensordot/ReshapeReshape0encoder/encoding/dense_1/Tensordot/transpose:y:01encoder/encoding/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
)encoder/encoding/dense_1/Tensordot/MatMulMatMul3encoder/encoding/dense_1/Tensordot/Reshape:output:09encoder/encoding/dense_1/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������u
*encoder/encoding/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�r
0encoder/encoding/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
+encoder/encoding/dense_1/Tensordot/concat_1ConcatV24encoder/encoding/dense_1/Tensordot/GatherV2:output:03encoder/encoding/dense_1/Tensordot/Const_2:output:09encoder/encoding/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
"encoder/encoding/dense_1/TensordotReshape3encoder/encoding/dense_1/Tensordot/MatMul:product:04encoder/encoding/dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������U��
/encoder/encoding/dense_1/BiasAdd/ReadVariableOpReadVariableOp8encoder_encoding_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
 encoder/encoding/dense_1/BiasAddBiasAdd+encoder/encoding/dense_1/Tensordot:output:07encoder/encoding/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������U��
#encoder/encoding/dropout_2/IdentityIdentity)encoder/encoding/dense_1/BiasAdd:output:0*
T0*,
_output_shapes
:���������U��
encoder/encoding/add_1AddV2,encoder/encoding/dropout_2/Identity:output:08encoder/encoding/layer_normalization/batchnorm/add_1:z:0*
T0*,
_output_shapes
:���������U��
Eencoder/encoding/layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
3encoder/encoding/layer_normalization_1/moments/meanMeanencoder/encoding/add_1:z:0Nencoder/encoding/layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������U*
	keep_dims(�
;encoder/encoding/layer_normalization_1/moments/StopGradientStopGradient<encoder/encoding/layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:���������U�
@encoder/encoding/layer_normalization_1/moments/SquaredDifferenceSquaredDifferenceencoder/encoding/add_1:z:0Dencoder/encoding/layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:���������U��
Iencoder/encoding/layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
7encoder/encoding/layer_normalization_1/moments/varianceMeanDencoder/encoding/layer_normalization_1/moments/SquaredDifference:z:0Rencoder/encoding/layer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������U*
	keep_dims({
6encoder/encoding/layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
4encoder/encoding/layer_normalization_1/batchnorm/addAddV2@encoder/encoding/layer_normalization_1/moments/variance:output:0?encoder/encoding/layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������U�
6encoder/encoding/layer_normalization_1/batchnorm/RsqrtRsqrt8encoder/encoding/layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:���������U�
Cencoder/encoding/layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpLencoder_encoding_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
4encoder/encoding/layer_normalization_1/batchnorm/mulMul:encoder/encoding/layer_normalization_1/batchnorm/Rsqrt:y:0Kencoder/encoding/layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������U��
6encoder/encoding/layer_normalization_1/batchnorm/mul_1Mulencoder/encoding/add_1:z:08encoder/encoding/layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������U��
6encoder/encoding/layer_normalization_1/batchnorm/mul_2Mul<encoder/encoding/layer_normalization_1/moments/mean:output:08encoder/encoding/layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������U��
?encoder/encoding/layer_normalization_1/batchnorm/ReadVariableOpReadVariableOpHencoder_encoding_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
4encoder/encoding/layer_normalization_1/batchnorm/subSubGencoder/encoding/layer_normalization_1/batchnorm/ReadVariableOp:value:0:encoder/encoding/layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:���������U��
6encoder/encoding/layer_normalization_1/batchnorm/add_1AddV2:encoder/encoding/layer_normalization_1/batchnorm/mul_1:z:08encoder/encoding/layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������U��
6encoder/tf.__operators__.getitem_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            �
8encoder/tf.__operators__.getitem_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           �
8encoder/tf.__operators__.getitem_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
0encoder/tf.__operators__.getitem_7/strided_sliceStridedSlice:encoder/encoding/layer_normalization_1/batchnorm/add_1:z:0?encoder/tf.__operators__.getitem_7/strided_slice/stack:output:0Aencoder/tf.__operators__.getitem_7/strided_slice/stack_1:output:0Aencoder/tf.__operators__.getitem_7/strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*

begin_mask*
end_mask*
shrink_axis_mask�
6encoder/tf.__operators__.getitem_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            �
8encoder/tf.__operators__.getitem_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           �
8encoder/tf.__operators__.getitem_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
0encoder/tf.__operators__.getitem_6/strided_sliceStridedSlice:encoder/encoding/layer_normalization_1/batchnorm/add_1:z:0?encoder/tf.__operators__.getitem_6/strided_slice/stack:output:0Aencoder/tf.__operators__.getitem_6/strided_slice/stack_1:output:0Aencoder/tf.__operators__.getitem_6/strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*

begin_mask*
end_mask*
shrink_axis_mask�
$encoder/z_mean/MatMul/ReadVariableOpReadVariableOp-encoder_z_mean_matmul_readvariableop_resource*
_output_shapes
:	�**
dtype0�
encoder/z_mean/MatMulMatMul9encoder/tf.__operators__.getitem_6/strided_slice:output:0,encoder/z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������*�
%encoder/z_mean/BiasAdd/ReadVariableOpReadVariableOp.encoder_z_mean_biasadd_readvariableop_resource*
_output_shapes
:**
dtype0�
encoder/z_mean/BiasAddBiasAddencoder/z_mean/MatMul:product:0-encoder/z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������*�
'encoder/z_log_var/MatMul/ReadVariableOpReadVariableOp0encoder_z_log_var_matmul_readvariableop_resource*
_output_shapes
:	�**
dtype0�
encoder/z_log_var/MatMulMatMul9encoder/tf.__operators__.getitem_7/strided_slice:output:0/encoder/z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������*�
(encoder/z_log_var/BiasAdd/ReadVariableOpReadVariableOp1encoder_z_log_var_biasadd_readvariableop_resource*
_output_shapes
:**
dtype0�
encoder/z_log_var/BiasAddBiasAdd"encoder/z_log_var/MatMul:product:00encoder/z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������*l
encoder/z/ShapeShapeencoder/z_mean/BiasAdd:output:0*
T0*
_output_shapes
::��a
encoder/z/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    c
encoder/z/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
,encoder/z/random_normal/RandomStandardNormalRandomStandardNormalencoder/z/Shape:output:0*
T0*'
_output_shapes
:���������**
dtype0*
seed2��i*
seed���)�
encoder/z/random_normal/mulMul5encoder/z/random_normal/RandomStandardNormal:output:0'encoder/z/random_normal/stddev:output:0*
T0*'
_output_shapes
:���������*�
encoder/z/random_normalAddV2encoder/z/random_normal/mul:z:0%encoder/z/random_normal/mean:output:0*
T0*'
_output_shapes
:���������*T
encoder/z/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
encoder/z/mulMulencoder/z/mul/x:output:0"encoder/z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:���������*Y
encoder/z/ExpExpencoder/z/mul:z:0*
T0*'
_output_shapes
:���������*x
encoder/z/mul_1Mulencoder/z/Exp:y:0encoder/z/random_normal:z:0*
T0*'
_output_shapes
:���������*~
encoder/z/addAddV2encoder/z_mean/BiasAdd:output:0encoder/z/mul_1:z:0*
T0*'
_output_shapes
:���������*V
encoder/z/mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Bw
encoder/z/mul_2Mulencoder/z/add:z:0encoder/z/mul_2/y:output:0*
T0*'
_output_shapes
:���������*_
encoder/z/RoundRoundencoder/z/mul_2:z:0*
T0*'
_output_shapes
:���������*l
encoder/z/CastCastencoder/z/Round:y:0*

DstT0*

SrcT0*'
_output_shapes
:���������*a
IdentityIdentityencoder/z/Cast:y:0^NoOp*
T0*'
_output_shapes
:���������*s

Identity_1Identity"encoder/z_log_var/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������*p

Identity_2Identityencoder/z_mean/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������*�

NoOpNoOp.^encoder/encoding/dense/BiasAdd/ReadVariableOp0^encoder/encoding/dense/Tensordot/ReadVariableOp0^encoder/encoding/dense_1/BiasAdd/ReadVariableOp2^encoder/encoding/dense_1/Tensordot/ReadVariableOp>^encoder/encoding/layer_normalization/batchnorm/ReadVariableOpB^encoder/encoding/layer_normalization/batchnorm/mul/ReadVariableOp@^encoder/encoding/layer_normalization_1/batchnorm/ReadVariableOpD^encoder/encoding/layer_normalization_1/batchnorm/mul/ReadVariableOpJ^encoder/encoding/multi_head_attention/attention_output/add/ReadVariableOpT^encoder/encoding/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp=^encoder/encoding/multi_head_attention/key/add/ReadVariableOpG^encoder/encoding/multi_head_attention/key/einsum/Einsum/ReadVariableOp?^encoder/encoding/multi_head_attention/query/add/ReadVariableOpI^encoder/encoding/multi_head_attention/query/einsum/Einsum/ReadVariableOp?^encoder/encoding/multi_head_attention/value/add/ReadVariableOpI^encoder/encoding/multi_head_attention/value/einsum/Einsum/ReadVariableOpA^encoder/input_embedding/position_embedding8/Slice/ReadVariableOp:^encoder/input_embedding/token_embedding8/embedding_lookup)^encoder/z_log_var/BiasAdd/ReadVariableOp(^encoder/z_log_var/MatMul/ReadVariableOp&^encoder/z_mean/BiasAdd/ReadVariableOp%^encoder/z_mean/MatMul/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������U: : : : : : : : : : : : : : : : : : : : : : 2^
-encoder/encoding/dense/BiasAdd/ReadVariableOp-encoder/encoding/dense/BiasAdd/ReadVariableOp2b
/encoder/encoding/dense/Tensordot/ReadVariableOp/encoder/encoding/dense/Tensordot/ReadVariableOp2b
/encoder/encoding/dense_1/BiasAdd/ReadVariableOp/encoder/encoding/dense_1/BiasAdd/ReadVariableOp2f
1encoder/encoding/dense_1/Tensordot/ReadVariableOp1encoder/encoding/dense_1/Tensordot/ReadVariableOp2~
=encoder/encoding/layer_normalization/batchnorm/ReadVariableOp=encoder/encoding/layer_normalization/batchnorm/ReadVariableOp2�
Aencoder/encoding/layer_normalization/batchnorm/mul/ReadVariableOpAencoder/encoding/layer_normalization/batchnorm/mul/ReadVariableOp2�
?encoder/encoding/layer_normalization_1/batchnorm/ReadVariableOp?encoder/encoding/layer_normalization_1/batchnorm/ReadVariableOp2�
Cencoder/encoding/layer_normalization_1/batchnorm/mul/ReadVariableOpCencoder/encoding/layer_normalization_1/batchnorm/mul/ReadVariableOp2�
Iencoder/encoding/multi_head_attention/attention_output/add/ReadVariableOpIencoder/encoding/multi_head_attention/attention_output/add/ReadVariableOp2�
Sencoder/encoding/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpSencoder/encoding/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp2|
<encoder/encoding/multi_head_attention/key/add/ReadVariableOp<encoder/encoding/multi_head_attention/key/add/ReadVariableOp2�
Fencoder/encoding/multi_head_attention/key/einsum/Einsum/ReadVariableOpFencoder/encoding/multi_head_attention/key/einsum/Einsum/ReadVariableOp2�
>encoder/encoding/multi_head_attention/query/add/ReadVariableOp>encoder/encoding/multi_head_attention/query/add/ReadVariableOp2�
Hencoder/encoding/multi_head_attention/query/einsum/Einsum/ReadVariableOpHencoder/encoding/multi_head_attention/query/einsum/Einsum/ReadVariableOp2�
>encoder/encoding/multi_head_attention/value/add/ReadVariableOp>encoder/encoding/multi_head_attention/value/add/ReadVariableOp2�
Hencoder/encoding/multi_head_attention/value/einsum/Einsum/ReadVariableOpHencoder/encoding/multi_head_attention/value/einsum/Einsum/ReadVariableOp2�
@encoder/input_embedding/position_embedding8/Slice/ReadVariableOp@encoder/input_embedding/position_embedding8/Slice/ReadVariableOp2v
9encoder/input_embedding/token_embedding8/embedding_lookup9encoder/input_embedding/token_embedding8/embedding_lookup2T
(encoder/z_log_var/BiasAdd/ReadVariableOp(encoder/z_log_var/BiasAdd/ReadVariableOp2R
'encoder/z_log_var/MatMul/ReadVariableOp'encoder/z_log_var/MatMul/ReadVariableOp2N
%encoder/z_mean/BiasAdd/ReadVariableOp%encoder/z_mean/BiasAdd/ReadVariableOp2L
$encoder/z_mean/MatMul/ReadVariableOp$encoder/z_mean/MatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:%!

_user_specified_name15363:W S
'
_output_shapes
:���������U
(
_user_specified_nameinput_word_ids
�
�
)__inference_z_log_var_layer_call_fn_16683

inputs
unknown:	�*
	unknown_0:*
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_z_log_var_layer_call_and_return_conditional_losses_15803o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������*<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name16679:%!

_user_specified_name16677:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
I
input_word_ids7
 serving_default_input_word_ids:0���������U=
	z_log_var0
StatefulPartitionedCall:1���������*:
z_mean0
StatefulPartitionedCall:2���������*5
z0
StatefulPartitionedCall:0���������*tensorflow/serving/predict:��
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
		variables

trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
token_embedding
position_embedding"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_self_attention_layer
 _self_attention_layernorm
!_self_attention_dropout
"_feedforward_layernorm
##_feedforward_intermediate_dense
$_feedforward_output_dense
%_feedforward_dropout"
_tf_keras_layer
(
&	keras_api"
_tf_keras_layer
(
'	keras_api"
_tf_keras_layer
�
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses

.kernel
/bias"
_tf_keras_layer
�
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses

6kernel
7bias"
_tf_keras_layer
�
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses"
_tf_keras_layer
�
>0
?1
@2
A3
B4
C5
D6
E7
F8
G9
H10
I11
J12
K13
L14
M15
N16
O17
.18
/19
620
721"
trackable_list_wrapper
�
>0
?1
@2
A3
B4
C5
D6
E7
F8
G9
H10
I11
J12
K13
L14
M15
N16
O17
.18
/19
620
721"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Pnon_trainable_variables

Qlayers
Rmetrics
Slayer_regularization_losses
Tlayer_metrics
		variables

trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Utrace_0
Vtrace_12�
'__inference_encoder_layer_call_fn_16078
'__inference_encoder_layer_call_fn_16131�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zUtrace_0zVtrace_1
�
Wtrace_0
Xtrace_12�
B__inference_encoder_layer_call_and_return_conditional_losses_15833
B__inference_encoder_layer_call_and_return_conditional_losses_16025�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zWtrace_0zXtrace_1
�B�
 __inference__wrapped_model_15575input_word_ids"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
,
Yserving_default"
signature_map
.
>0
?1"
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
_trace_02�
/__inference_input_embedding_layer_call_fn_16301�
���
FullArgSpec$
args�
jinputs
jstart_index
varargs
 
varkw
 
defaults�
` 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z_trace_0
�
`trace_02�
J__inference_input_embedding_layer_call_and_return_conditional_losses_16327�
���
FullArgSpec$
args�
jinputs
jstart_index
varargs
 
varkw
 
defaults�
` 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z`trace_0
�
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses
>
embeddings"
_tf_keras_layer
�
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses
?
embeddings
?position_embeddings"
_tf_keras_layer
�
@0
A1
B2
C3
D4
E5
F6
G7
H8
I9
J10
K11
L12
M13
N14
O15"
trackable_list_wrapper
�
@0
A1
B2
C3
D4
E5
F6
G7
H8
I9
J10
K11
L12
M13
N14
O15"
trackable_list_wrapper
 "
trackable_list_wrapper
�
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
rtrace_0
strace_12�
(__inference_encoding_layer_call_fn_16364
(__inference_encoding_layer_call_fn_16401�
���
FullArgSpec7
args/�,
jinputs
jpadding_mask
jattention_mask
varargs
 
varkw
 
defaults�

 

 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 zrtrace_0zstrace_1
�
ttrace_0
utrace_12�
C__inference_encoding_layer_call_and_return_conditional_losses_16528
C__inference_encoding_layer_call_and_return_conditional_losses_16655�
���
FullArgSpec7
args/�,
jinputs
jpadding_mask
jattention_mask
varargs
 
varkw
 
defaults�

 

 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 zttrace_0zutrace_1
�
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses
|_query_dense
}
_key_dense
~_value_dense
_softmax
�_dropout_layer
�_output_dense"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
	Hgamma
Ibeta"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
	Jgamma
Kbeta"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

Lkernel
Mbias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

Nkernel
Obias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
"
_generic_user_object
"
_generic_user_object
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_z_mean_layer_call_fn_16664�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_z_mean_layer_call_and_return_conditional_losses_16674�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 :	�*2z_mean/kernel
:*2z_mean/bias
.
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_z_log_var_layer_call_fn_16683�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_z_log_var_layer_call_and_return_conditional_losses_16693�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
#:!	�*2z_log_var/kernel
:*2z_log_var/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
!__inference_z_layer_call_fn_16699�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
<__inference_z_layer_call_and_return_conditional_losses_16719�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.:,
�'�2input_embedding/embeddings
-:+	U�2input_embedding/embeddings
A:?� 2*encoding/multi_head_attention/query/kernel
::8 2(encoding/multi_head_attention/query/bias
?:=� 2(encoding/multi_head_attention/key/kernel
8:6 2&encoding/multi_head_attention/key/bias
A:?� 2*encoding/multi_head_attention/value/kernel
::8 2(encoding/multi_head_attention/value/bias
L:J �25encoding/multi_head_attention/attention_output/kernel
B:@�23encoding/multi_head_attention/attention_output/bias
:�2encoding/gamma
:�2encoding/beta
:�2encoding/gamma
:�2encoding/beta
": 	�*2encoding/kernel
:*2encoding/bias
": 	*�2encoding/kernel
:�2encoding/bias
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_encoder_layer_call_fn_16078input_word_ids"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
'__inference_encoder_layer_call_fn_16131input_word_ids"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_encoder_layer_call_and_return_conditional_losses_15833input_word_ids"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_encoder_layer_call_and_return_conditional_losses_16025input_word_ids"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference_signature_wrapper_16292input_word_ids"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
/__inference_input_embedding_layer_call_fn_16301inputs"�
���
FullArgSpec$
args�
jinputs
jstart_index
varargs
 
varkw
 
defaults�
` 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_input_embedding_layer_call_and_return_conditional_losses_16327inputs"�
���
FullArgSpec$
args�
jinputs
jstart_index
varargs
 
varkw
 
defaults�
` 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
'
>0"
trackable_list_wrapper
'
>0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
'
?0"
trackable_list_wrapper
'
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec$
args�
jinputs
jstart_index
varargs
 
varkw
 
defaults�
` 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec$
args�
jinputs
jstart_index
varargs
 
varkw
 
defaults�
` 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
Q
0
 1
!2
"3
#4
$5
%6"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_encoding_layer_call_fn_16364inputs"�
���
FullArgSpec7
args/�,
jinputs
jpadding_mask
jattention_mask
varargs
 
varkw
 
defaults�

 

 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
(__inference_encoding_layer_call_fn_16401inputs"�
���
FullArgSpec7
args/�,
jinputs
jpadding_mask
jattention_mask
varargs
 
varkw
 
defaults�

 

 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
C__inference_encoding_layer_call_and_return_conditional_losses_16528inputs"�
���
FullArgSpec7
args/�,
jinputs
jpadding_mask
jattention_mask
varargs
 
varkw
 
defaults�

 

 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
C__inference_encoding_layer_call_and_return_conditional_losses_16655inputs"�
���
FullArgSpec7
args/�,
jinputs
jpadding_mask
jattention_mask
varargs
 
varkw
 
defaults�

 

 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
X
@0
A1
B2
C3
D4
E5
F6
G7"
trackable_list_wrapper
X
@0
A1
B2
C3
D4
E5
F6
G7"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 #
defaults�

 

 
p 
p 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 #
defaults�

 

 
p 
p 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

@kernel
Abias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

Bkernel
Cbias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

Dkernel
Ebias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

Fkernel
Gbias"
_tf_keras_layer
.
H0
I1"
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
"
_generic_user_object
.
J0
K1"
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
N0
O1"
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_z_mean_layer_call_fn_16664inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_z_mean_layer_call_and_return_conditional_losses_16674inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_z_log_var_layer_call_fn_16683inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_z_log_var_layer_call_and_return_conditional_losses_16693inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
!__inference_z_layer_call_fn_16699inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
<__inference_z_layer_call_and_return_conditional_losses_16719inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
L
|0
}1
~2
3
�4
�5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
@0
A1"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
"
_generic_user_object
.
F0
G1"
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper�
 __inference__wrapped_model_15575�>?@ABCDEFGHILMNOJK./677�4
-�*
(�%
input_word_ids���������U
� "���
0
	z_log_var#� 
	z_log_var���������*
*
z_mean �
z_mean���������*
 
z�
z���������*�
B__inference_encoder_layer_call_and_return_conditional_losses_15833�>?@ABCDEFGHILMNOJK./67?�<
5�2
(�%
input_word_ids���������U
p

 
� "�|
u�r
$�!

tensor_0_0���������*
$�!

tensor_0_1���������*
$�!

tensor_0_2���������*
� �
B__inference_encoder_layer_call_and_return_conditional_losses_16025�>?@ABCDEFGHILMNOJK./67?�<
5�2
(�%
input_word_ids���������U
p 

 
� "�|
u�r
$�!

tensor_0_0���������*
$�!

tensor_0_1���������*
$�!

tensor_0_2���������*
� �
'__inference_encoder_layer_call_fn_16078�>?@ABCDEFGHILMNOJK./67?�<
5�2
(�%
input_word_ids���������U
p

 
� "o�l
"�
tensor_0���������*
"�
tensor_1���������*
"�
tensor_2���������*�
'__inference_encoder_layer_call_fn_16131�>?@ABCDEFGHILMNOJK./67?�<
5�2
(�%
input_word_ids���������U
p 

 
� "o�l
"�
tensor_0���������*
"�
tensor_1���������*
"�
tensor_2���������*�
C__inference_encoding_layer_call_and_return_conditional_losses_16528�@ABCDEFGHILMNOJKL�I
2�/
%�"
inputs���������U�

 

 
�

trainingp"1�.
'�$
tensor_0���������U�
� �
C__inference_encoding_layer_call_and_return_conditional_losses_16655�@ABCDEFGHILMNOJKL�I
2�/
%�"
inputs���������U�

 

 
�

trainingp "1�.
'�$
tensor_0���������U�
� �
(__inference_encoding_layer_call_fn_16364�@ABCDEFGHILMNOJKL�I
2�/
%�"
inputs���������U�

 

 
�

trainingp"&�#
unknown���������U��
(__inference_encoding_layer_call_fn_16401�@ABCDEFGHILMNOJKL�I
2�/
%�"
inputs���������U�

 

 
�

trainingp "&�#
unknown���������U��
J__inference_input_embedding_layer_call_and_return_conditional_losses_16327l>?3�0
)�&
 �
inputs���������U
` 
� "1�.
'�$
tensor_0���������U�
� �
/__inference_input_embedding_layer_call_fn_16301a>?3�0
)�&
 �
inputs���������U
` 
� "&�#
unknown���������U��
#__inference_signature_wrapper_16292�>?@ABCDEFGHILMNOJK./67I�F
� 
?�<
:
input_word_ids(�%
input_word_ids���������U"���
0
	z_log_var#� 
	z_log_var���������*
*
z_mean �
z_mean���������*
 
z�
z���������*�
<__inference_z_layer_call_and_return_conditional_losses_16719�Z�W
P�M
K�H
"�
inputs_0���������*
"�
inputs_1���������*
� ",�)
"�
tensor_0���������*
� �
!__inference_z_layer_call_fn_16699Z�W
P�M
K�H
"�
inputs_0���������*
"�
inputs_1���������*
� "!�
unknown���������*�
D__inference_z_log_var_layer_call_and_return_conditional_losses_16693d670�-
&�#
!�
inputs����������
� ",�)
"�
tensor_0���������*
� �
)__inference_z_log_var_layer_call_fn_16683Y670�-
&�#
!�
inputs����������
� "!�
unknown���������*�
A__inference_z_mean_layer_call_and_return_conditional_losses_16674d./0�-
&�#
!�
inputs����������
� ",�)
"�
tensor_0���������*
� �
&__inference_z_mean_layer_call_fn_16664Y./0�-
&�#
!�
inputs����������
� "!�
unknown���������*