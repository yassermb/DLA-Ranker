║й 
ж¤
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
╛
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
executor_typestring И
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeИ"serve*2.2.02v2.2.0-0-g2b96f3662b8ыб
Г
conv3d/kernelVarHandleOp*
_output_shapes
: *
dtype0* 
shape:н*
shared_nameconv3d/kernel
|
!conv3d/kernel/Read/ReadVariableOpReadVariableOpconv3d/kernel*+
_output_shapes
:н*
dtype0
n
conv3d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d/bias
g
conv3d/bias/Read/ReadVariableOpReadVariableOpconv3d/bias*
_output_shapes
:*
dtype0
К
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namebatch_normalization/gamma
Г
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
:*
dtype0
И
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namebatch_normalization/beta
Б
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
:*
dtype0
Ц
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!batch_normalization/moving_mean
П
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:*
dtype0
Ю
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization/moving_variance
Ч
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
:*
dtype0
Ж
conv3d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv3d_1/kernel

#conv3d_1/kernel/Read/ReadVariableOpReadVariableOpconv3d_1/kernel**
_output_shapes
:*
dtype0
r
conv3d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_1/bias
k
!conv3d_1/bias/Read/ReadVariableOpReadVariableOpconv3d_1/bias*
_output_shapes
:*
dtype0
О
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_1/gamma
З
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
:*
dtype0
М
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_1/beta
Е
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
:*
dtype0
Ъ
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_1/moving_mean
У
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
:*
dtype0
в
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_1/moving_variance
Ы
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
:*
dtype0
Ж
conv3d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv3d_2/kernel

#conv3d_2/kernel/Read/ReadVariableOpReadVariableOpconv3d_2/kernel**
_output_shapes
:*
dtype0
r
conv3d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_2/bias
k
!conv3d_2/bias/Read/ReadVariableOpReadVariableOpconv3d_2/bias*
_output_shapes
:*
dtype0
О
batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_2/gamma
З
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes
:*
dtype0
М
batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_2/beta
Е
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes
:*
dtype0
Ъ
!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_2/moving_mean
У
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes
:*
dtype0
в
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_2/moving_variance
Ы
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes
:*
dtype0
Ж
conv3d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv3d_3/kernel

#conv3d_3/kernel/Read/ReadVariableOpReadVariableOpconv3d_3/kernel**
_output_shapes
:*
dtype0
r
conv3d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_3/bias
k
!conv3d_3/bias/Read/ReadVariableOpReadVariableOpconv3d_3/bias*
_output_shapes
:*
dtype0
О
batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_3/gamma
З
/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
_output_shapes
:*
dtype0
М
batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_3/beta
Е
.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
_output_shapes
:*
dtype0
Ъ
!batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_3/moving_mean
У
5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
_output_shapes
:*
dtype0
в
%batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_3/moving_variance
Ы
9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
_output_shapes
:*
dtype0
x
layer1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Г
╚*
shared_namelayer1/kernel
q
!layer1/kernel/Read/ReadVariableOpReadVariableOplayer1/kernel* 
_output_shapes
:
Г
╚*
dtype0
o
layer1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:╚*
shared_namelayer1/bias
h
layer1/bias/Read/ReadVariableOpReadVariableOplayer1/bias*
_output_shapes	
:╚*
dtype0
w
layer2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	╚*
shared_namelayer2/kernel
p
!layer2/kernel/Read/ReadVariableOpReadVariableOplayer2/kernel*
_output_shapes
:	╚*
dtype0
n
layer2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namelayer2/bias
g
layer2/bias/Read/ReadVariableOpReadVariableOplayer2/bias*
_output_shapes
:*
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
С
Adam/conv3d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0* 
shape:н*%
shared_nameAdam/conv3d/kernel/m
К
(Adam/conv3d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d/kernel/m*+
_output_shapes
:н*
dtype0
|
Adam/conv3d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv3d/bias/m
u
&Adam/conv3d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d/bias/m*
_output_shapes
:*
dtype0
Ш
 Adam/batch_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/batch_normalization/gamma/m
С
4Adam/batch_normalization/gamma/m/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/m*
_output_shapes
:*
dtype0
Ц
Adam/batch_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/batch_normalization/beta/m
П
3Adam/batch_normalization/beta/m/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/m*
_output_shapes
:*
dtype0
Ф
Adam/conv3d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_1/kernel/m
Н
*Adam/conv3d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_1/kernel/m**
_output_shapes
:*
dtype0
А
Adam/conv3d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv3d_1/bias/m
y
(Adam/conv3d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_1/bias/m*
_output_shapes
:*
dtype0
Ь
"Adam/batch_normalization_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_1/gamma/m
Х
6Adam/batch_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_1/gamma/m*
_output_shapes
:*
dtype0
Ъ
!Adam/batch_normalization_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_1/beta/m
У
5Adam/batch_normalization_1/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_1/beta/m*
_output_shapes
:*
dtype0
Ф
Adam/conv3d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_2/kernel/m
Н
*Adam/conv3d_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_2/kernel/m**
_output_shapes
:*
dtype0
А
Adam/conv3d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv3d_2/bias/m
y
(Adam/conv3d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_2/bias/m*
_output_shapes
:*
dtype0
Ь
"Adam/batch_normalization_2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_2/gamma/m
Х
6Adam/batch_normalization_2/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_2/gamma/m*
_output_shapes
:*
dtype0
Ъ
!Adam/batch_normalization_2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_2/beta/m
У
5Adam/batch_normalization_2/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_2/beta/m*
_output_shapes
:*
dtype0
Ф
Adam/conv3d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_3/kernel/m
Н
*Adam/conv3d_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_3/kernel/m**
_output_shapes
:*
dtype0
А
Adam/conv3d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv3d_3/bias/m
y
(Adam/conv3d_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_3/bias/m*
_output_shapes
:*
dtype0
Ь
"Adam/batch_normalization_3/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_3/gamma/m
Х
6Adam/batch_normalization_3/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_3/gamma/m*
_output_shapes
:*
dtype0
Ъ
!Adam/batch_normalization_3/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_3/beta/m
У
5Adam/batch_normalization_3/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_3/beta/m*
_output_shapes
:*
dtype0
Ж
Adam/layer1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Г
╚*%
shared_nameAdam/layer1/kernel/m

(Adam/layer1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/layer1/kernel/m* 
_output_shapes
:
Г
╚*
dtype0
}
Adam/layer1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:╚*#
shared_nameAdam/layer1/bias/m
v
&Adam/layer1/bias/m/Read/ReadVariableOpReadVariableOpAdam/layer1/bias/m*
_output_shapes	
:╚*
dtype0
Е
Adam/layer2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	╚*%
shared_nameAdam/layer2/kernel/m
~
(Adam/layer2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/layer2/kernel/m*
_output_shapes
:	╚*
dtype0
|
Adam/layer2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/layer2/bias/m
u
&Adam/layer2/bias/m/Read/ReadVariableOpReadVariableOpAdam/layer2/bias/m*
_output_shapes
:*
dtype0
В
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*$
shared_nameAdam/dense/kernel/m
{
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes

:*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:*
dtype0
С
Adam/conv3d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0* 
shape:н*%
shared_nameAdam/conv3d/kernel/v
К
(Adam/conv3d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d/kernel/v*+
_output_shapes
:н*
dtype0
|
Adam/conv3d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv3d/bias/v
u
&Adam/conv3d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d/bias/v*
_output_shapes
:*
dtype0
Ш
 Adam/batch_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/batch_normalization/gamma/v
С
4Adam/batch_normalization/gamma/v/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/v*
_output_shapes
:*
dtype0
Ц
Adam/batch_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/batch_normalization/beta/v
П
3Adam/batch_normalization/beta/v/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/v*
_output_shapes
:*
dtype0
Ф
Adam/conv3d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_1/kernel/v
Н
*Adam/conv3d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_1/kernel/v**
_output_shapes
:*
dtype0
А
Adam/conv3d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv3d_1/bias/v
y
(Adam/conv3d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_1/bias/v*
_output_shapes
:*
dtype0
Ь
"Adam/batch_normalization_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_1/gamma/v
Х
6Adam/batch_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_1/gamma/v*
_output_shapes
:*
dtype0
Ъ
!Adam/batch_normalization_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_1/beta/v
У
5Adam/batch_normalization_1/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_1/beta/v*
_output_shapes
:*
dtype0
Ф
Adam/conv3d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_2/kernel/v
Н
*Adam/conv3d_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_2/kernel/v**
_output_shapes
:*
dtype0
А
Adam/conv3d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv3d_2/bias/v
y
(Adam/conv3d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_2/bias/v*
_output_shapes
:*
dtype0
Ь
"Adam/batch_normalization_2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_2/gamma/v
Х
6Adam/batch_normalization_2/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_2/gamma/v*
_output_shapes
:*
dtype0
Ъ
!Adam/batch_normalization_2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_2/beta/v
У
5Adam/batch_normalization_2/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_2/beta/v*
_output_shapes
:*
dtype0
Ф
Adam/conv3d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_3/kernel/v
Н
*Adam/conv3d_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_3/kernel/v**
_output_shapes
:*
dtype0
А
Adam/conv3d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv3d_3/bias/v
y
(Adam/conv3d_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_3/bias/v*
_output_shapes
:*
dtype0
Ь
"Adam/batch_normalization_3/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_3/gamma/v
Х
6Adam/batch_normalization_3/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_3/gamma/v*
_output_shapes
:*
dtype0
Ъ
!Adam/batch_normalization_3/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_3/beta/v
У
5Adam/batch_normalization_3/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_3/beta/v*
_output_shapes
:*
dtype0
Ж
Adam/layer1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Г
╚*%
shared_nameAdam/layer1/kernel/v

(Adam/layer1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/layer1/kernel/v* 
_output_shapes
:
Г
╚*
dtype0
}
Adam/layer1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:╚*#
shared_nameAdam/layer1/bias/v
v
&Adam/layer1/bias/v/Read/ReadVariableOpReadVariableOpAdam/layer1/bias/v*
_output_shapes	
:╚*
dtype0
Е
Adam/layer2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	╚*%
shared_nameAdam/layer2/kernel/v
~
(Adam/layer2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/layer2/kernel/v*
_output_shapes
:	╚*
dtype0
|
Adam/layer2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/layer2/bias/v
u
&Adam/layer2/bias/v/Read/ReadVariableOpReadVariableOpAdam/layer2/bias/v*
_output_shapes
:*
dtype0
В
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*$
shared_nameAdam/dense/kernel/v
{
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes

:*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
чЕ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*бЕ
valueЦЕBТЕ BКЕ
Р
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer_with_weights-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer_with_weights-8
layer-14
layer-15
layer_with_weights-9
layer-16
layer-17
layer_with_weights-10
layer-18
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
 
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
Ч
 axis
	!gamma
"beta
#moving_mean
$moving_variance
%trainable_variables
&	variables
'regularization_losses
(	keras_api
h

)kernel
*bias
+trainable_variables
,	variables
-regularization_losses
.	keras_api
Ч
/axis
	0gamma
1beta
2moving_mean
3moving_variance
4trainable_variables
5	variables
6regularization_losses
7	keras_api
h

8kernel
9bias
:trainable_variables
;	variables
<regularization_losses
=	keras_api
Ч
>axis
	?gamma
@beta
Amoving_mean
Bmoving_variance
Ctrainable_variables
D	variables
Eregularization_losses
F	keras_api
h

Gkernel
Hbias
Itrainable_variables
J	variables
Kregularization_losses
L	keras_api
Ч
Maxis
	Ngamma
Obeta
Pmoving_mean
Qmoving_variance
Rtrainable_variables
S	variables
Tregularization_losses
U	keras_api
R
Vtrainable_variables
W	variables
Xregularization_losses
Y	keras_api
R
Ztrainable_variables
[	variables
\regularization_losses
]	keras_api
R
^trainable_variables
_	variables
`regularization_losses
a	keras_api
 
R
btrainable_variables
c	variables
dregularization_losses
e	keras_api
h

fkernel
gbias
htrainable_variables
i	variables
jregularization_losses
k	keras_api
R
ltrainable_variables
m	variables
nregularization_losses
o	keras_api
h

pkernel
qbias
rtrainable_variables
s	variables
tregularization_losses
u	keras_api
R
vtrainable_variables
w	variables
xregularization_losses
y	keras_api
h

zkernel
{bias
|trainable_variables
}	variables
~regularization_losses
	keras_api
¤
	Аiter
Бbeta_1
Вbeta_2

Гdecay
Дlearning_ratemфmх!mц"mч)mш*mщ0mъ1mы8mь9mэ?mю@mяGmЁHmёNmЄOmєfmЇgmїpmЎqmўzm°{m∙v·v√!v№"v¤)v■*v 0vА1vБ8vВ9vГ?vД@vЕGvЖHvЗNvИOvЙfvКgvЛpvМqvНzvО{vП
ж
0
1
!2
"3
)4
*5
06
17
88
99
?10
@11
G12
H13
N14
O15
f16
g17
p18
q19
z20
{21
ц
0
1
!2
"3
#4
$5
)6
*7
08
19
210
311
812
913
?14
@15
A16
B17
G18
H19
N20
O21
P22
Q23
f24
g25
p26
q27
z28
{29
 
▓
Еlayer_metrics
trainable_variables
Жmetrics
	variables
 Зlayer_regularization_losses
regularization_losses
Иnon_trainable_variables
Йlayers
 
YW
VARIABLE_VALUEconv3d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv3d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
▓
Кlayer_metrics
Лmetrics
trainable_variables
	variables
 Мlayer_regularization_losses
regularization_losses
Нnon_trainable_variables
Оlayers
 
db
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

!0
"1

!0
"1
#2
$3
 
▓
Пlayer_metrics
Рmetrics
%trainable_variables
&	variables
 Сlayer_regularization_losses
'regularization_losses
Тnon_trainable_variables
Уlayers
[Y
VARIABLE_VALUEconv3d_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3d_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

)0
*1

)0
*1
 
▓
Фlayer_metrics
Хmetrics
+trainable_variables
,	variables
 Цlayer_regularization_losses
-regularization_losses
Чnon_trainable_variables
Шlayers
 
fd
VARIABLE_VALUEbatch_normalization_1/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_1/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_1/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_1/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

00
11

00
11
22
33
 
▓
Щlayer_metrics
Ъmetrics
4trainable_variables
5	variables
 Ыlayer_regularization_losses
6regularization_losses
Ьnon_trainable_variables
Эlayers
[Y
VARIABLE_VALUEconv3d_2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3d_2/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

80
91

80
91
 
▓
Юlayer_metrics
Яmetrics
:trainable_variables
;	variables
 аlayer_regularization_losses
<regularization_losses
бnon_trainable_variables
вlayers
 
fd
VARIABLE_VALUEbatch_normalization_2/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_2/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_2/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_2/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

?0
@1

?0
@1
A2
B3
 
▓
гlayer_metrics
дmetrics
Ctrainable_variables
D	variables
 еlayer_regularization_losses
Eregularization_losses
жnon_trainable_variables
зlayers
[Y
VARIABLE_VALUEconv3d_3/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3d_3/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

G0
H1

G0
H1
 
▓
иlayer_metrics
йmetrics
Itrainable_variables
J	variables
 кlayer_regularization_losses
Kregularization_losses
лnon_trainable_variables
мlayers
 
fd
VARIABLE_VALUEbatch_normalization_3/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_3/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_3/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_3/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

N0
O1

N0
O1
P2
Q3
 
▓
нlayer_metrics
оmetrics
Rtrainable_variables
S	variables
 пlayer_regularization_losses
Tregularization_losses
░non_trainable_variables
▒layers
 
 
 
▓
▓layer_metrics
│metrics
Vtrainable_variables
W	variables
 ┤layer_regularization_losses
Xregularization_losses
╡non_trainable_variables
╢layers
 
 
 
▓
╖layer_metrics
╕metrics
Ztrainable_variables
[	variables
 ╣layer_regularization_losses
\regularization_losses
║non_trainable_variables
╗layers
 
 
 
▓
╝layer_metrics
╜metrics
^trainable_variables
_	variables
 ╛layer_regularization_losses
`regularization_losses
┐non_trainable_variables
└layers
 
 
 
▓
┴layer_metrics
┬metrics
btrainable_variables
c	variables
 ├layer_regularization_losses
dregularization_losses
─non_trainable_variables
┼layers
YW
VARIABLE_VALUElayer1/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElayer1/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

f0
g1

f0
g1
 
▓
╞layer_metrics
╟metrics
htrainable_variables
i	variables
 ╚layer_regularization_losses
jregularization_losses
╔non_trainable_variables
╩layers
 
 
 
▓
╦layer_metrics
╠metrics
ltrainable_variables
m	variables
 ═layer_regularization_losses
nregularization_losses
╬non_trainable_variables
╧layers
YW
VARIABLE_VALUElayer2/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElayer2/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

p0
q1

p0
q1
 
▓
╨layer_metrics
╤metrics
rtrainable_variables
s	variables
 ╥layer_regularization_losses
tregularization_losses
╙non_trainable_variables
╘layers
 
 
 
▓
╒layer_metrics
╓metrics
vtrainable_variables
w	variables
 ╫layer_regularization_losses
xregularization_losses
╪non_trainable_variables
┘layers
YW
VARIABLE_VALUEdense/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUE
dense/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

z0
{1

z0
{1
 
▓
┌layer_metrics
█metrics
|trainable_variables
}	variables
 ▄layer_regularization_losses
~regularization_losses
▌non_trainable_variables
▐layers
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 

▀0
 
8
#0
$1
22
33
A4
B5
P6
Q7
О
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
 
 
 
 
 
 
 
 

#0
$1
 
 
 
 
 
 
 
 
 

20
31
 
 
 
 
 
 
 
 
 

A0
B1
 
 
 
 
 
 
 
 
 

P0
Q1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

рtotal

сcount
т	variables
у	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

р0
с1

т	variables
|z
VARIABLE_VALUEAdam/conv3d/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv3d/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE Adam/batch_normalization/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЖГ
VARIABLE_VALUEAdam/batch_normalization/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv3d_1/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_1/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE"Adam/batch_normalization_1/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE!Adam/batch_normalization_1/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv3d_2/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_2/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE"Adam/batch_normalization_2/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE!Adam/batch_normalization_2/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv3d_3/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_3/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE"Adam/batch_normalization_3/gamma/mQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE!Adam/batch_normalization_3/beta/mPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/layer1/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/layer1/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/layer2/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/layer2/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv3d/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE Adam/batch_normalization/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЖГ
VARIABLE_VALUEAdam/batch_normalization/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv3d_1/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_1/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE"Adam/batch_normalization_1/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE!Adam/batch_normalization_1/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv3d_2/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_2/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE"Adam/batch_normalization_2/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE!Adam/batch_normalization_2/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv3d_3/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_3/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE"Adam/batch_normalization_3/gamma/vQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE!Adam/batch_normalization_3/beta/vPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/layer1/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/layer1/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/layer2/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/layer2/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Ф
serving_default_input_1Placeholder*4
_output_shapes"
 :         н*
dtype0*)
shape :         н
z
serving_default_input_2Placeholder*'
_output_shapes
:         *
dtype0*
shape:         
╜
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2conv3d/kernelconv3d/bias#batch_normalization/moving_variancebatch_normalization/gammabatch_normalization/moving_meanbatch_normalization/betaconv3d_1/kernelconv3d_1/bias%batch_normalization_1/moving_variancebatch_normalization_1/gamma!batch_normalization_1/moving_meanbatch_normalization_1/betaconv3d_2/kernelconv3d_2/bias%batch_normalization_2/moving_variancebatch_normalization_2/gamma!batch_normalization_2/moving_meanbatch_normalization_2/betaconv3d_3/kernelconv3d_3/bias%batch_normalization_3/moving_variancebatch_normalization_3/gamma!batch_normalization_3/moving_meanbatch_normalization_3/betalayer1/kernellayer1/biaslayer2/kernellayer2/biasdense/kernel
dense/bias*+
Tin$
"2 *
Tout
2*'
_output_shapes
:         *@
_read_only_resource_inputs"
 	
*-
config_proto

GPU

CPU2*0J 8*/
f*R(
&__inference_signature_wrapper_11218702
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
╪
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv3d/kernel/Read/ReadVariableOpconv3d/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp#conv3d_1/kernel/Read/ReadVariableOp!conv3d_1/bias/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp#conv3d_2/kernel/Read/ReadVariableOp!conv3d_2/bias/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOp#conv3d_3/kernel/Read/ReadVariableOp!conv3d_3/bias/Read/ReadVariableOp/batch_normalization_3/gamma/Read/ReadVariableOp.batch_normalization_3/beta/Read/ReadVariableOp5batch_normalization_3/moving_mean/Read/ReadVariableOp9batch_normalization_3/moving_variance/Read/ReadVariableOp!layer1/kernel/Read/ReadVariableOplayer1/bias/Read/ReadVariableOp!layer2/kernel/Read/ReadVariableOplayer2/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp(Adam/conv3d/kernel/m/Read/ReadVariableOp&Adam/conv3d/bias/m/Read/ReadVariableOp4Adam/batch_normalization/gamma/m/Read/ReadVariableOp3Adam/batch_normalization/beta/m/Read/ReadVariableOp*Adam/conv3d_1/kernel/m/Read/ReadVariableOp(Adam/conv3d_1/bias/m/Read/ReadVariableOp6Adam/batch_normalization_1/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_1/beta/m/Read/ReadVariableOp*Adam/conv3d_2/kernel/m/Read/ReadVariableOp(Adam/conv3d_2/bias/m/Read/ReadVariableOp6Adam/batch_normalization_2/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_2/beta/m/Read/ReadVariableOp*Adam/conv3d_3/kernel/m/Read/ReadVariableOp(Adam/conv3d_3/bias/m/Read/ReadVariableOp6Adam/batch_normalization_3/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_3/beta/m/Read/ReadVariableOp(Adam/layer1/kernel/m/Read/ReadVariableOp&Adam/layer1/bias/m/Read/ReadVariableOp(Adam/layer2/kernel/m/Read/ReadVariableOp&Adam/layer2/bias/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp(Adam/conv3d/kernel/v/Read/ReadVariableOp&Adam/conv3d/bias/v/Read/ReadVariableOp4Adam/batch_normalization/gamma/v/Read/ReadVariableOp3Adam/batch_normalization/beta/v/Read/ReadVariableOp*Adam/conv3d_1/kernel/v/Read/ReadVariableOp(Adam/conv3d_1/bias/v/Read/ReadVariableOp6Adam/batch_normalization_1/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_1/beta/v/Read/ReadVariableOp*Adam/conv3d_2/kernel/v/Read/ReadVariableOp(Adam/conv3d_2/bias/v/Read/ReadVariableOp6Adam/batch_normalization_2/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_2/beta/v/Read/ReadVariableOp*Adam/conv3d_3/kernel/v/Read/ReadVariableOp(Adam/conv3d_3/bias/v/Read/ReadVariableOp6Adam/batch_normalization_3/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_3/beta/v/Read/ReadVariableOp(Adam/layer1/kernel/v/Read/ReadVariableOp&Adam/layer1/bias/v/Read/ReadVariableOp(Adam/layer2/kernel/v/Read/ReadVariableOp&Adam/layer2/bias/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOpConst*^
TinW
U2S	*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8**
f%R#
!__inference__traced_save_11220405
 
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv3d/kernelconv3d/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv3d_1/kernelconv3d_1/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv3d_2/kernelconv3d_2/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceconv3d_3/kernelconv3d_3/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_variancelayer1/kernellayer1/biaslayer2/kernellayer2/biasdense/kernel
dense/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/conv3d/kernel/mAdam/conv3d/bias/m Adam/batch_normalization/gamma/mAdam/batch_normalization/beta/mAdam/conv3d_1/kernel/mAdam/conv3d_1/bias/m"Adam/batch_normalization_1/gamma/m!Adam/batch_normalization_1/beta/mAdam/conv3d_2/kernel/mAdam/conv3d_2/bias/m"Adam/batch_normalization_2/gamma/m!Adam/batch_normalization_2/beta/mAdam/conv3d_3/kernel/mAdam/conv3d_3/bias/m"Adam/batch_normalization_3/gamma/m!Adam/batch_normalization_3/beta/mAdam/layer1/kernel/mAdam/layer1/bias/mAdam/layer2/kernel/mAdam/layer2/bias/mAdam/dense/kernel/mAdam/dense/bias/mAdam/conv3d/kernel/vAdam/conv3d/bias/v Adam/batch_normalization/gamma/vAdam/batch_normalization/beta/vAdam/conv3d_1/kernel/vAdam/conv3d_1/bias/v"Adam/batch_normalization_1/gamma/v!Adam/batch_normalization_1/beta/vAdam/conv3d_2/kernel/vAdam/conv3d_2/bias/v"Adam/batch_normalization_2/gamma/v!Adam/batch_normalization_2/beta/vAdam/conv3d_3/kernel/vAdam/conv3d_3/bias/v"Adam/batch_normalization_3/gamma/v!Adam/batch_normalization_3/beta/vAdam/layer1/kernel/vAdam/layer1/bias/vAdam/layer2/kernel/vAdam/layer2/bias/vAdam/dense/kernel/vAdam/dense/bias/v*]
TinV
T2R*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*-
f(R&
$__inference__traced_restore_11220660╧╡
Й
e
,__inference_dropout_1_layer_call_fn_11220062

inputs
identityИвStatefulPartitionedCall┐
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:         ╚* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_1_layer_call_and_return_conditional_losses_112180002
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ╚2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ╚22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ╚
 
_user_specified_nameinputs
¤
~
)__inference_layer1_layer_call_fn_11220040

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCall╓
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*(
_output_shapes
:         ╚*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_layer1_layer_call_and_return_conditional_losses_112179722
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ╚2

Identity"
identityIdentity:output:0*/
_input_shapes
:         Г
::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         Г

 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
В
┬
(__inference_model_layer_call_fn_11218482
input_1
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28
identityИвStatefulPartitionedCall▐
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28*+
Tin$
"2 *
Tout
2*'
_output_shapes
:         *@
_read_only_resource_inputs"
 	
*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_112184192
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*└
_input_shapesо
л:         н:         ::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
4
_output_shapes"
 :         н
!
_user_specified_name	input_1:PL
'
_output_shapes
:         
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
тм
╫
#__inference__wrapped_model_11216852
input_1
input_2/
+model_conv3d_conv3d_readvariableop_resource0
,model_conv3d_biasadd_readvariableop_resource?
;model_batch_normalization_batchnorm_readvariableop_resourceC
?model_batch_normalization_batchnorm_mul_readvariableop_resourceA
=model_batch_normalization_batchnorm_readvariableop_1_resourceA
=model_batch_normalization_batchnorm_readvariableop_2_resource1
-model_conv3d_1_conv3d_readvariableop_resource2
.model_conv3d_1_biasadd_readvariableop_resourceA
=model_batch_normalization_1_batchnorm_readvariableop_resourceE
Amodel_batch_normalization_1_batchnorm_mul_readvariableop_resourceC
?model_batch_normalization_1_batchnorm_readvariableop_1_resourceC
?model_batch_normalization_1_batchnorm_readvariableop_2_resource1
-model_conv3d_2_conv3d_readvariableop_resource2
.model_conv3d_2_biasadd_readvariableop_resourceA
=model_batch_normalization_2_batchnorm_readvariableop_resourceE
Amodel_batch_normalization_2_batchnorm_mul_readvariableop_resourceC
?model_batch_normalization_2_batchnorm_readvariableop_1_resourceC
?model_batch_normalization_2_batchnorm_readvariableop_2_resource1
-model_conv3d_3_conv3d_readvariableop_resource2
.model_conv3d_3_biasadd_readvariableop_resourceA
=model_batch_normalization_3_batchnorm_readvariableop_resourceE
Amodel_batch_normalization_3_batchnorm_mul_readvariableop_resourceC
?model_batch_normalization_3_batchnorm_readvariableop_1_resourceC
?model_batch_normalization_3_batchnorm_readvariableop_2_resource/
+model_layer1_matmul_readvariableop_resource0
,model_layer1_biasadd_readvariableop_resource/
+model_layer2_matmul_readvariableop_resource0
,model_layer2_biasadd_readvariableop_resource.
*model_dense_matmul_readvariableop_resource/
+model_dense_biasadd_readvariableop_resource
identityИ┴
"model/conv3d/Conv3D/ReadVariableOpReadVariableOp+model_conv3d_conv3d_readvariableop_resource*+
_output_shapes
:н*
dtype02$
"model/conv3d/Conv3D/ReadVariableOp╤
model/conv3d/Conv3DConv3Dinput_1*model/conv3d/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         *
paddingVALID*
strides	
2
model/conv3d/Conv3D│
#model/conv3d/BiasAdd/ReadVariableOpReadVariableOp,model_conv3d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#model/conv3d/BiasAdd/ReadVariableOp└
model/conv3d/BiasAddBiasAddmodel/conv3d/Conv3D:output:0+model/conv3d/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         2
model/conv3d/BiasAddр
2model/batch_normalization/batchnorm/ReadVariableOpReadVariableOp;model_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype024
2model/batch_normalization/batchnorm/ReadVariableOpЫ
)model/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2+
)model/batch_normalization/batchnorm/add/yЁ
'model/batch_normalization/batchnorm/addAddV2:model/batch_normalization/batchnorm/ReadVariableOp:value:02model/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:2)
'model/batch_normalization/batchnorm/add▒
)model/batch_normalization/batchnorm/RsqrtRsqrt+model/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:2+
)model/batch_normalization/batchnorm/Rsqrtь
6model/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp?model_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype028
6model/batch_normalization/batchnorm/mul/ReadVariableOpэ
'model/batch_normalization/batchnorm/mulMul-model/batch_normalization/batchnorm/Rsqrt:y:0>model/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2)
'model/batch_normalization/batchnorm/mulч
)model/batch_normalization/batchnorm/mul_1Mulmodel/conv3d/BiasAdd:output:0+model/batch_normalization/batchnorm/mul:z:0*
T0*3
_output_shapes!
:         2+
)model/batch_normalization/batchnorm/mul_1ц
4model/batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp=model_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype026
4model/batch_normalization/batchnorm/ReadVariableOp_1э
)model/batch_normalization/batchnorm/mul_2Mul<model/batch_normalization/batchnorm/ReadVariableOp_1:value:0+model/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:2+
)model/batch_normalization/batchnorm/mul_2ц
4model/batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp=model_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype026
4model/batch_normalization/batchnorm/ReadVariableOp_2ы
'model/batch_normalization/batchnorm/subSub<model/batch_normalization/batchnorm/ReadVariableOp_2:value:0-model/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2)
'model/batch_normalization/batchnorm/sub∙
)model/batch_normalization/batchnorm/add_1AddV2-model/batch_normalization/batchnorm/mul_1:z:0+model/batch_normalization/batchnorm/sub:z:0*
T0*3
_output_shapes!
:         2+
)model/batch_normalization/batchnorm/add_1╞
$model/conv3d_1/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_1_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02&
$model/conv3d_1/Conv3D/ReadVariableOp¤
model/conv3d_1/Conv3DConv3D-model/batch_normalization/batchnorm/add_1:z:0,model/conv3d_1/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         *
paddingVALID*
strides	
2
model/conv3d_1/Conv3D╣
%model/conv3d_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%model/conv3d_1/BiasAdd/ReadVariableOp╚
model/conv3d_1/BiasAddBiasAddmodel/conv3d_1/Conv3D:output:0-model/conv3d_1/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         2
model/conv3d_1/BiasAddО
model/conv3d_1/EluElumodel/conv3d_1/BiasAdd:output:0*
T0*3
_output_shapes!
:         2
model/conv3d_1/Eluц
4model/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp=model_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype026
4model/batch_normalization_1/batchnorm/ReadVariableOpЯ
+model/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2-
+model/batch_normalization_1/batchnorm/add/y°
)model/batch_normalization_1/batchnorm/addAddV2<model/batch_normalization_1/batchnorm/ReadVariableOp:value:04model/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:2+
)model/batch_normalization_1/batchnorm/add╖
+model/batch_normalization_1/batchnorm/RsqrtRsqrt-model/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:2-
+model/batch_normalization_1/batchnorm/RsqrtЄ
8model/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpAmodel_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02:
8model/batch_normalization_1/batchnorm/mul/ReadVariableOpї
)model/batch_normalization_1/batchnorm/mulMul/model/batch_normalization_1/batchnorm/Rsqrt:y:0@model/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2+
)model/batch_normalization_1/batchnorm/mulЁ
+model/batch_normalization_1/batchnorm/mul_1Mul model/conv3d_1/Elu:activations:0-model/batch_normalization_1/batchnorm/mul:z:0*
T0*3
_output_shapes!
:         2-
+model/batch_normalization_1/batchnorm/mul_1ь
6model/batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOp?model_batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype028
6model/batch_normalization_1/batchnorm/ReadVariableOp_1ї
+model/batch_normalization_1/batchnorm/mul_2Mul>model/batch_normalization_1/batchnorm/ReadVariableOp_1:value:0-model/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:2-
+model/batch_normalization_1/batchnorm/mul_2ь
6model/batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOp?model_batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype028
6model/batch_normalization_1/batchnorm/ReadVariableOp_2є
)model/batch_normalization_1/batchnorm/subSub>model/batch_normalization_1/batchnorm/ReadVariableOp_2:value:0/model/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2+
)model/batch_normalization_1/batchnorm/subБ
+model/batch_normalization_1/batchnorm/add_1AddV2/model/batch_normalization_1/batchnorm/mul_1:z:0-model/batch_normalization_1/batchnorm/sub:z:0*
T0*3
_output_shapes!
:         2-
+model/batch_normalization_1/batchnorm/add_1╞
$model/conv3d_2/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_2_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02&
$model/conv3d_2/Conv3D/ReadVariableOp 
model/conv3d_2/Conv3DConv3D/model/batch_normalization_1/batchnorm/add_1:z:0,model/conv3d_2/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         *
paddingVALID*
strides	
2
model/conv3d_2/Conv3D╣
%model/conv3d_2/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%model/conv3d_2/BiasAdd/ReadVariableOp╚
model/conv3d_2/BiasAddBiasAddmodel/conv3d_2/Conv3D:output:0-model/conv3d_2/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         2
model/conv3d_2/BiasAddО
model/conv3d_2/EluElumodel/conv3d_2/BiasAdd:output:0*
T0*3
_output_shapes!
:         2
model/conv3d_2/Eluц
4model/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp=model_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype026
4model/batch_normalization_2/batchnorm/ReadVariableOpЯ
+model/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2-
+model/batch_normalization_2/batchnorm/add/y°
)model/batch_normalization_2/batchnorm/addAddV2<model/batch_normalization_2/batchnorm/ReadVariableOp:value:04model/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:2+
)model/batch_normalization_2/batchnorm/add╖
+model/batch_normalization_2/batchnorm/RsqrtRsqrt-model/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:2-
+model/batch_normalization_2/batchnorm/RsqrtЄ
8model/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpAmodel_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02:
8model/batch_normalization_2/batchnorm/mul/ReadVariableOpї
)model/batch_normalization_2/batchnorm/mulMul/model/batch_normalization_2/batchnorm/Rsqrt:y:0@model/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2+
)model/batch_normalization_2/batchnorm/mulЁ
+model/batch_normalization_2/batchnorm/mul_1Mul model/conv3d_2/Elu:activations:0-model/batch_normalization_2/batchnorm/mul:z:0*
T0*3
_output_shapes!
:         2-
+model/batch_normalization_2/batchnorm/mul_1ь
6model/batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOp?model_batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype028
6model/batch_normalization_2/batchnorm/ReadVariableOp_1ї
+model/batch_normalization_2/batchnorm/mul_2Mul>model/batch_normalization_2/batchnorm/ReadVariableOp_1:value:0-model/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:2-
+model/batch_normalization_2/batchnorm/mul_2ь
6model/batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOp?model_batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype028
6model/batch_normalization_2/batchnorm/ReadVariableOp_2є
)model/batch_normalization_2/batchnorm/subSub>model/batch_normalization_2/batchnorm/ReadVariableOp_2:value:0/model/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2+
)model/batch_normalization_2/batchnorm/subБ
+model/batch_normalization_2/batchnorm/add_1AddV2/model/batch_normalization_2/batchnorm/mul_1:z:0-model/batch_normalization_2/batchnorm/sub:z:0*
T0*3
_output_shapes!
:         2-
+model/batch_normalization_2/batchnorm/add_1╞
$model/conv3d_3/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_3_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02&
$model/conv3d_3/Conv3D/ReadVariableOp 
model/conv3d_3/Conv3DConv3D/model/batch_normalization_2/batchnorm/add_1:z:0,model/conv3d_3/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         *
paddingVALID*
strides	
2
model/conv3d_3/Conv3D╣
%model/conv3d_3/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%model/conv3d_3/BiasAdd/ReadVariableOp╚
model/conv3d_3/BiasAddBiasAddmodel/conv3d_3/Conv3D:output:0-model/conv3d_3/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         2
model/conv3d_3/BiasAddО
model/conv3d_3/EluElumodel/conv3d_3/BiasAdd:output:0*
T0*3
_output_shapes!
:         2
model/conv3d_3/Eluц
4model/batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp=model_batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype026
4model/batch_normalization_3/batchnorm/ReadVariableOpЯ
+model/batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2-
+model/batch_normalization_3/batchnorm/add/y°
)model/batch_normalization_3/batchnorm/addAddV2<model/batch_normalization_3/batchnorm/ReadVariableOp:value:04model/batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:2+
)model/batch_normalization_3/batchnorm/add╖
+model/batch_normalization_3/batchnorm/RsqrtRsqrt-model/batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:2-
+model/batch_normalization_3/batchnorm/RsqrtЄ
8model/batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOpAmodel_batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02:
8model/batch_normalization_3/batchnorm/mul/ReadVariableOpї
)model/batch_normalization_3/batchnorm/mulMul/model/batch_normalization_3/batchnorm/Rsqrt:y:0@model/batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2+
)model/batch_normalization_3/batchnorm/mulЁ
+model/batch_normalization_3/batchnorm/mul_1Mul model/conv3d_3/Elu:activations:0-model/batch_normalization_3/batchnorm/mul:z:0*
T0*3
_output_shapes!
:         2-
+model/batch_normalization_3/batchnorm/mul_1ь
6model/batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOp?model_batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype028
6model/batch_normalization_3/batchnorm/ReadVariableOp_1ї
+model/batch_normalization_3/batchnorm/mul_2Mul>model/batch_normalization_3/batchnorm/ReadVariableOp_1:value:0-model/batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:2-
+model/batch_normalization_3/batchnorm/mul_2ь
6model/batch_normalization_3/batchnorm/ReadVariableOp_2ReadVariableOp?model_batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype028
6model/batch_normalization_3/batchnorm/ReadVariableOp_2є
)model/batch_normalization_3/batchnorm/subSub>model/batch_normalization_3/batchnorm/ReadVariableOp_2:value:0/model/batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2+
)model/batch_normalization_3/batchnorm/subБ
+model/batch_normalization_3/batchnorm/add_1AddV2/model/batch_normalization_3/batchnorm/mul_1:z:0-model/batch_normalization_3/batchnorm/sub:z:0*
T0*3
_output_shapes!
:         2-
+model/batch_normalization_3/batchnorm/add_1А
!model/average_pooling3d/AvgPool3D	AvgPool3D/model/batch_normalization_3/batchnorm/add_1:z:0*
T0*3
_output_shapes!
:         *
ksize	
*
paddingVALID*
strides	
2#
!model/average_pooling3d/AvgPool3D{
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
model/flatten/Const╢
model/flatten/ReshapeReshape*model/average_pooling3d/AvgPool3D:output:0model/flatten/Const:output:0*
T0*(
_output_shapes
:         А
2
model/flatten/ReshapeП
model/dropout/IdentityIdentitymodel/flatten/Reshape:output:0*
T0*(
_output_shapes
:         А
2
model/dropout/IdentityА
model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
model/concatenate/concat/axis╬
model/concatenate/concatConcatV2model/dropout/Identity:output:0input_2&model/concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:         Г
2
model/concatenate/concat╢
"model/layer1/MatMul/ReadVariableOpReadVariableOp+model_layer1_matmul_readvariableop_resource* 
_output_shapes
:
Г
╚*
dtype02$
"model/layer1/MatMul/ReadVariableOp╢
model/layer1/MatMulMatMul!model/concatenate/concat:output:0*model/layer1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
model/layer1/MatMul┤
#model/layer1/BiasAdd/ReadVariableOpReadVariableOp,model_layer1_biasadd_readvariableop_resource*
_output_shapes	
:╚*
dtype02%
#model/layer1/BiasAdd/ReadVariableOp╢
model/layer1/BiasAddBiasAddmodel/layer1/MatMul:product:0+model/layer1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
model/layer1/BiasAdd}
model/layer1/EluElumodel/layer1/BiasAdd:output:0*
T0*(
_output_shapes
:         ╚2
model/layer1/EluУ
model/dropout_1/IdentityIdentitymodel/layer1/Elu:activations:0*
T0*(
_output_shapes
:         ╚2
model/dropout_1/Identity╡
"model/layer2/MatMul/ReadVariableOpReadVariableOp+model_layer2_matmul_readvariableop_resource*
_output_shapes
:	╚*
dtype02$
"model/layer2/MatMul/ReadVariableOp╡
model/layer2/MatMulMatMul!model/dropout_1/Identity:output:0*model/layer2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
model/layer2/MatMul│
#model/layer2/BiasAdd/ReadVariableOpReadVariableOp,model_layer2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#model/layer2/BiasAdd/ReadVariableOp╡
model/layer2/BiasAddBiasAddmodel/layer2/MatMul:product:0+model/layer2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
model/layer2/BiasAdd|
model/layer2/EluElumodel/layer2/BiasAdd:output:0*
T0*'
_output_shapes
:         2
model/layer2/EluТ
model/dropout_2/IdentityIdentitymodel/layer2/Elu:activations:0*
T0*'
_output_shapes
:         2
model/dropout_2/Identity▒
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02#
!model/dense/MatMul/ReadVariableOp▓
model/dense/MatMulMatMul!model/dropout_2/Identity:output:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
model/dense/MatMul░
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"model/dense/BiasAdd/ReadVariableOp▒
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
model/dense/BiasAddЕ
model/dense/SigmoidSigmoidmodel/dense/BiasAdd:output:0*
T0*'
_output_shapes
:         2
model/dense/Sigmoidk
IdentityIdentitymodel/dense/Sigmoid:y:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*└
_input_shapesо
л:         н:         :::::::::::::::::::::::::::::::] Y
4
_output_shapes"
 :         н
!
_user_specified_name	input_1:PL
'
_output_shapes
:         
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
╟+
╨
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_11219641

inputs
assignmovingavg_11219616
assignmovingavg_1_11219622)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpЩ
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indicesЫ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/meanИ
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
:2
moments/StopGradient░
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*3
_output_shapes!
:         2
moments/SquaredDifferenceб
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices╛
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/varianceГ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/SqueezeЛ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1а
AssignMovingAvg/decayConst*+
_class!
loc:@AssignMovingAvg/11219616*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayХ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_11219616*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp┼
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*+
_class!
loc:@AssignMovingAvg/11219616*
_output_shapes
:2
AssignMovingAvg/sub╝
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg/11219616*
_output_shapes
:2
AssignMovingAvg/mulЕ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_11219616AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/11219616*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpж
AssignMovingAvg_1/decayConst*-
_class#
!loc:@AssignMovingAvg_1/11219622*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayЫ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_11219622*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╧
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/11219622*
_output_shapes
:2
AssignMovingAvg_1/sub╞
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/11219622*
_output_shapes
:2
AssignMovingAvg_1/mulС
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_11219622AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/11219622*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulВ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:         2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subС
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:         2
batchnorm/add_1┴
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*3
_output_shapes!
:         2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:         ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:[ W
3
_output_shapes!
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
А
─
(__inference_model_layer_call_fn_11219103
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28
identityИвStatefulPartitionedCall╪
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28*+
Tin$
"2 *
Tout
2*'
_output_shapes
:         *8
_read_only_resource_inputs
	*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_112182712
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*└
_input_shapesо
л:         н:         ::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :         н
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
к	
й
6__inference_batch_normalization_layer_call_fn_11219274

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallб
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*N
_output_shapes<
::8                                    *$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_112169692
StatefulPartitionedCall╡
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8                                    2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:8                                    ::::22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8                                    
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
щ,
╨
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_11219723

inputs
assignmovingavg_11219698
assignmovingavg_1_11219704)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpЩ
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indicesЫ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/meanИ
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
:2
moments/StopGradient╦
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*N
_output_shapes<
::8                                    2
moments/SquaredDifferenceб
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices╛
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/varianceГ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/SqueezeЛ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1а
AssignMovingAvg/decayConst*+
_class!
loc:@AssignMovingAvg/11219698*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayХ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_11219698*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp┼
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*+
_class!
loc:@AssignMovingAvg/11219698*
_output_shapes
:2
AssignMovingAvg/sub╝
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg/11219698*
_output_shapes
:2
AssignMovingAvg/mulЕ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_11219698AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/11219698*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpж
AssignMovingAvg_1/decayConst*-
_class#
!loc:@AssignMovingAvg_1/11219704*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayЫ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_11219704*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╧
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/11219704*
_output_shapes
:2
AssignMovingAvg_1/sub╞
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/11219704*
_output_shapes
:2
AssignMovingAvg_1/mulС
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_11219704AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/11219704*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulЭ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8                                    2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subм
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8                                    2
batchnorm/add_1▄
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*N
_output_shapes<
::8                                    2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:8                                    ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:v r
N
_output_shapes<
::8                                    
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
┬
u
I__inference_concatenate_layer_call_and_return_conditional_losses_11220014
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisВ
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:         Г
2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:         Г
2

Identity"
identityIdentity:output:0*:
_input_shapes)
':         А
:         :R N
(
_output_shapes
:         А

"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/1
┬
Ц
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_11217488

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИТ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulЭ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8                                    2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subм
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8                                    2
batchnorm/add_1О
IdentityIdentitybatchnorm/add_1:z:0*
T0*N
_output_shapes<
::8                                    2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:8                                    :::::v r
N
_output_shapes<
::8                                    
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
╩
e
G__inference_dropout_2_layer_call_and_return_conditional_losses_11218062

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:         2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
┼+
╬
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_11219323

inputs
assignmovingavg_11219298
assignmovingavg_1_11219304)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpЩ
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indicesЫ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/meanИ
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
:2
moments/StopGradient░
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*3
_output_shapes!
:         2
moments/SquaredDifferenceб
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices╛
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/varianceГ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/SqueezeЛ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1а
AssignMovingAvg/decayConst*+
_class!
loc:@AssignMovingAvg/11219298*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayХ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_11219298*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp┼
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*+
_class!
loc:@AssignMovingAvg/11219298*
_output_shapes
:2
AssignMovingAvg/sub╝
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg/11219298*
_output_shapes
:2
AssignMovingAvg/mulЕ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_11219298AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/11219298*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpж
AssignMovingAvg_1/decayConst*-
_class#
!loc:@AssignMovingAvg_1/11219304*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayЫ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_11219304*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╧
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/11219304*
_output_shapes
:2
AssignMovingAvg_1/sub╞
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/11219304*
_output_shapes
:2
AssignMovingAvg_1/mulС
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_11219304AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/11219304*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulВ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:         2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subС
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:         2
batchnorm/add_1┴
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*3
_output_shapes!
:         2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:         ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:[ W
3
_output_shapes!
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
┬
Ц
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_11217326

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИТ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulЭ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8                                    2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subм
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8                                    2
batchnorm/add_1О
IdentityIdentitybatchnorm/add_1:z:0*
T0*N
_output_shapes<
::8                                    2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:8                                    :::::v r
N
_output_shapes<
::8                                    
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
╬
e
G__inference_dropout_1_layer_call_and_return_conditional_losses_11218005

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         ╚2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         ╚2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:         ╚:P L
(
_output_shapes
:         ╚
 
_user_specified_nameinputs
∙
H
,__inference_dropout_2_layer_call_fn_11220114

inputs
identityж
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_2_layer_call_and_return_conditional_losses_112180622
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
┬
Ц
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_11219861

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИТ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulЭ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8                                    2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subм
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8                                    2
batchnorm/add_1О
IdentityIdentitybatchnorm/add_1:z:0*
T0*N
_output_shapes<
::8                                    2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:8                                    :::::v r
N
_output_shapes<
::8                                    
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ОY
В
C__inference_model_layer_call_and_return_conditional_losses_11218185
input_1
input_2
conv3d_11218107
conv3d_11218109 
batch_normalization_11218112 
batch_normalization_11218114 
batch_normalization_11218116 
batch_normalization_11218118
conv3d_1_11218121
conv3d_1_11218123"
batch_normalization_1_11218126"
batch_normalization_1_11218128"
batch_normalization_1_11218130"
batch_normalization_1_11218132
conv3d_2_11218135
conv3d_2_11218137"
batch_normalization_2_11218140"
batch_normalization_2_11218142"
batch_normalization_2_11218144"
batch_normalization_2_11218146
conv3d_3_11218149
conv3d_3_11218151"
batch_normalization_3_11218154"
batch_normalization_3_11218156"
batch_normalization_3_11218158"
batch_normalization_3_11218160
layer1_11218167
layer1_11218169
layer2_11218173
layer2_11218175
dense_11218179
dense_11218181
identityИв+batch_normalization/StatefulPartitionedCallв-batch_normalization_1/StatefulPartitionedCallв-batch_normalization_2/StatefulPartitionedCallв-batch_normalization_3/StatefulPartitionedCallвconv3d/StatefulPartitionedCallв conv3d_1/StatefulPartitionedCallв conv3d_2/StatefulPartitionedCallв conv3d_3/StatefulPartitionedCallвdense/StatefulPartitionedCallвlayer1/StatefulPartitionedCallвlayer2/StatefulPartitionedCall■
conv3d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv3d_11218107conv3d_11218109*
Tin
2*
Tout
2*3
_output_shapes!
:         *$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_conv3d_layer_call_and_return_conditional_losses_112168632 
conv3d/StatefulPartitionedCallЯ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv3d/StatefulPartitionedCall:output:0batch_normalization_11218112batch_normalization_11218114batch_normalization_11218116batch_normalization_11218118*
Tin	
2*
Tout
2*3
_output_shapes!
:         *&
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_112175762-
+batch_normalization/StatefulPartitionedCall╡
 conv3d_1/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0conv3d_1_11218121conv3d_1_11218123*
Tin
2*
Tout
2*3
_output_shapes!
:         *$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_conv3d_1_layer_call_and_return_conditional_losses_112170252"
 conv3d_1/StatefulPartitionedCallп
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv3d_1/StatefulPartitionedCall:output:0batch_normalization_1_11218126batch_normalization_1_11218128batch_normalization_1_11218130batch_normalization_1_11218132*
Tin	
2*
Tout
2*3
_output_shapes!
:         *&
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_112176722/
-batch_normalization_1/StatefulPartitionedCall╖
 conv3d_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0conv3d_2_11218135conv3d_2_11218137*
Tin
2*
Tout
2*3
_output_shapes!
:         *$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_conv3d_2_layer_call_and_return_conditional_losses_112171872"
 conv3d_2/StatefulPartitionedCallп
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv3d_2/StatefulPartitionedCall:output:0batch_normalization_2_11218140batch_normalization_2_11218142batch_normalization_2_11218144batch_normalization_2_11218146*
Tin	
2*
Tout
2*3
_output_shapes!
:         *&
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_112177682/
-batch_normalization_2/StatefulPartitionedCall╖
 conv3d_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0conv3d_3_11218149conv3d_3_11218151*
Tin
2*
Tout
2*3
_output_shapes!
:         *$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_conv3d_3_layer_call_and_return_conditional_losses_112173492"
 conv3d_3/StatefulPartitionedCallп
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv3d_3/StatefulPartitionedCall:output:0batch_normalization_3_11218154batch_normalization_3_11218156batch_normalization_3_11218158batch_normalization_3_11218160*
Tin	
2*
Tout
2*3
_output_shapes!
:         *&
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*\
fWRU
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_112178642/
-batch_normalization_3/StatefulPartitionedCallО
!average_pooling3d/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:         * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_average_pooling3d_layer_call_and_return_conditional_losses_112175052#
!average_pooling3d/PartitionedCall┘
flatten/PartitionedCallPartitionedCall*average_pooling3d/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:         А
* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_112179072
flatten/PartitionedCall╧
dropout/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:         А
* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_112179322
dropout/PartitionedCallх
concatenate/PartitionedCallPartitionedCall dropout/PartitionedCall:output:0input_2*
Tin
2*
Tout
2*(
_output_shapes
:         Г
* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_concatenate_layer_call_and_return_conditional_losses_112179522
concatenate/PartitionedCallР
layer1/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0layer1_11218167layer1_11218169*
Tin
2*
Tout
2*(
_output_shapes
:         ╚*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_layer1_layer_call_and_return_conditional_losses_112179722 
layer1/StatefulPartitionedCall▄
dropout_1/PartitionedCallPartitionedCall'layer1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:         ╚* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_1_layer_call_and_return_conditional_losses_112180052
dropout_1/PartitionedCallН
layer2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0layer2_11218173layer2_11218175*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_layer2_layer_call_and_return_conditional_losses_112180292 
layer2/StatefulPartitionedCall█
dropout_2/PartitionedCallPartitionedCall'layer2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_2_layer_call_and_return_conditional_losses_112180622
dropout_2/PartitionedCallИ
dense/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_11218179dense_11218181*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_112180862
dense/StatefulPartitionedCallд
IdentityIdentity&dense/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall^conv3d/StatefulPartitionedCall!^conv3d_1/StatefulPartitionedCall!^conv3d_2/StatefulPartitionedCall!^conv3d_3/StatefulPartitionedCall^dense/StatefulPartitionedCall^layer1/StatefulPartitionedCall^layer2/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*└
_input_shapesо
л:         н:         ::::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2@
conv3d/StatefulPartitionedCallconv3d/StatefulPartitionedCall2D
 conv3d_1/StatefulPartitionedCall conv3d_1/StatefulPartitionedCall2D
 conv3d_2/StatefulPartitionedCall conv3d_2/StatefulPartitionedCall2D
 conv3d_3/StatefulPartitionedCall conv3d_3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2@
layer2/StatefulPartitionedCalllayer2/StatefulPartitionedCall:] Y
4
_output_shapes"
 :         н
!
_user_specified_name	input_1:PL
'
_output_shapes
:         
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
о	
л
8__inference_batch_normalization_3_layer_call_fn_11219874

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallг
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*N
_output_shapes<
::8                                    *$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*\
fWRU
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_112174552
StatefulPartitionedCall╡
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8                                    2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:8                                    ::::22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8                                    
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
┼+
╬
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_11217556

inputs
assignmovingavg_11217531
assignmovingavg_1_11217537)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpЩ
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indicesЫ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/meanИ
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
:2
moments/StopGradient░
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*3
_output_shapes!
:         2
moments/SquaredDifferenceб
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices╛
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/varianceГ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/SqueezeЛ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1а
AssignMovingAvg/decayConst*+
_class!
loc:@AssignMovingAvg/11217531*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayХ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_11217531*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp┼
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*+
_class!
loc:@AssignMovingAvg/11217531*
_output_shapes
:2
AssignMovingAvg/sub╝
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg/11217531*
_output_shapes
:2
AssignMovingAvg/mulЕ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_11217531AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/11217531*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpж
AssignMovingAvg_1/decayConst*-
_class#
!loc:@AssignMovingAvg_1/11217537*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayЫ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_11217537*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╧
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/11217537*
_output_shapes
:2
AssignMovingAvg_1/sub╞
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/11217537*
_output_shapes
:2
AssignMovingAvg_1/mulС
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_11217537AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/11217537*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulВ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:         2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subС
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:         2
batchnorm/add_1┴
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*3
_output_shapes!
:         2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:         ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:[ W
3
_output_shapes!
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
щ,
╨
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_11217293

inputs
assignmovingavg_11217268
assignmovingavg_1_11217274)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpЩ
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indicesЫ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/meanИ
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
:2
moments/StopGradient╦
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*N
_output_shapes<
::8                                    2
moments/SquaredDifferenceб
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices╛
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/varianceГ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/SqueezeЛ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1а
AssignMovingAvg/decayConst*+
_class!
loc:@AssignMovingAvg/11217268*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayХ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_11217268*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp┼
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*+
_class!
loc:@AssignMovingAvg/11217268*
_output_shapes
:2
AssignMovingAvg/sub╝
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg/11217268*
_output_shapes
:2
AssignMovingAvg/mulЕ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_11217268AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/11217268*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpж
AssignMovingAvg_1/decayConst*-
_class#
!loc:@AssignMovingAvg_1/11217274*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayЫ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_11217274*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╧
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/11217274*
_output_shapes
:2
AssignMovingAvg_1/sub╞
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/11217274*
_output_shapes
:2
AssignMovingAvg_1/mulС
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_11217274AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/11217274*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulЭ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8                                    2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subм
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8                                    2
batchnorm/add_1▄
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*N
_output_shapes<
::8                                    2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:8                                    ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:v r
N
_output_shapes<
::8                                    
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
─
л
8__inference_batch_normalization_2_layer_call_fn_11219687

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallК
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*3
_output_shapes!
:         *&
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_112177682
StatefulPartitionedCallЪ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:         2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:         ::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
щ,
╨
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_11217455

inputs
assignmovingavg_11217430
assignmovingavg_1_11217436)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpЩ
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indicesЫ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/meanИ
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
:2
moments/StopGradient╦
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*N
_output_shapes<
::8                                    2
moments/SquaredDifferenceб
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices╛
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/varianceГ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/SqueezeЛ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1а
AssignMovingAvg/decayConst*+
_class!
loc:@AssignMovingAvg/11217430*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayХ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_11217430*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp┼
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*+
_class!
loc:@AssignMovingAvg/11217430*
_output_shapes
:2
AssignMovingAvg/sub╝
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg/11217430*
_output_shapes
:2
AssignMovingAvg/mulЕ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_11217430AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/11217430*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpж
AssignMovingAvg_1/decayConst*-
_class#
!loc:@AssignMovingAvg_1/11217436*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayЫ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_11217436*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╧
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/11217436*
_output_shapes
:2
AssignMovingAvg_1/sub╞
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/11217436*
_output_shapes
:2
AssignMovingAvg_1/mulС
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_11217436AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/11217436*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulЭ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8                                    2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subм
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8                                    2
batchnorm/add_1▄
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*N
_output_shapes<
::8                                    2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:8                                    ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:v r
N
_output_shapes<
::8                                    
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
√
~
)__inference_layer2_layer_call_fn_11220087

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCall╒
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_layer2_layer_call_and_return_conditional_losses_112180292
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ╚::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ╚
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
о	
л
8__inference_batch_normalization_2_layer_call_fn_11219756

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallг
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*N
_output_shapes<
::8                                    *$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_112172932
StatefulPartitionedCall╡
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8                                    2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:8                                    ::::22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8                                    
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
┼
a
E__inference_flatten_layer_call_and_return_conditional_losses_11217907

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         А
2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         А
2

Identity"
identityIdentity:output:0*2
_input_shapes!
:         :[ W
3
_output_shapes!
:         
 
_user_specified_nameinputs
┬
Ц
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_11219743

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИТ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulЭ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8                                    2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subм
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8                                    2
batchnorm/add_1О
IdentityIdentitybatchnorm/add_1:z:0*
T0*N
_output_shapes<
::8                                    2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:8                                    :::::v r
N
_output_shapes<
::8                                    
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ч,
╬
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_11216969

inputs
assignmovingavg_11216944
assignmovingavg_1_11216950)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpЩ
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indicesЫ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/meanИ
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
:2
moments/StopGradient╦
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*N
_output_shapes<
::8                                    2
moments/SquaredDifferenceб
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices╛
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/varianceГ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/SqueezeЛ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1а
AssignMovingAvg/decayConst*+
_class!
loc:@AssignMovingAvg/11216944*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayХ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_11216944*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp┼
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*+
_class!
loc:@AssignMovingAvg/11216944*
_output_shapes
:2
AssignMovingAvg/sub╝
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg/11216944*
_output_shapes
:2
AssignMovingAvg/mulЕ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_11216944AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/11216944*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpж
AssignMovingAvg_1/decayConst*-
_class#
!loc:@AssignMovingAvg_1/11216950*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayЫ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_11216950*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╧
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/11216950*
_output_shapes
:2
AssignMovingAvg_1/sub╞
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/11216950*
_output_shapes
:2
AssignMovingAvg_1/mulС
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_11216950AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/11216950*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulЭ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8                                    2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subм
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8                                    2
batchnorm/add_1▄
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*N
_output_shapes<
::8                                    2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:8                                    ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:v r
N
_output_shapes<
::8                                    
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
╠
c
E__inference_dropout_layer_call_and_return_conditional_losses_11219997

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         А
2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А
2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:         А
:P L
(
_output_shapes
:         А

 
_user_specified_nameinputs
║
Ц
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_11217672

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИТ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulВ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:         2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subС
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:         2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0*
T0*3
_output_shapes!
:         2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:         :::::[ W
3
_output_shapes!
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
щ,
╨
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_11219441

inputs
assignmovingavg_11219416
assignmovingavg_1_11219422)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpЩ
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indicesЫ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/meanИ
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
:2
moments/StopGradient╦
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*N
_output_shapes<
::8                                    2
moments/SquaredDifferenceб
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices╛
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/varianceГ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/SqueezeЛ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1а
AssignMovingAvg/decayConst*+
_class!
loc:@AssignMovingAvg/11219416*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayХ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_11219416*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp┼
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*+
_class!
loc:@AssignMovingAvg/11219416*
_output_shapes
:2
AssignMovingAvg/sub╝
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg/11219416*
_output_shapes
:2
AssignMovingAvg/mulЕ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_11219416AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/11219416*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpж
AssignMovingAvg_1/decayConst*-
_class#
!loc:@AssignMovingAvg_1/11219422*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayЫ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_11219422*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╧
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/11219422*
_output_shapes
:2
AssignMovingAvg_1/sub╞
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/11219422*
_output_shapes
:2
AssignMovingAvg_1/mulС
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_11219422AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/11219422*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulЭ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8                                    2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subм
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8                                    2
batchnorm/add_1▄
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*N
_output_shapes<
::8                                    2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:8                                    ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:v r
N
_output_shapes<
::8                                    
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
К
d
E__inference_dropout_layer_call_and_return_conditional_losses_11217927

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU╒?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         А
2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╡
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А
*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠>2
dropout/GreaterEqual/y┐
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А
2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А
2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         А
2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         А
2

Identity"
identityIdentity:output:0*'
_input_shapes
:         А
:P L
(
_output_shapes
:         А

 
_user_specified_nameinputs
└
Ф
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_11219261

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИТ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulЭ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8                                    2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subм
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8                                    2
batchnorm/add_1О
IdentityIdentitybatchnorm/add_1:z:0*
T0*N
_output_shapes<
::8                                    2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:8                                    :::::v r
N
_output_shapes<
::8                                    
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ч
~
)__inference_conv3d_layer_call_fn_11216873

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCall№
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*N
_output_shapes<
::8                                    *$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_conv3d_layer_call_and_return_conditional_losses_112168632
StatefulPartitionedCall╡
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8                                    2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:9                                    н::22
StatefulPartitionedCallStatefulPartitionedCall:w s
O
_output_shapes=
;:9                                    н
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
░	
л
8__inference_batch_normalization_2_layer_call_fn_11219769

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallе
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*N
_output_shapes<
::8                                    *&
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_112173262
StatefulPartitionedCall╡
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8                                    2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:8                                    ::::22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8                                    
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
щ,
╨
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_11219841

inputs
assignmovingavg_11219816
assignmovingavg_1_11219822)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpЩ
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indicesЫ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/meanИ
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
:2
moments/StopGradient╦
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*N
_output_shapes<
::8                                    2
moments/SquaredDifferenceб
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices╛
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/varianceГ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/SqueezeЛ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1а
AssignMovingAvg/decayConst*+
_class!
loc:@AssignMovingAvg/11219816*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayХ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_11219816*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp┼
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*+
_class!
loc:@AssignMovingAvg/11219816*
_output_shapes
:2
AssignMovingAvg/sub╝
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg/11219816*
_output_shapes
:2
AssignMovingAvg/mulЕ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_11219816AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/11219816*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpж
AssignMovingAvg_1/decayConst*-
_class#
!loc:@AssignMovingAvg_1/11219822*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayЫ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_11219822*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╧
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/11219822*
_output_shapes
:2
AssignMovingAvg_1/sub╞
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/11219822*
_output_shapes
:2
AssignMovingAvg_1/mulС
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_11219822AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/11219822*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulЭ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8                                    2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subм
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8                                    2
batchnorm/add_1▄
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*N
_output_shapes<
::8                                    2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:8                                    ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:v r
N
_output_shapes<
::8                                    
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ж
Z
.__inference_concatenate_layer_call_fn_11220020
inputs_0
inputs_1
identity╢
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*(
_output_shapes
:         Г
* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_concatenate_layer_call_and_return_conditional_losses_112179522
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         Г
2

Identity"
identityIdentity:output:0*:
_input_shapes)
':         А
:         :R N
(
_output_shapes
:         А

"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/1
╘]
ь
C__inference_model_layer_call_and_return_conditional_losses_11218103
input_1
input_2
conv3d_11217516
conv3d_11217518 
batch_normalization_11217603 
batch_normalization_11217605 
batch_normalization_11217607 
batch_normalization_11217609
conv3d_1_11217612
conv3d_1_11217614"
batch_normalization_1_11217699"
batch_normalization_1_11217701"
batch_normalization_1_11217703"
batch_normalization_1_11217705
conv3d_2_11217708
conv3d_2_11217710"
batch_normalization_2_11217795"
batch_normalization_2_11217797"
batch_normalization_2_11217799"
batch_normalization_2_11217801
conv3d_3_11217804
conv3d_3_11217806"
batch_normalization_3_11217891"
batch_normalization_3_11217893"
batch_normalization_3_11217895"
batch_normalization_3_11217897
layer1_11217983
layer1_11217985
layer2_11218040
layer2_11218042
dense_11218097
dense_11218099
identityИв+batch_normalization/StatefulPartitionedCallв-batch_normalization_1/StatefulPartitionedCallв-batch_normalization_2/StatefulPartitionedCallв-batch_normalization_3/StatefulPartitionedCallвconv3d/StatefulPartitionedCallв conv3d_1/StatefulPartitionedCallв conv3d_2/StatefulPartitionedCallв conv3d_3/StatefulPartitionedCallвdense/StatefulPartitionedCallвdropout/StatefulPartitionedCallв!dropout_1/StatefulPartitionedCallв!dropout_2/StatefulPartitionedCallвlayer1/StatefulPartitionedCallвlayer2/StatefulPartitionedCall■
conv3d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv3d_11217516conv3d_11217518*
Tin
2*
Tout
2*3
_output_shapes!
:         *$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_conv3d_layer_call_and_return_conditional_losses_112168632 
conv3d/StatefulPartitionedCallЭ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv3d/StatefulPartitionedCall:output:0batch_normalization_11217603batch_normalization_11217605batch_normalization_11217607batch_normalization_11217609*
Tin	
2*
Tout
2*3
_output_shapes!
:         *$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_112175562-
+batch_normalization/StatefulPartitionedCall╡
 conv3d_1/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0conv3d_1_11217612conv3d_1_11217614*
Tin
2*
Tout
2*3
_output_shapes!
:         *$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_conv3d_1_layer_call_and_return_conditional_losses_112170252"
 conv3d_1/StatefulPartitionedCallн
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv3d_1/StatefulPartitionedCall:output:0batch_normalization_1_11217699batch_normalization_1_11217701batch_normalization_1_11217703batch_normalization_1_11217705*
Tin	
2*
Tout
2*3
_output_shapes!
:         *$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_112176522/
-batch_normalization_1/StatefulPartitionedCall╖
 conv3d_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0conv3d_2_11217708conv3d_2_11217710*
Tin
2*
Tout
2*3
_output_shapes!
:         *$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_conv3d_2_layer_call_and_return_conditional_losses_112171872"
 conv3d_2/StatefulPartitionedCallн
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv3d_2/StatefulPartitionedCall:output:0batch_normalization_2_11217795batch_normalization_2_11217797batch_normalization_2_11217799batch_normalization_2_11217801*
Tin	
2*
Tout
2*3
_output_shapes!
:         *$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_112177482/
-batch_normalization_2/StatefulPartitionedCall╖
 conv3d_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0conv3d_3_11217804conv3d_3_11217806*
Tin
2*
Tout
2*3
_output_shapes!
:         *$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_conv3d_3_layer_call_and_return_conditional_losses_112173492"
 conv3d_3/StatefulPartitionedCallн
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv3d_3/StatefulPartitionedCall:output:0batch_normalization_3_11217891batch_normalization_3_11217893batch_normalization_3_11217895batch_normalization_3_11217897*
Tin	
2*
Tout
2*3
_output_shapes!
:         *$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*\
fWRU
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_112178442/
-batch_normalization_3/StatefulPartitionedCallО
!average_pooling3d/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:         * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_average_pooling3d_layer_call_and_return_conditional_losses_112175052#
!average_pooling3d/PartitionedCall┘
flatten/PartitionedCallPartitionedCall*average_pooling3d/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:         А
* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_112179072
flatten/PartitionedCallч
dropout/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:         А
* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_112179272!
dropout/StatefulPartitionedCallэ
concatenate/PartitionedCallPartitionedCall(dropout/StatefulPartitionedCall:output:0input_2*
Tin
2*
Tout
2*(
_output_shapes
:         Г
* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_concatenate_layer_call_and_return_conditional_losses_112179522
concatenate/PartitionedCallР
layer1/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0layer1_11217983layer1_11217985*
Tin
2*
Tout
2*(
_output_shapes
:         ╚*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_layer1_layer_call_and_return_conditional_losses_112179722 
layer1/StatefulPartitionedCallЦ
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall'layer1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*(
_output_shapes
:         ╚* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_1_layer_call_and_return_conditional_losses_112180002#
!dropout_1/StatefulPartitionedCallХ
layer2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0layer2_11218040layer2_11218042*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_layer2_layer_call_and_return_conditional_losses_112180292 
layer2/StatefulPartitionedCallЧ
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall'layer2/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_2_layer_call_and_return_conditional_losses_112180572#
!dropout_2/StatefulPartitionedCallР
dense/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_11218097dense_11218099*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_112180862
dense/StatefulPartitionedCallО
IdentityIdentity&dense/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall^conv3d/StatefulPartitionedCall!^conv3d_1/StatefulPartitionedCall!^conv3d_2/StatefulPartitionedCall!^conv3d_3/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall^layer1/StatefulPartitionedCall^layer2/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*└
_input_shapesо
л:         н:         ::::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2@
conv3d/StatefulPartitionedCallconv3d/StatefulPartitionedCall2D
 conv3d_1/StatefulPartitionedCall conv3d_1/StatefulPartitionedCall2D
 conv3d_2/StatefulPartitionedCall conv3d_2/StatefulPartitionedCall2D
 conv3d_3/StatefulPartitionedCall conv3d_3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2@
layer2/StatefulPartitionedCalllayer2/StatefulPartitionedCall:] Y
4
_output_shapes"
 :         н
!
_user_specified_name	input_1:PL
'
_output_shapes
:         
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ч,
╬
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_11219241

inputs
assignmovingavg_11219216
assignmovingavg_1_11219222)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpЩ
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indicesЫ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/meanИ
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
:2
moments/StopGradient╦
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*N
_output_shapes<
::8                                    2
moments/SquaredDifferenceб
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices╛
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/varianceГ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/SqueezeЛ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1а
AssignMovingAvg/decayConst*+
_class!
loc:@AssignMovingAvg/11219216*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayХ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_11219216*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp┼
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*+
_class!
loc:@AssignMovingAvg/11219216*
_output_shapes
:2
AssignMovingAvg/sub╝
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg/11219216*
_output_shapes
:2
AssignMovingAvg/mulЕ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_11219216AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/11219216*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpж
AssignMovingAvg_1/decayConst*-
_class#
!loc:@AssignMovingAvg_1/11219222*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayЫ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_11219222*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╧
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/11219222*
_output_shapes
:2
AssignMovingAvg_1/sub╞
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/11219222*
_output_shapes
:2
AssignMovingAvg_1/mulС
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_11219222AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/11219222*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulЭ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8                                    2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subм
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8                                    2
batchnorm/add_1▄
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*N
_output_shapes<
::8                                    2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:8                                    ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:v r
N
_output_shapes<
::8                                    
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
┬
Ц
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_11219461

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИТ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulЭ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8                                    2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subм
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8                                    2
batchnorm/add_1О
IdentityIdentitybatchnorm/add_1:z:0*
T0*N
_output_shapes<
::8                                    2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:8                                    :::::v r
N
_output_shapes<
::8                                    
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
оЭ
┼
C__inference_model_layer_call_and_return_conditional_losses_11219037
inputs_0
inputs_1)
%conv3d_conv3d_readvariableop_resource*
&conv3d_biasadd_readvariableop_resource9
5batch_normalization_batchnorm_readvariableop_resource=
9batch_normalization_batchnorm_mul_readvariableop_resource;
7batch_normalization_batchnorm_readvariableop_1_resource;
7batch_normalization_batchnorm_readvariableop_2_resource+
'conv3d_1_conv3d_readvariableop_resource,
(conv3d_1_biasadd_readvariableop_resource;
7batch_normalization_1_batchnorm_readvariableop_resource?
;batch_normalization_1_batchnorm_mul_readvariableop_resource=
9batch_normalization_1_batchnorm_readvariableop_1_resource=
9batch_normalization_1_batchnorm_readvariableop_2_resource+
'conv3d_2_conv3d_readvariableop_resource,
(conv3d_2_biasadd_readvariableop_resource;
7batch_normalization_2_batchnorm_readvariableop_resource?
;batch_normalization_2_batchnorm_mul_readvariableop_resource=
9batch_normalization_2_batchnorm_readvariableop_1_resource=
9batch_normalization_2_batchnorm_readvariableop_2_resource+
'conv3d_3_conv3d_readvariableop_resource,
(conv3d_3_biasadd_readvariableop_resource;
7batch_normalization_3_batchnorm_readvariableop_resource?
;batch_normalization_3_batchnorm_mul_readvariableop_resource=
9batch_normalization_3_batchnorm_readvariableop_1_resource=
9batch_normalization_3_batchnorm_readvariableop_2_resource)
%layer1_matmul_readvariableop_resource*
&layer1_biasadd_readvariableop_resource)
%layer2_matmul_readvariableop_resource*
&layer2_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identityИп
conv3d/Conv3D/ReadVariableOpReadVariableOp%conv3d_conv3d_readvariableop_resource*+
_output_shapes
:н*
dtype02
conv3d/Conv3D/ReadVariableOp└
conv3d/Conv3DConv3Dinputs_0$conv3d/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         *
paddingVALID*
strides	
2
conv3d/Conv3Dб
conv3d/BiasAdd/ReadVariableOpReadVariableOp&conv3d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv3d/BiasAdd/ReadVariableOpи
conv3d/BiasAddBiasAddconv3d/Conv3D:output:0%conv3d/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         2
conv3d/BiasAdd╬
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization/batchnorm/ReadVariableOpП
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2%
#batch_normalization/batchnorm/add/y╪
!batch_normalization/batchnorm/addAddV24batch_normalization/batchnorm/ReadVariableOp:value:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:2#
!batch_normalization/batchnorm/addЯ
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:2%
#batch_normalization/batchnorm/Rsqrt┌
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype022
0batch_normalization/batchnorm/mul/ReadVariableOp╒
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!batch_normalization/batchnorm/mul╧
#batch_normalization/batchnorm/mul_1Mulconv3d/BiasAdd:output:0%batch_normalization/batchnorm/mul:z:0*
T0*3
_output_shapes!
:         2%
#batch_normalization/batchnorm/mul_1╘
.batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp7batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype020
.batch_normalization/batchnorm/ReadVariableOp_1╒
#batch_normalization/batchnorm/mul_2Mul6batch_normalization/batchnorm/ReadVariableOp_1:value:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:2%
#batch_normalization/batchnorm/mul_2╘
.batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp7batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype020
.batch_normalization/batchnorm/ReadVariableOp_2╙
!batch_normalization/batchnorm/subSub6batch_normalization/batchnorm/ReadVariableOp_2:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2#
!batch_normalization/batchnorm/subс
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*3
_output_shapes!
:         2%
#batch_normalization/batchnorm/add_1┤
conv3d_1/Conv3D/ReadVariableOpReadVariableOp'conv3d_1_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02 
conv3d_1/Conv3D/ReadVariableOpх
conv3d_1/Conv3DConv3D'batch_normalization/batchnorm/add_1:z:0&conv3d_1/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         *
paddingVALID*
strides	
2
conv3d_1/Conv3Dз
conv3d_1/BiasAdd/ReadVariableOpReadVariableOp(conv3d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv3d_1/BiasAdd/ReadVariableOp░
conv3d_1/BiasAddBiasAddconv3d_1/Conv3D:output:0'conv3d_1/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         2
conv3d_1/BiasAdd|
conv3d_1/EluEluconv3d_1/BiasAdd:output:0*
T0*3
_output_shapes!
:         2
conv3d_1/Elu╘
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype020
.batch_normalization_1/batchnorm/ReadVariableOpУ
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_1/batchnorm/add/yр
#batch_normalization_1/batchnorm/addAddV26batch_normalization_1/batchnorm/ReadVariableOp:value:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_1/batchnorm/addе
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_1/batchnorm/Rsqrtр
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype024
2batch_normalization_1/batchnorm/mul/ReadVariableOp▌
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_1/batchnorm/mul╪
%batch_normalization_1/batchnorm/mul_1Mulconv3d_1/Elu:activations:0'batch_normalization_1/batchnorm/mul:z:0*
T0*3
_output_shapes!
:         2'
%batch_normalization_1/batchnorm/mul_1┌
0batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype022
0batch_normalization_1/batchnorm/ReadVariableOp_1▌
%batch_normalization_1/batchnorm/mul_2Mul8batch_normalization_1/batchnorm/ReadVariableOp_1:value:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_1/batchnorm/mul_2┌
0batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype022
0batch_normalization_1/batchnorm/ReadVariableOp_2█
#batch_normalization_1/batchnorm/subSub8batch_normalization_1/batchnorm/ReadVariableOp_2:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_1/batchnorm/subщ
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*3
_output_shapes!
:         2'
%batch_normalization_1/batchnorm/add_1┤
conv3d_2/Conv3D/ReadVariableOpReadVariableOp'conv3d_2_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02 
conv3d_2/Conv3D/ReadVariableOpч
conv3d_2/Conv3DConv3D)batch_normalization_1/batchnorm/add_1:z:0&conv3d_2/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         *
paddingVALID*
strides	
2
conv3d_2/Conv3Dз
conv3d_2/BiasAdd/ReadVariableOpReadVariableOp(conv3d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv3d_2/BiasAdd/ReadVariableOp░
conv3d_2/BiasAddBiasAddconv3d_2/Conv3D:output:0'conv3d_2/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         2
conv3d_2/BiasAdd|
conv3d_2/EluEluconv3d_2/BiasAdd:output:0*
T0*3
_output_shapes!
:         2
conv3d_2/Elu╘
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype020
.batch_normalization_2/batchnorm/ReadVariableOpУ
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_2/batchnorm/add/yр
#batch_normalization_2/batchnorm/addAddV26batch_normalization_2/batchnorm/ReadVariableOp:value:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_2/batchnorm/addе
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_2/batchnorm/Rsqrtр
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype024
2batch_normalization_2/batchnorm/mul/ReadVariableOp▌
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_2/batchnorm/mul╪
%batch_normalization_2/batchnorm/mul_1Mulconv3d_2/Elu:activations:0'batch_normalization_2/batchnorm/mul:z:0*
T0*3
_output_shapes!
:         2'
%batch_normalization_2/batchnorm/mul_1┌
0batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype022
0batch_normalization_2/batchnorm/ReadVariableOp_1▌
%batch_normalization_2/batchnorm/mul_2Mul8batch_normalization_2/batchnorm/ReadVariableOp_1:value:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_2/batchnorm/mul_2┌
0batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype022
0batch_normalization_2/batchnorm/ReadVariableOp_2█
#batch_normalization_2/batchnorm/subSub8batch_normalization_2/batchnorm/ReadVariableOp_2:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_2/batchnorm/subщ
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*3
_output_shapes!
:         2'
%batch_normalization_2/batchnorm/add_1┤
conv3d_3/Conv3D/ReadVariableOpReadVariableOp'conv3d_3_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02 
conv3d_3/Conv3D/ReadVariableOpч
conv3d_3/Conv3DConv3D)batch_normalization_2/batchnorm/add_1:z:0&conv3d_3/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         *
paddingVALID*
strides	
2
conv3d_3/Conv3Dз
conv3d_3/BiasAdd/ReadVariableOpReadVariableOp(conv3d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv3d_3/BiasAdd/ReadVariableOp░
conv3d_3/BiasAddBiasAddconv3d_3/Conv3D:output:0'conv3d_3/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         2
conv3d_3/BiasAdd|
conv3d_3/EluEluconv3d_3/BiasAdd:output:0*
T0*3
_output_shapes!
:         2
conv3d_3/Elu╘
.batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype020
.batch_normalization_3/batchnorm/ReadVariableOpУ
%batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_3/batchnorm/add/yр
#batch_normalization_3/batchnorm/addAddV26batch_normalization_3/batchnorm/ReadVariableOp:value:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_3/batchnorm/addе
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_3/batchnorm/Rsqrtр
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype024
2batch_normalization_3/batchnorm/mul/ReadVariableOp▌
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:0:batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_3/batchnorm/mul╪
%batch_normalization_3/batchnorm/mul_1Mulconv3d_3/Elu:activations:0'batch_normalization_3/batchnorm/mul:z:0*
T0*3
_output_shapes!
:         2'
%batch_normalization_3/batchnorm/mul_1┌
0batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype022
0batch_normalization_3/batchnorm/ReadVariableOp_1▌
%batch_normalization_3/batchnorm/mul_2Mul8batch_normalization_3/batchnorm/ReadVariableOp_1:value:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_3/batchnorm/mul_2┌
0batch_normalization_3/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype022
0batch_normalization_3/batchnorm/ReadVariableOp_2█
#batch_normalization_3/batchnorm/subSub8batch_normalization_3/batchnorm/ReadVariableOp_2:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_3/batchnorm/subщ
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*3
_output_shapes!
:         2'
%batch_normalization_3/batchnorm/add_1ю
average_pooling3d/AvgPool3D	AvgPool3D)batch_normalization_3/batchnorm/add_1:z:0*
T0*3
_output_shapes!
:         *
ksize	
*
paddingVALID*
strides	
2
average_pooling3d/AvgPool3Do
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
flatten/ConstЮ
flatten/ReshapeReshape$average_pooling3d/AvgPool3D:output:0flatten/Const:output:0*
T0*(
_output_shapes
:         А
2
flatten/Reshape}
dropout/IdentityIdentityflatten/Reshape:output:0*
T0*(
_output_shapes
:         А
2
dropout/Identityt
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis╖
concatenate/concatConcatV2dropout/Identity:output:0inputs_1 concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:         Г
2
concatenate/concatд
layer1/MatMul/ReadVariableOpReadVariableOp%layer1_matmul_readvariableop_resource* 
_output_shapes
:
Г
╚*
dtype02
layer1/MatMul/ReadVariableOpЮ
layer1/MatMulMatMulconcatenate/concat:output:0$layer1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
layer1/MatMulв
layer1/BiasAdd/ReadVariableOpReadVariableOp&layer1_biasadd_readvariableop_resource*
_output_shapes	
:╚*
dtype02
layer1/BiasAdd/ReadVariableOpЮ
layer1/BiasAddBiasAddlayer1/MatMul:product:0%layer1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
layer1/BiasAddk

layer1/EluElulayer1/BiasAdd:output:0*
T0*(
_output_shapes
:         ╚2

layer1/EluБ
dropout_1/IdentityIdentitylayer1/Elu:activations:0*
T0*(
_output_shapes
:         ╚2
dropout_1/Identityг
layer2/MatMul/ReadVariableOpReadVariableOp%layer2_matmul_readvariableop_resource*
_output_shapes
:	╚*
dtype02
layer2/MatMul/ReadVariableOpЭ
layer2/MatMulMatMuldropout_1/Identity:output:0$layer2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
layer2/MatMulб
layer2/BiasAdd/ReadVariableOpReadVariableOp&layer2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
layer2/BiasAdd/ReadVariableOpЭ
layer2/BiasAddBiasAddlayer2/MatMul:product:0%layer2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
layer2/BiasAddj

layer2/EluElulayer2/BiasAdd:output:0*
T0*'
_output_shapes
:         2

layer2/EluА
dropout_2/IdentityIdentitylayer2/Elu:activations:0*
T0*'
_output_shapes
:         2
dropout_2/IdentityЯ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense/MatMul/ReadVariableOpЪ
dense/MatMulMatMuldropout_2/Identity:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense/MatMulЮ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOpЩ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense/BiasAdds
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense/Sigmoide
IdentityIdentitydense/Sigmoid:y:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*└
_input_shapesо
л:         н:         :::::::::::::::::::::::::::::::^ Z
4
_output_shapes"
 :         н
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
╛
й
6__inference_batch_normalization_layer_call_fn_11219356

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallЖ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*3
_output_shapes!
:         *$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_112175562
StatefulPartitionedCallЪ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:         2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:         ::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
╬
e
G__inference_dropout_1_layer_call_and_return_conditional_losses_11220057

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         ╚2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         ╚2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:         ╚:P L
(
_output_shapes
:         ╚
 
_user_specified_nameinputs
░	
л
8__inference_batch_normalization_1_layer_call_fn_11219487

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallе
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*N
_output_shapes<
::8                                    *&
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_112171642
StatefulPartitionedCall╡
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8                                    2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:8                                    ::::22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8                                    
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
╟+
╨
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_11217652

inputs
assignmovingavg_11217627
assignmovingavg_1_11217633)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpЩ
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indicesЫ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/meanИ
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
:2
moments/StopGradient░
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*3
_output_shapes!
:         2
moments/SquaredDifferenceб
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices╛
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/varianceГ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/SqueezeЛ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1а
AssignMovingAvg/decayConst*+
_class!
loc:@AssignMovingAvg/11217627*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayХ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_11217627*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp┼
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*+
_class!
loc:@AssignMovingAvg/11217627*
_output_shapes
:2
AssignMovingAvg/sub╝
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg/11217627*
_output_shapes
:2
AssignMovingAvg/mulЕ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_11217627AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/11217627*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpж
AssignMovingAvg_1/decayConst*-
_class#
!loc:@AssignMovingAvg_1/11217633*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayЫ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_11217633*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╧
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/11217633*
_output_shapes
:2
AssignMovingAvg_1/sub╞
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/11217633*
_output_shapes
:2
AssignMovingAvg_1/mulС
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_11217633AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/11217633*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulВ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:         2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subС
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:         2
batchnorm/add_1┴
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*3
_output_shapes!
:         2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:         ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:[ W
3
_output_shapes!
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
╟+
╨
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_11219523

inputs
assignmovingavg_11219498
assignmovingavg_1_11219504)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpЩ
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indicesЫ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/meanИ
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
:2
moments/StopGradient░
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*3
_output_shapes!
:         2
moments/SquaredDifferenceб
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices╛
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/varianceГ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/SqueezeЛ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1а
AssignMovingAvg/decayConst*+
_class!
loc:@AssignMovingAvg/11219498*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayХ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_11219498*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp┼
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*+
_class!
loc:@AssignMovingAvg/11219498*
_output_shapes
:2
AssignMovingAvg/sub╝
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg/11219498*
_output_shapes
:2
AssignMovingAvg/mulЕ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_11219498AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/11219498*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpж
AssignMovingAvg_1/decayConst*-
_class#
!loc:@AssignMovingAvg_1/11219504*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayЫ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_11219504*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╧
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/11219504*
_output_shapes
:2
AssignMovingAvg_1/sub╞
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/11219504*
_output_shapes
:2
AssignMovingAvg_1/mulС
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_11219504AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/11219504*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulВ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:         2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subС
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:         2
batchnorm/add_1┴
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*3
_output_shapes!
:         2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:         ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:[ W
3
_output_shapes!
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Д
о
F__inference_conv3d_3_layer_call_and_return_conditional_losses_11217349

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identityИЩ
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype02
Conv3D/ReadVariableOp─
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8                                    *
paddingVALID*
strides	
2
Conv3DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpз
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8                                    2	
BiasAdd|
EluEluBiasAdd:output:0*
T0*N
_output_shapes<
::8                                    2
EluМ
IdentityIdentityElu:activations:0*
T0*N
_output_shapes<
::8                                    2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8                                    :::v r
N
_output_shapes<
::8                                    
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Е
e
,__inference_dropout_2_layer_call_fn_11220109

inputs
identityИвStatefulPartitionedCall╛
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_2_layer_call_and_return_conditional_losses_112180572
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*&
_input_shapes
:         22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Ъ
А
+__inference_conv3d_3_layer_call_fn_11217359

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCall■
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*N
_output_shapes<
::8                                    *$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_conv3d_3_layer_call_and_return_conditional_losses_112173492
StatefulPartitionedCall╡
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8                                    2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8                                    ::22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8                                    
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
р
└
&__inference_signature_wrapper_11218702
input_1
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28
identityИвStatefulPartitionedCall╛
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28*+
Tin$
"2 *
Tout
2*'
_output_shapes
:         *@
_read_only_resource_inputs"
 	
*-
config_proto

GPU

CPU2*0J 8*,
f'R%
#__inference__wrapped_model_112168522
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*└
_input_shapesо
л:         н:         ::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
4
_output_shapes"
 :         н
!
_user_specified_name	input_1:PL
'
_output_shapes
:         
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
┬
л
8__inference_batch_normalization_2_layer_call_fn_11219674

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallИ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*3
_output_shapes!
:         *$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_112177482
StatefulPartitionedCallЪ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:         2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:         ::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
║
Ц
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_11217768

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИТ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulВ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:         2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subС
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:         2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0*
T0*3
_output_shapes!
:         2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:         :::::[ W
3
_output_shapes!
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
╠
c
E__inference_dropout_layer_call_and_return_conditional_losses_11217932

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         А
2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А
2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:         А
:P L
(
_output_shapes
:         А

 
_user_specified_nameinputs
л▄
Р-
$__inference__traced_restore_11220660
file_prefix"
assignvariableop_conv3d_kernel"
assignvariableop_1_conv3d_bias0
,assignvariableop_2_batch_normalization_gamma/
+assignvariableop_3_batch_normalization_beta6
2assignvariableop_4_batch_normalization_moving_mean:
6assignvariableop_5_batch_normalization_moving_variance&
"assignvariableop_6_conv3d_1_kernel$
 assignvariableop_7_conv3d_1_bias2
.assignvariableop_8_batch_normalization_1_gamma1
-assignvariableop_9_batch_normalization_1_beta9
5assignvariableop_10_batch_normalization_1_moving_mean=
9assignvariableop_11_batch_normalization_1_moving_variance'
#assignvariableop_12_conv3d_2_kernel%
!assignvariableop_13_conv3d_2_bias3
/assignvariableop_14_batch_normalization_2_gamma2
.assignvariableop_15_batch_normalization_2_beta9
5assignvariableop_16_batch_normalization_2_moving_mean=
9assignvariableop_17_batch_normalization_2_moving_variance'
#assignvariableop_18_conv3d_3_kernel%
!assignvariableop_19_conv3d_3_bias3
/assignvariableop_20_batch_normalization_3_gamma2
.assignvariableop_21_batch_normalization_3_beta9
5assignvariableop_22_batch_normalization_3_moving_mean=
9assignvariableop_23_batch_normalization_3_moving_variance%
!assignvariableop_24_layer1_kernel#
assignvariableop_25_layer1_bias%
!assignvariableop_26_layer2_kernel#
assignvariableop_27_layer2_bias$
 assignvariableop_28_dense_kernel"
assignvariableop_29_dense_bias!
assignvariableop_30_adam_iter#
assignvariableop_31_adam_beta_1#
assignvariableop_32_adam_beta_2"
assignvariableop_33_adam_decay*
&assignvariableop_34_adam_learning_rate
assignvariableop_35_total
assignvariableop_36_count,
(assignvariableop_37_adam_conv3d_kernel_m*
&assignvariableop_38_adam_conv3d_bias_m8
4assignvariableop_39_adam_batch_normalization_gamma_m7
3assignvariableop_40_adam_batch_normalization_beta_m.
*assignvariableop_41_adam_conv3d_1_kernel_m,
(assignvariableop_42_adam_conv3d_1_bias_m:
6assignvariableop_43_adam_batch_normalization_1_gamma_m9
5assignvariableop_44_adam_batch_normalization_1_beta_m.
*assignvariableop_45_adam_conv3d_2_kernel_m,
(assignvariableop_46_adam_conv3d_2_bias_m:
6assignvariableop_47_adam_batch_normalization_2_gamma_m9
5assignvariableop_48_adam_batch_normalization_2_beta_m.
*assignvariableop_49_adam_conv3d_3_kernel_m,
(assignvariableop_50_adam_conv3d_3_bias_m:
6assignvariableop_51_adam_batch_normalization_3_gamma_m9
5assignvariableop_52_adam_batch_normalization_3_beta_m,
(assignvariableop_53_adam_layer1_kernel_m*
&assignvariableop_54_adam_layer1_bias_m,
(assignvariableop_55_adam_layer2_kernel_m*
&assignvariableop_56_adam_layer2_bias_m+
'assignvariableop_57_adam_dense_kernel_m)
%assignvariableop_58_adam_dense_bias_m,
(assignvariableop_59_adam_conv3d_kernel_v*
&assignvariableop_60_adam_conv3d_bias_v8
4assignvariableop_61_adam_batch_normalization_gamma_v7
3assignvariableop_62_adam_batch_normalization_beta_v.
*assignvariableop_63_adam_conv3d_1_kernel_v,
(assignvariableop_64_adam_conv3d_1_bias_v:
6assignvariableop_65_adam_batch_normalization_1_gamma_v9
5assignvariableop_66_adam_batch_normalization_1_beta_v.
*assignvariableop_67_adam_conv3d_2_kernel_v,
(assignvariableop_68_adam_conv3d_2_bias_v:
6assignvariableop_69_adam_batch_normalization_2_gamma_v9
5assignvariableop_70_adam_batch_normalization_2_beta_v.
*assignvariableop_71_adam_conv3d_3_kernel_v,
(assignvariableop_72_adam_conv3d_3_bias_v:
6assignvariableop_73_adam_batch_normalization_3_gamma_v9
5assignvariableop_74_adam_batch_normalization_3_beta_v,
(assignvariableop_75_adam_layer1_kernel_v*
&assignvariableop_76_adam_layer1_bias_v,
(assignvariableop_77_adam_layer2_kernel_v*
&assignvariableop_78_adam_layer2_bias_v+
'assignvariableop_79_adam_dense_kernel_v)
%assignvariableop_80_adam_dense_bias_v
identity_82ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_33вAssignVariableOp_34вAssignVariableOp_35вAssignVariableOp_36вAssignVariableOp_37вAssignVariableOp_38вAssignVariableOp_39вAssignVariableOp_4вAssignVariableOp_40вAssignVariableOp_41вAssignVariableOp_42вAssignVariableOp_43вAssignVariableOp_44вAssignVariableOp_45вAssignVariableOp_46вAssignVariableOp_47вAssignVariableOp_48вAssignVariableOp_49вAssignVariableOp_5вAssignVariableOp_50вAssignVariableOp_51вAssignVariableOp_52вAssignVariableOp_53вAssignVariableOp_54вAssignVariableOp_55вAssignVariableOp_56вAssignVariableOp_57вAssignVariableOp_58вAssignVariableOp_59вAssignVariableOp_6вAssignVariableOp_60вAssignVariableOp_61вAssignVariableOp_62вAssignVariableOp_63вAssignVariableOp_64вAssignVariableOp_65вAssignVariableOp_66вAssignVariableOp_67вAssignVariableOp_68вAssignVariableOp_69вAssignVariableOp_7вAssignVariableOp_70вAssignVariableOp_71вAssignVariableOp_72вAssignVariableOp_73вAssignVariableOp_74вAssignVariableOp_75вAssignVariableOp_76вAssignVariableOp_77вAssignVariableOp_78вAssignVariableOp_79вAssignVariableOp_8вAssignVariableOp_80вAssignVariableOp_9в	RestoreV2вRestoreV2_1╩-
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:Q*
dtype0*╓,
value╠,B╔,QB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names│
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:Q*
dtype0*╖
valueнBкQB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices├
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*┌
_output_shapes╟
─:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*_
dtypesU
S2Q	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

IdentityО
AssignVariableOpAssignVariableOpassignvariableop_conv3d_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1Ф
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv3d_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2в
AssignVariableOp_2AssignVariableOp,assignvariableop_2_batch_normalization_gammaIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3б
AssignVariableOp_3AssignVariableOp+assignvariableop_3_batch_normalization_betaIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4и
AssignVariableOp_4AssignVariableOp2assignvariableop_4_batch_normalization_moving_meanIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5м
AssignVariableOp_5AssignVariableOp6assignvariableop_5_batch_normalization_moving_varianceIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6Ш
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv3d_1_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7Ц
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv3d_1_biasIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8д
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_1_gammaIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9г
AssignVariableOp_9AssignVariableOp-assignvariableop_9_batch_normalization_1_betaIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10о
AssignVariableOp_10AssignVariableOp5assignvariableop_10_batch_normalization_1_moving_meanIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11▓
AssignVariableOp_11AssignVariableOp9assignvariableop_11_batch_normalization_1_moving_varianceIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12Ь
AssignVariableOp_12AssignVariableOp#assignvariableop_12_conv3d_2_kernelIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13Ъ
AssignVariableOp_13AssignVariableOp!assignvariableop_13_conv3d_2_biasIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14и
AssignVariableOp_14AssignVariableOp/assignvariableop_14_batch_normalization_2_gammaIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15з
AssignVariableOp_15AssignVariableOp.assignvariableop_15_batch_normalization_2_betaIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16о
AssignVariableOp_16AssignVariableOp5assignvariableop_16_batch_normalization_2_moving_meanIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17▓
AssignVariableOp_17AssignVariableOp9assignvariableop_17_batch_normalization_2_moving_varianceIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18Ь
AssignVariableOp_18AssignVariableOp#assignvariableop_18_conv3d_3_kernelIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19Ъ
AssignVariableOp_19AssignVariableOp!assignvariableop_19_conv3d_3_biasIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20и
AssignVariableOp_20AssignVariableOp/assignvariableop_20_batch_normalization_3_gammaIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21з
AssignVariableOp_21AssignVariableOp.assignvariableop_21_batch_normalization_3_betaIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22о
AssignVariableOp_22AssignVariableOp5assignvariableop_22_batch_normalization_3_moving_meanIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23▓
AssignVariableOp_23AssignVariableOp9assignvariableop_23_batch_normalization_3_moving_varianceIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24Ъ
AssignVariableOp_24AssignVariableOp!assignvariableop_24_layer1_kernelIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25Ш
AssignVariableOp_25AssignVariableOpassignvariableop_25_layer1_biasIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26Ъ
AssignVariableOp_26AssignVariableOp!assignvariableop_26_layer2_kernelIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27Ш
AssignVariableOp_27AssignVariableOpassignvariableop_27_layer2_biasIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28Щ
AssignVariableOp_28AssignVariableOp assignvariableop_28_dense_kernelIdentity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29Ч
AssignVariableOp_29AssignVariableOpassignvariableop_29_dense_biasIdentity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29_
Identity_30IdentityRestoreV2:tensors:30*
T0	*
_output_shapes
:2
Identity_30Ц
AssignVariableOp_30AssignVariableOpassignvariableop_30_adam_iterIdentity_30:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_30_
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:2
Identity_31Ш
AssignVariableOp_31AssignVariableOpassignvariableop_31_adam_beta_1Identity_31:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_31_
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:2
Identity_32Ш
AssignVariableOp_32AssignVariableOpassignvariableop_32_adam_beta_2Identity_32:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_32_
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:2
Identity_33Ч
AssignVariableOp_33AssignVariableOpassignvariableop_33_adam_decayIdentity_33:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_33_
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:2
Identity_34Я
AssignVariableOp_34AssignVariableOp&assignvariableop_34_adam_learning_rateIdentity_34:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_34_
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:2
Identity_35Т
AssignVariableOp_35AssignVariableOpassignvariableop_35_totalIdentity_35:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_35_
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:2
Identity_36Т
AssignVariableOp_36AssignVariableOpassignvariableop_36_countIdentity_36:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_36_
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:2
Identity_37б
AssignVariableOp_37AssignVariableOp(assignvariableop_37_adam_conv3d_kernel_mIdentity_37:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_37_
Identity_38IdentityRestoreV2:tensors:38*
T0*
_output_shapes
:2
Identity_38Я
AssignVariableOp_38AssignVariableOp&assignvariableop_38_adam_conv3d_bias_mIdentity_38:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_38_
Identity_39IdentityRestoreV2:tensors:39*
T0*
_output_shapes
:2
Identity_39н
AssignVariableOp_39AssignVariableOp4assignvariableop_39_adam_batch_normalization_gamma_mIdentity_39:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_39_
Identity_40IdentityRestoreV2:tensors:40*
T0*
_output_shapes
:2
Identity_40м
AssignVariableOp_40AssignVariableOp3assignvariableop_40_adam_batch_normalization_beta_mIdentity_40:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_40_
Identity_41IdentityRestoreV2:tensors:41*
T0*
_output_shapes
:2
Identity_41г
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_conv3d_1_kernel_mIdentity_41:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_41_
Identity_42IdentityRestoreV2:tensors:42*
T0*
_output_shapes
:2
Identity_42б
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_conv3d_1_bias_mIdentity_42:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_42_
Identity_43IdentityRestoreV2:tensors:43*
T0*
_output_shapes
:2
Identity_43п
AssignVariableOp_43AssignVariableOp6assignvariableop_43_adam_batch_normalization_1_gamma_mIdentity_43:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_43_
Identity_44IdentityRestoreV2:tensors:44*
T0*
_output_shapes
:2
Identity_44о
AssignVariableOp_44AssignVariableOp5assignvariableop_44_adam_batch_normalization_1_beta_mIdentity_44:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_44_
Identity_45IdentityRestoreV2:tensors:45*
T0*
_output_shapes
:2
Identity_45г
AssignVariableOp_45AssignVariableOp*assignvariableop_45_adam_conv3d_2_kernel_mIdentity_45:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_45_
Identity_46IdentityRestoreV2:tensors:46*
T0*
_output_shapes
:2
Identity_46б
AssignVariableOp_46AssignVariableOp(assignvariableop_46_adam_conv3d_2_bias_mIdentity_46:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_46_
Identity_47IdentityRestoreV2:tensors:47*
T0*
_output_shapes
:2
Identity_47п
AssignVariableOp_47AssignVariableOp6assignvariableop_47_adam_batch_normalization_2_gamma_mIdentity_47:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_47_
Identity_48IdentityRestoreV2:tensors:48*
T0*
_output_shapes
:2
Identity_48о
AssignVariableOp_48AssignVariableOp5assignvariableop_48_adam_batch_normalization_2_beta_mIdentity_48:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_48_
Identity_49IdentityRestoreV2:tensors:49*
T0*
_output_shapes
:2
Identity_49г
AssignVariableOp_49AssignVariableOp*assignvariableop_49_adam_conv3d_3_kernel_mIdentity_49:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_49_
Identity_50IdentityRestoreV2:tensors:50*
T0*
_output_shapes
:2
Identity_50б
AssignVariableOp_50AssignVariableOp(assignvariableop_50_adam_conv3d_3_bias_mIdentity_50:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_50_
Identity_51IdentityRestoreV2:tensors:51*
T0*
_output_shapes
:2
Identity_51п
AssignVariableOp_51AssignVariableOp6assignvariableop_51_adam_batch_normalization_3_gamma_mIdentity_51:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_51_
Identity_52IdentityRestoreV2:tensors:52*
T0*
_output_shapes
:2
Identity_52о
AssignVariableOp_52AssignVariableOp5assignvariableop_52_adam_batch_normalization_3_beta_mIdentity_52:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_52_
Identity_53IdentityRestoreV2:tensors:53*
T0*
_output_shapes
:2
Identity_53б
AssignVariableOp_53AssignVariableOp(assignvariableop_53_adam_layer1_kernel_mIdentity_53:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_53_
Identity_54IdentityRestoreV2:tensors:54*
T0*
_output_shapes
:2
Identity_54Я
AssignVariableOp_54AssignVariableOp&assignvariableop_54_adam_layer1_bias_mIdentity_54:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_54_
Identity_55IdentityRestoreV2:tensors:55*
T0*
_output_shapes
:2
Identity_55б
AssignVariableOp_55AssignVariableOp(assignvariableop_55_adam_layer2_kernel_mIdentity_55:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_55_
Identity_56IdentityRestoreV2:tensors:56*
T0*
_output_shapes
:2
Identity_56Я
AssignVariableOp_56AssignVariableOp&assignvariableop_56_adam_layer2_bias_mIdentity_56:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_56_
Identity_57IdentityRestoreV2:tensors:57*
T0*
_output_shapes
:2
Identity_57а
AssignVariableOp_57AssignVariableOp'assignvariableop_57_adam_dense_kernel_mIdentity_57:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_57_
Identity_58IdentityRestoreV2:tensors:58*
T0*
_output_shapes
:2
Identity_58Ю
AssignVariableOp_58AssignVariableOp%assignvariableop_58_adam_dense_bias_mIdentity_58:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_58_
Identity_59IdentityRestoreV2:tensors:59*
T0*
_output_shapes
:2
Identity_59б
AssignVariableOp_59AssignVariableOp(assignvariableop_59_adam_conv3d_kernel_vIdentity_59:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_59_
Identity_60IdentityRestoreV2:tensors:60*
T0*
_output_shapes
:2
Identity_60Я
AssignVariableOp_60AssignVariableOp&assignvariableop_60_adam_conv3d_bias_vIdentity_60:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_60_
Identity_61IdentityRestoreV2:tensors:61*
T0*
_output_shapes
:2
Identity_61н
AssignVariableOp_61AssignVariableOp4assignvariableop_61_adam_batch_normalization_gamma_vIdentity_61:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_61_
Identity_62IdentityRestoreV2:tensors:62*
T0*
_output_shapes
:2
Identity_62м
AssignVariableOp_62AssignVariableOp3assignvariableop_62_adam_batch_normalization_beta_vIdentity_62:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_62_
Identity_63IdentityRestoreV2:tensors:63*
T0*
_output_shapes
:2
Identity_63г
AssignVariableOp_63AssignVariableOp*assignvariableop_63_adam_conv3d_1_kernel_vIdentity_63:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_63_
Identity_64IdentityRestoreV2:tensors:64*
T0*
_output_shapes
:2
Identity_64б
AssignVariableOp_64AssignVariableOp(assignvariableop_64_adam_conv3d_1_bias_vIdentity_64:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_64_
Identity_65IdentityRestoreV2:tensors:65*
T0*
_output_shapes
:2
Identity_65п
AssignVariableOp_65AssignVariableOp6assignvariableop_65_adam_batch_normalization_1_gamma_vIdentity_65:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_65_
Identity_66IdentityRestoreV2:tensors:66*
T0*
_output_shapes
:2
Identity_66о
AssignVariableOp_66AssignVariableOp5assignvariableop_66_adam_batch_normalization_1_beta_vIdentity_66:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_66_
Identity_67IdentityRestoreV2:tensors:67*
T0*
_output_shapes
:2
Identity_67г
AssignVariableOp_67AssignVariableOp*assignvariableop_67_adam_conv3d_2_kernel_vIdentity_67:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_67_
Identity_68IdentityRestoreV2:tensors:68*
T0*
_output_shapes
:2
Identity_68б
AssignVariableOp_68AssignVariableOp(assignvariableop_68_adam_conv3d_2_bias_vIdentity_68:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_68_
Identity_69IdentityRestoreV2:tensors:69*
T0*
_output_shapes
:2
Identity_69п
AssignVariableOp_69AssignVariableOp6assignvariableop_69_adam_batch_normalization_2_gamma_vIdentity_69:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_69_
Identity_70IdentityRestoreV2:tensors:70*
T0*
_output_shapes
:2
Identity_70о
AssignVariableOp_70AssignVariableOp5assignvariableop_70_adam_batch_normalization_2_beta_vIdentity_70:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_70_
Identity_71IdentityRestoreV2:tensors:71*
T0*
_output_shapes
:2
Identity_71г
AssignVariableOp_71AssignVariableOp*assignvariableop_71_adam_conv3d_3_kernel_vIdentity_71:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_71_
Identity_72IdentityRestoreV2:tensors:72*
T0*
_output_shapes
:2
Identity_72б
AssignVariableOp_72AssignVariableOp(assignvariableop_72_adam_conv3d_3_bias_vIdentity_72:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_72_
Identity_73IdentityRestoreV2:tensors:73*
T0*
_output_shapes
:2
Identity_73п
AssignVariableOp_73AssignVariableOp6assignvariableop_73_adam_batch_normalization_3_gamma_vIdentity_73:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_73_
Identity_74IdentityRestoreV2:tensors:74*
T0*
_output_shapes
:2
Identity_74о
AssignVariableOp_74AssignVariableOp5assignvariableop_74_adam_batch_normalization_3_beta_vIdentity_74:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_74_
Identity_75IdentityRestoreV2:tensors:75*
T0*
_output_shapes
:2
Identity_75б
AssignVariableOp_75AssignVariableOp(assignvariableop_75_adam_layer1_kernel_vIdentity_75:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_75_
Identity_76IdentityRestoreV2:tensors:76*
T0*
_output_shapes
:2
Identity_76Я
AssignVariableOp_76AssignVariableOp&assignvariableop_76_adam_layer1_bias_vIdentity_76:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_76_
Identity_77IdentityRestoreV2:tensors:77*
T0*
_output_shapes
:2
Identity_77б
AssignVariableOp_77AssignVariableOp(assignvariableop_77_adam_layer2_kernel_vIdentity_77:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_77_
Identity_78IdentityRestoreV2:tensors:78*
T0*
_output_shapes
:2
Identity_78Я
AssignVariableOp_78AssignVariableOp&assignvariableop_78_adam_layer2_bias_vIdentity_78:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_78_
Identity_79IdentityRestoreV2:tensors:79*
T0*
_output_shapes
:2
Identity_79а
AssignVariableOp_79AssignVariableOp'assignvariableop_79_adam_dense_kernel_vIdentity_79:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_79_
Identity_80IdentityRestoreV2:tensors:80*
T0*
_output_shapes
:2
Identity_80Ю
AssignVariableOp_80AssignVariableOp%assignvariableop_80_adam_dense_bias_vIdentity_80:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_80и
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_namesФ
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices─
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp╘
Identity_81Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_81с
Identity_82IdentityIdentity_81:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_82"#
identity_82Identity_82:output:0*█
_input_shapes╔
╞: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
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
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: :1

_output_shapes
: :2

_output_shapes
: :3

_output_shapes
: :4

_output_shapes
: :5

_output_shapes
: :6

_output_shapes
: :7

_output_shapes
: :8

_output_shapes
: :9

_output_shapes
: ::

_output_shapes
: :;

_output_shapes
: :<

_output_shapes
: :=

_output_shapes
: :>

_output_shapes
: :?

_output_shapes
: :@

_output_shapes
: :A

_output_shapes
: :B

_output_shapes
: :C

_output_shapes
: :D

_output_shapes
: :E

_output_shapes
: :F

_output_shapes
: :G

_output_shapes
: :H

_output_shapes
: :I

_output_shapes
: :J

_output_shapes
: :K

_output_shapes
: :L

_output_shapes
: :M

_output_shapes
: :N

_output_shapes
: :O

_output_shapes
: :P

_output_shapes
: :Q

_output_shapes
: 
║
Ц
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_11219543

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИТ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulВ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:         2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subС
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:         2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0*
T0*3
_output_shapes!
:         2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:         :::::[ W
3
_output_shapes!
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
─
л
8__inference_batch_normalization_3_layer_call_fn_11219969

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallК
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*3
_output_shapes!
:         *&
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*\
fWRU
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_112178642
StatefulPartitionedCallЪ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:         2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:         ::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
М
f
G__inference_dropout_1_layer_call_and_return_conditional_losses_11218000

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         ╚2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╡
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         ╚*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/y┐
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ╚2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ╚2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         ╚2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         ╚2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ╚:P L
(
_output_shapes
:         ╚
 
_user_specified_nameinputs
 ╠
┘
C__inference_model_layer_call_and_return_conditional_losses_11218912
inputs_0
inputs_1)
%conv3d_conv3d_readvariableop_resource*
&conv3d_biasadd_readvariableop_resource0
,batch_normalization_assignmovingavg_112187202
.batch_normalization_assignmovingavg_1_11218726=
9batch_normalization_batchnorm_mul_readvariableop_resource9
5batch_normalization_batchnorm_readvariableop_resource+
'conv3d_1_conv3d_readvariableop_resource,
(conv3d_1_biasadd_readvariableop_resource2
.batch_normalization_1_assignmovingavg_112187594
0batch_normalization_1_assignmovingavg_1_11218765?
;batch_normalization_1_batchnorm_mul_readvariableop_resource;
7batch_normalization_1_batchnorm_readvariableop_resource+
'conv3d_2_conv3d_readvariableop_resource,
(conv3d_2_biasadd_readvariableop_resource2
.batch_normalization_2_assignmovingavg_112187984
0batch_normalization_2_assignmovingavg_1_11218804?
;batch_normalization_2_batchnorm_mul_readvariableop_resource;
7batch_normalization_2_batchnorm_readvariableop_resource+
'conv3d_3_conv3d_readvariableop_resource,
(conv3d_3_biasadd_readvariableop_resource2
.batch_normalization_3_assignmovingavg_112188374
0batch_normalization_3_assignmovingavg_1_11218843?
;batch_normalization_3_batchnorm_mul_readvariableop_resource;
7batch_normalization_3_batchnorm_readvariableop_resource)
%layer1_matmul_readvariableop_resource*
&layer1_biasadd_readvariableop_resource)
%layer2_matmul_readvariableop_resource*
&layer2_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identityИв7batch_normalization/AssignMovingAvg/AssignSubVariableOpв9batch_normalization/AssignMovingAvg_1/AssignSubVariableOpв9batch_normalization_1/AssignMovingAvg/AssignSubVariableOpв;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOpв9batch_normalization_2/AssignMovingAvg/AssignSubVariableOpв;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOpв9batch_normalization_3/AssignMovingAvg/AssignSubVariableOpв;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOpп
conv3d/Conv3D/ReadVariableOpReadVariableOp%conv3d_conv3d_readvariableop_resource*+
_output_shapes
:н*
dtype02
conv3d/Conv3D/ReadVariableOp└
conv3d/Conv3DConv3Dinputs_0$conv3d/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         *
paddingVALID*
strides	
2
conv3d/Conv3Dб
conv3d/BiasAdd/ReadVariableOpReadVariableOp&conv3d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv3d/BiasAdd/ReadVariableOpи
conv3d/BiasAddBiasAddconv3d/Conv3D:output:0%conv3d/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         2
conv3d/BiasAdd┴
2batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             24
2batch_normalization/moments/mean/reduction_indicesш
 batch_normalization/moments/meanMeanconv3d/BiasAdd:output:0;batch_normalization/moments/mean/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2"
 batch_normalization/moments/mean─
(batch_normalization/moments/StopGradientStopGradient)batch_normalization/moments/mean:output:0*
T0**
_output_shapes
:2*
(batch_normalization/moments/StopGradient¤
-batch_normalization/moments/SquaredDifferenceSquaredDifferenceconv3d/BiasAdd:output:01batch_normalization/moments/StopGradient:output:0*
T0*3
_output_shapes!
:         2/
-batch_normalization/moments/SquaredDifference╔
6batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             28
6batch_normalization/moments/variance/reduction_indicesО
$batch_normalization/moments/varianceMean1batch_normalization/moments/SquaredDifference:z:0?batch_normalization/moments/variance/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2&
$batch_normalization/moments/variance┐
#batch_normalization/moments/SqueezeSqueeze)batch_normalization/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2%
#batch_normalization/moments/Squeeze╟
%batch_normalization/moments/Squeeze_1Squeeze-batch_normalization/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2'
%batch_normalization/moments/Squeeze_1▄
)batch_normalization/AssignMovingAvg/decayConst*?
_class5
31loc:@batch_normalization/AssignMovingAvg/11218720*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2+
)batch_normalization/AssignMovingAvg/decay╤
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOp,batch_normalization_assignmovingavg_11218720*
_output_shapes
:*
dtype024
2batch_normalization/AssignMovingAvg/ReadVariableOpй
'batch_normalization/AssignMovingAvg/subSub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:0,batch_normalization/moments/Squeeze:output:0*
T0*?
_class5
31loc:@batch_normalization/AssignMovingAvg/11218720*
_output_shapes
:2)
'batch_normalization/AssignMovingAvg/subа
'batch_normalization/AssignMovingAvg/mulMul+batch_normalization/AssignMovingAvg/sub:z:02batch_normalization/AssignMovingAvg/decay:output:0*
T0*?
_class5
31loc:@batch_normalization/AssignMovingAvg/11218720*
_output_shapes
:2)
'batch_normalization/AssignMovingAvg/mul¤
7batch_normalization/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,batch_normalization_assignmovingavg_11218720+batch_normalization/AssignMovingAvg/mul:z:03^batch_normalization/AssignMovingAvg/ReadVariableOp*?
_class5
31loc:@batch_normalization/AssignMovingAvg/11218720*
_output_shapes
 *
dtype029
7batch_normalization/AssignMovingAvg/AssignSubVariableOpт
+batch_normalization/AssignMovingAvg_1/decayConst*A
_class7
53loc:@batch_normalization/AssignMovingAvg_1/11218726*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2-
+batch_normalization/AssignMovingAvg_1/decay╫
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp.batch_normalization_assignmovingavg_1_11218726*
_output_shapes
:*
dtype026
4batch_normalization/AssignMovingAvg_1/ReadVariableOp│
)batch_normalization/AssignMovingAvg_1/subSub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0.batch_normalization/moments/Squeeze_1:output:0*
T0*A
_class7
53loc:@batch_normalization/AssignMovingAvg_1/11218726*
_output_shapes
:2+
)batch_normalization/AssignMovingAvg_1/subк
)batch_normalization/AssignMovingAvg_1/mulMul-batch_normalization/AssignMovingAvg_1/sub:z:04batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*A
_class7
53loc:@batch_normalization/AssignMovingAvg_1/11218726*
_output_shapes
:2+
)batch_normalization/AssignMovingAvg_1/mulЙ
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.batch_normalization_assignmovingavg_1_11218726-batch_normalization/AssignMovingAvg_1/mul:z:05^batch_normalization/AssignMovingAvg_1/ReadVariableOp*A
_class7
53loc:@batch_normalization/AssignMovingAvg_1/11218726*
_output_shapes
 *
dtype02;
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOpП
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2%
#batch_normalization/batchnorm/add/y╥
!batch_normalization/batchnorm/addAddV2.batch_normalization/moments/Squeeze_1:output:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:2#
!batch_normalization/batchnorm/addЯ
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:2%
#batch_normalization/batchnorm/Rsqrt┌
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype022
0batch_normalization/batchnorm/mul/ReadVariableOp╒
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!batch_normalization/batchnorm/mul╧
#batch_normalization/batchnorm/mul_1Mulconv3d/BiasAdd:output:0%batch_normalization/batchnorm/mul:z:0*
T0*3
_output_shapes!
:         2%
#batch_normalization/batchnorm/mul_1╦
#batch_normalization/batchnorm/mul_2Mul,batch_normalization/moments/Squeeze:output:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:2%
#batch_normalization/batchnorm/mul_2╬
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization/batchnorm/ReadVariableOp╤
!batch_normalization/batchnorm/subSub4batch_normalization/batchnorm/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2#
!batch_normalization/batchnorm/subс
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*3
_output_shapes!
:         2%
#batch_normalization/batchnorm/add_1┤
conv3d_1/Conv3D/ReadVariableOpReadVariableOp'conv3d_1_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02 
conv3d_1/Conv3D/ReadVariableOpх
conv3d_1/Conv3DConv3D'batch_normalization/batchnorm/add_1:z:0&conv3d_1/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         *
paddingVALID*
strides	
2
conv3d_1/Conv3Dз
conv3d_1/BiasAdd/ReadVariableOpReadVariableOp(conv3d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv3d_1/BiasAdd/ReadVariableOp░
conv3d_1/BiasAddBiasAddconv3d_1/Conv3D:output:0'conv3d_1/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         2
conv3d_1/BiasAdd|
conv3d_1/EluEluconv3d_1/BiasAdd:output:0*
T0*3
_output_shapes!
:         2
conv3d_1/Elu┼
4batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             26
4batch_normalization_1/moments/mean/reduction_indicesё
"batch_normalization_1/moments/meanMeanconv3d_1/Elu:activations:0=batch_normalization_1/moments/mean/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2$
"batch_normalization_1/moments/mean╩
*batch_normalization_1/moments/StopGradientStopGradient+batch_normalization_1/moments/mean:output:0*
T0**
_output_shapes
:2,
*batch_normalization_1/moments/StopGradientЖ
/batch_normalization_1/moments/SquaredDifferenceSquaredDifferenceconv3d_1/Elu:activations:03batch_normalization_1/moments/StopGradient:output:0*
T0*3
_output_shapes!
:         21
/batch_normalization_1/moments/SquaredDifference═
8batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2:
8batch_normalization_1/moments/variance/reduction_indicesЦ
&batch_normalization_1/moments/varianceMean3batch_normalization_1/moments/SquaredDifference:z:0Abatch_normalization_1/moments/variance/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2(
&batch_normalization_1/moments/variance┼
%batch_normalization_1/moments/SqueezeSqueeze+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2'
%batch_normalization_1/moments/Squeeze═
'batch_normalization_1/moments/Squeeze_1Squeeze/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2)
'batch_normalization_1/moments/Squeeze_1т
+batch_normalization_1/AssignMovingAvg/decayConst*A
_class7
53loc:@batch_normalization_1/AssignMovingAvg/11218759*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2-
+batch_normalization_1/AssignMovingAvg/decay╫
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOp.batch_normalization_1_assignmovingavg_11218759*
_output_shapes
:*
dtype026
4batch_normalization_1/AssignMovingAvg/ReadVariableOp│
)batch_normalization_1/AssignMovingAvg/subSub<batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_1/moments/Squeeze:output:0*
T0*A
_class7
53loc:@batch_normalization_1/AssignMovingAvg/11218759*
_output_shapes
:2+
)batch_normalization_1/AssignMovingAvg/subк
)batch_normalization_1/AssignMovingAvg/mulMul-batch_normalization_1/AssignMovingAvg/sub:z:04batch_normalization_1/AssignMovingAvg/decay:output:0*
T0*A
_class7
53loc:@batch_normalization_1/AssignMovingAvg/11218759*
_output_shapes
:2+
)batch_normalization_1/AssignMovingAvg/mulЙ
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp.batch_normalization_1_assignmovingavg_11218759-batch_normalization_1/AssignMovingAvg/mul:z:05^batch_normalization_1/AssignMovingAvg/ReadVariableOp*A
_class7
53loc:@batch_normalization_1/AssignMovingAvg/11218759*
_output_shapes
 *
dtype02;
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOpш
-batch_normalization_1/AssignMovingAvg_1/decayConst*C
_class9
75loc:@batch_normalization_1/AssignMovingAvg_1/11218765*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2/
-batch_normalization_1/AssignMovingAvg_1/decay▌
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp0batch_normalization_1_assignmovingavg_1_11218765*
_output_shapes
:*
dtype028
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp╜
+batch_normalization_1/AssignMovingAvg_1/subSub>batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_1/moments/Squeeze_1:output:0*
T0*C
_class9
75loc:@batch_normalization_1/AssignMovingAvg_1/11218765*
_output_shapes
:2-
+batch_normalization_1/AssignMovingAvg_1/sub┤
+batch_normalization_1/AssignMovingAvg_1/mulMul/batch_normalization_1/AssignMovingAvg_1/sub:z:06batch_normalization_1/AssignMovingAvg_1/decay:output:0*
T0*C
_class9
75loc:@batch_normalization_1/AssignMovingAvg_1/11218765*
_output_shapes
:2-
+batch_normalization_1/AssignMovingAvg_1/mulХ
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp0batch_normalization_1_assignmovingavg_1_11218765/batch_normalization_1/AssignMovingAvg_1/mul:z:07^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*C
_class9
75loc:@batch_normalization_1/AssignMovingAvg_1/11218765*
_output_shapes
 *
dtype02=
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOpУ
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_1/batchnorm/add/y┌
#batch_normalization_1/batchnorm/addAddV20batch_normalization_1/moments/Squeeze_1:output:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_1/batchnorm/addе
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_1/batchnorm/Rsqrtр
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype024
2batch_normalization_1/batchnorm/mul/ReadVariableOp▌
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_1/batchnorm/mul╪
%batch_normalization_1/batchnorm/mul_1Mulconv3d_1/Elu:activations:0'batch_normalization_1/batchnorm/mul:z:0*
T0*3
_output_shapes!
:         2'
%batch_normalization_1/batchnorm/mul_1╙
%batch_normalization_1/batchnorm/mul_2Mul.batch_normalization_1/moments/Squeeze:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_1/batchnorm/mul_2╘
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype020
.batch_normalization_1/batchnorm/ReadVariableOp┘
#batch_normalization_1/batchnorm/subSub6batch_normalization_1/batchnorm/ReadVariableOp:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_1/batchnorm/subщ
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*3
_output_shapes!
:         2'
%batch_normalization_1/batchnorm/add_1┤
conv3d_2/Conv3D/ReadVariableOpReadVariableOp'conv3d_2_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02 
conv3d_2/Conv3D/ReadVariableOpч
conv3d_2/Conv3DConv3D)batch_normalization_1/batchnorm/add_1:z:0&conv3d_2/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         *
paddingVALID*
strides	
2
conv3d_2/Conv3Dз
conv3d_2/BiasAdd/ReadVariableOpReadVariableOp(conv3d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv3d_2/BiasAdd/ReadVariableOp░
conv3d_2/BiasAddBiasAddconv3d_2/Conv3D:output:0'conv3d_2/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         2
conv3d_2/BiasAdd|
conv3d_2/EluEluconv3d_2/BiasAdd:output:0*
T0*3
_output_shapes!
:         2
conv3d_2/Elu┼
4batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             26
4batch_normalization_2/moments/mean/reduction_indicesё
"batch_normalization_2/moments/meanMeanconv3d_2/Elu:activations:0=batch_normalization_2/moments/mean/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2$
"batch_normalization_2/moments/mean╩
*batch_normalization_2/moments/StopGradientStopGradient+batch_normalization_2/moments/mean:output:0*
T0**
_output_shapes
:2,
*batch_normalization_2/moments/StopGradientЖ
/batch_normalization_2/moments/SquaredDifferenceSquaredDifferenceconv3d_2/Elu:activations:03batch_normalization_2/moments/StopGradient:output:0*
T0*3
_output_shapes!
:         21
/batch_normalization_2/moments/SquaredDifference═
8batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2:
8batch_normalization_2/moments/variance/reduction_indicesЦ
&batch_normalization_2/moments/varianceMean3batch_normalization_2/moments/SquaredDifference:z:0Abatch_normalization_2/moments/variance/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2(
&batch_normalization_2/moments/variance┼
%batch_normalization_2/moments/SqueezeSqueeze+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2'
%batch_normalization_2/moments/Squeeze═
'batch_normalization_2/moments/Squeeze_1Squeeze/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2)
'batch_normalization_2/moments/Squeeze_1т
+batch_normalization_2/AssignMovingAvg/decayConst*A
_class7
53loc:@batch_normalization_2/AssignMovingAvg/11218798*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2-
+batch_normalization_2/AssignMovingAvg/decay╫
4batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOp.batch_normalization_2_assignmovingavg_11218798*
_output_shapes
:*
dtype026
4batch_normalization_2/AssignMovingAvg/ReadVariableOp│
)batch_normalization_2/AssignMovingAvg/subSub<batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_2/moments/Squeeze:output:0*
T0*A
_class7
53loc:@batch_normalization_2/AssignMovingAvg/11218798*
_output_shapes
:2+
)batch_normalization_2/AssignMovingAvg/subк
)batch_normalization_2/AssignMovingAvg/mulMul-batch_normalization_2/AssignMovingAvg/sub:z:04batch_normalization_2/AssignMovingAvg/decay:output:0*
T0*A
_class7
53loc:@batch_normalization_2/AssignMovingAvg/11218798*
_output_shapes
:2+
)batch_normalization_2/AssignMovingAvg/mulЙ
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp.batch_normalization_2_assignmovingavg_11218798-batch_normalization_2/AssignMovingAvg/mul:z:05^batch_normalization_2/AssignMovingAvg/ReadVariableOp*A
_class7
53loc:@batch_normalization_2/AssignMovingAvg/11218798*
_output_shapes
 *
dtype02;
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOpш
-batch_normalization_2/AssignMovingAvg_1/decayConst*C
_class9
75loc:@batch_normalization_2/AssignMovingAvg_1/11218804*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2/
-batch_normalization_2/AssignMovingAvg_1/decay▌
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp0batch_normalization_2_assignmovingavg_1_11218804*
_output_shapes
:*
dtype028
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp╜
+batch_normalization_2/AssignMovingAvg_1/subSub>batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_2/moments/Squeeze_1:output:0*
T0*C
_class9
75loc:@batch_normalization_2/AssignMovingAvg_1/11218804*
_output_shapes
:2-
+batch_normalization_2/AssignMovingAvg_1/sub┤
+batch_normalization_2/AssignMovingAvg_1/mulMul/batch_normalization_2/AssignMovingAvg_1/sub:z:06batch_normalization_2/AssignMovingAvg_1/decay:output:0*
T0*C
_class9
75loc:@batch_normalization_2/AssignMovingAvg_1/11218804*
_output_shapes
:2-
+batch_normalization_2/AssignMovingAvg_1/mulХ
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp0batch_normalization_2_assignmovingavg_1_11218804/batch_normalization_2/AssignMovingAvg_1/mul:z:07^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*C
_class9
75loc:@batch_normalization_2/AssignMovingAvg_1/11218804*
_output_shapes
 *
dtype02=
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOpУ
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_2/batchnorm/add/y┌
#batch_normalization_2/batchnorm/addAddV20batch_normalization_2/moments/Squeeze_1:output:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_2/batchnorm/addе
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_2/batchnorm/Rsqrtр
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype024
2batch_normalization_2/batchnorm/mul/ReadVariableOp▌
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_2/batchnorm/mul╪
%batch_normalization_2/batchnorm/mul_1Mulconv3d_2/Elu:activations:0'batch_normalization_2/batchnorm/mul:z:0*
T0*3
_output_shapes!
:         2'
%batch_normalization_2/batchnorm/mul_1╙
%batch_normalization_2/batchnorm/mul_2Mul.batch_normalization_2/moments/Squeeze:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_2/batchnorm/mul_2╘
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype020
.batch_normalization_2/batchnorm/ReadVariableOp┘
#batch_normalization_2/batchnorm/subSub6batch_normalization_2/batchnorm/ReadVariableOp:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_2/batchnorm/subщ
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*3
_output_shapes!
:         2'
%batch_normalization_2/batchnorm/add_1┤
conv3d_3/Conv3D/ReadVariableOpReadVariableOp'conv3d_3_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02 
conv3d_3/Conv3D/ReadVariableOpч
conv3d_3/Conv3DConv3D)batch_normalization_2/batchnorm/add_1:z:0&conv3d_3/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         *
paddingVALID*
strides	
2
conv3d_3/Conv3Dз
conv3d_3/BiasAdd/ReadVariableOpReadVariableOp(conv3d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv3d_3/BiasAdd/ReadVariableOp░
conv3d_3/BiasAddBiasAddconv3d_3/Conv3D:output:0'conv3d_3/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         2
conv3d_3/BiasAdd|
conv3d_3/EluEluconv3d_3/BiasAdd:output:0*
T0*3
_output_shapes!
:         2
conv3d_3/Elu┼
4batch_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             26
4batch_normalization_3/moments/mean/reduction_indicesё
"batch_normalization_3/moments/meanMeanconv3d_3/Elu:activations:0=batch_normalization_3/moments/mean/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2$
"batch_normalization_3/moments/mean╩
*batch_normalization_3/moments/StopGradientStopGradient+batch_normalization_3/moments/mean:output:0*
T0**
_output_shapes
:2,
*batch_normalization_3/moments/StopGradientЖ
/batch_normalization_3/moments/SquaredDifferenceSquaredDifferenceconv3d_3/Elu:activations:03batch_normalization_3/moments/StopGradient:output:0*
T0*3
_output_shapes!
:         21
/batch_normalization_3/moments/SquaredDifference═
8batch_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2:
8batch_normalization_3/moments/variance/reduction_indicesЦ
&batch_normalization_3/moments/varianceMean3batch_normalization_3/moments/SquaredDifference:z:0Abatch_normalization_3/moments/variance/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2(
&batch_normalization_3/moments/variance┼
%batch_normalization_3/moments/SqueezeSqueeze+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2'
%batch_normalization_3/moments/Squeeze═
'batch_normalization_3/moments/Squeeze_1Squeeze/batch_normalization_3/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2)
'batch_normalization_3/moments/Squeeze_1т
+batch_normalization_3/AssignMovingAvg/decayConst*A
_class7
53loc:@batch_normalization_3/AssignMovingAvg/11218837*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2-
+batch_normalization_3/AssignMovingAvg/decay╫
4batch_normalization_3/AssignMovingAvg/ReadVariableOpReadVariableOp.batch_normalization_3_assignmovingavg_11218837*
_output_shapes
:*
dtype026
4batch_normalization_3/AssignMovingAvg/ReadVariableOp│
)batch_normalization_3/AssignMovingAvg/subSub<batch_normalization_3/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_3/moments/Squeeze:output:0*
T0*A
_class7
53loc:@batch_normalization_3/AssignMovingAvg/11218837*
_output_shapes
:2+
)batch_normalization_3/AssignMovingAvg/subк
)batch_normalization_3/AssignMovingAvg/mulMul-batch_normalization_3/AssignMovingAvg/sub:z:04batch_normalization_3/AssignMovingAvg/decay:output:0*
T0*A
_class7
53loc:@batch_normalization_3/AssignMovingAvg/11218837*
_output_shapes
:2+
)batch_normalization_3/AssignMovingAvg/mulЙ
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp.batch_normalization_3_assignmovingavg_11218837-batch_normalization_3/AssignMovingAvg/mul:z:05^batch_normalization_3/AssignMovingAvg/ReadVariableOp*A
_class7
53loc:@batch_normalization_3/AssignMovingAvg/11218837*
_output_shapes
 *
dtype02;
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOpш
-batch_normalization_3/AssignMovingAvg_1/decayConst*C
_class9
75loc:@batch_normalization_3/AssignMovingAvg_1/11218843*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2/
-batch_normalization_3/AssignMovingAvg_1/decay▌
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpReadVariableOp0batch_normalization_3_assignmovingavg_1_11218843*
_output_shapes
:*
dtype028
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp╜
+batch_normalization_3/AssignMovingAvg_1/subSub>batch_normalization_3/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_3/moments/Squeeze_1:output:0*
T0*C
_class9
75loc:@batch_normalization_3/AssignMovingAvg_1/11218843*
_output_shapes
:2-
+batch_normalization_3/AssignMovingAvg_1/sub┤
+batch_normalization_3/AssignMovingAvg_1/mulMul/batch_normalization_3/AssignMovingAvg_1/sub:z:06batch_normalization_3/AssignMovingAvg_1/decay:output:0*
T0*C
_class9
75loc:@batch_normalization_3/AssignMovingAvg_1/11218843*
_output_shapes
:2-
+batch_normalization_3/AssignMovingAvg_1/mulХ
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp0batch_normalization_3_assignmovingavg_1_11218843/batch_normalization_3/AssignMovingAvg_1/mul:z:07^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp*C
_class9
75loc:@batch_normalization_3/AssignMovingAvg_1/11218843*
_output_shapes
 *
dtype02=
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOpУ
%batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_3/batchnorm/add/y┌
#batch_normalization_3/batchnorm/addAddV20batch_normalization_3/moments/Squeeze_1:output:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_3/batchnorm/addе
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_3/batchnorm/Rsqrtр
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype024
2batch_normalization_3/batchnorm/mul/ReadVariableOp▌
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:0:batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_3/batchnorm/mul╪
%batch_normalization_3/batchnorm/mul_1Mulconv3d_3/Elu:activations:0'batch_normalization_3/batchnorm/mul:z:0*
T0*3
_output_shapes!
:         2'
%batch_normalization_3/batchnorm/mul_1╙
%batch_normalization_3/batchnorm/mul_2Mul.batch_normalization_3/moments/Squeeze:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_3/batchnorm/mul_2╘
.batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype020
.batch_normalization_3/batchnorm/ReadVariableOp┘
#batch_normalization_3/batchnorm/subSub6batch_normalization_3/batchnorm/ReadVariableOp:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_3/batchnorm/subщ
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*3
_output_shapes!
:         2'
%batch_normalization_3/batchnorm/add_1ю
average_pooling3d/AvgPool3D	AvgPool3D)batch_normalization_3/batchnorm/add_1:z:0*
T0*3
_output_shapes!
:         *
ksize	
*
paddingVALID*
strides	
2
average_pooling3d/AvgPool3Do
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
flatten/ConstЮ
flatten/ReshapeReshape$average_pooling3d/AvgPool3D:output:0flatten/Const:output:0*
T0*(
_output_shapes
:         А
2
flatten/Reshapes
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU╒?2
dropout/dropout/ConstЮ
dropout/dropout/MulMulflatten/Reshape:output:0dropout/dropout/Const:output:0*
T0*(
_output_shapes
:         А
2
dropout/dropout/Mulv
dropout/dropout/ShapeShapeflatten/Reshape:output:0*
T0*
_output_shapes
:2
dropout/dropout/Shape═
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*(
_output_shapes
:         А
*
dtype02.
,dropout/dropout/random_uniform/RandomUniformЕ
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠>2 
dropout/dropout/GreaterEqual/y▀
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А
2
dropout/dropout/GreaterEqualШ
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А
2
dropout/dropout/CastЫ
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:         А
2
dropout/dropout/Mul_1t
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis╖
concatenate/concatConcatV2dropout/dropout/Mul_1:z:0inputs_1 concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:         Г
2
concatenate/concatд
layer1/MatMul/ReadVariableOpReadVariableOp%layer1_matmul_readvariableop_resource* 
_output_shapes
:
Г
╚*
dtype02
layer1/MatMul/ReadVariableOpЮ
layer1/MatMulMatMulconcatenate/concat:output:0$layer1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
layer1/MatMulв
layer1/BiasAdd/ReadVariableOpReadVariableOp&layer1_biasadd_readvariableop_resource*
_output_shapes	
:╚*
dtype02
layer1/BiasAdd/ReadVariableOpЮ
layer1/BiasAddBiasAddlayer1/MatMul:product:0%layer1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
layer1/BiasAddk

layer1/EluElulayer1/BiasAdd:output:0*
T0*(
_output_shapes
:         ╚2

layer1/Eluw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout_1/dropout/Constд
dropout_1/dropout/MulMullayer1/Elu:activations:0 dropout_1/dropout/Const:output:0*
T0*(
_output_shapes
:         ╚2
dropout_1/dropout/Mulz
dropout_1/dropout/ShapeShapelayer1/Elu:activations:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shape╙
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*(
_output_shapes
:         ╚*
dtype020
.dropout_1/dropout/random_uniform/RandomUniformЙ
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2"
 dropout_1/dropout/GreaterEqual/yч
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ╚2 
dropout_1/dropout/GreaterEqualЮ
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ╚2
dropout_1/dropout/Castг
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*(
_output_shapes
:         ╚2
dropout_1/dropout/Mul_1г
layer2/MatMul/ReadVariableOpReadVariableOp%layer2_matmul_readvariableop_resource*
_output_shapes
:	╚*
dtype02
layer2/MatMul/ReadVariableOpЭ
layer2/MatMulMatMuldropout_1/dropout/Mul_1:z:0$layer2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
layer2/MatMulб
layer2/BiasAdd/ReadVariableOpReadVariableOp&layer2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
layer2/BiasAdd/ReadVariableOpЭ
layer2/BiasAddBiasAddlayer2/MatMul:product:0%layer2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
layer2/BiasAddj

layer2/EluElulayer2/BiasAdd:output:0*
T0*'
_output_shapes
:         2

layer2/Eluw
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?2
dropout_2/dropout/Constг
dropout_2/dropout/MulMullayer2/Elu:activations:0 dropout_2/dropout/Const:output:0*
T0*'
_output_shapes
:         2
dropout_2/dropout/Mulz
dropout_2/dropout/ShapeShapelayer2/Elu:activations:0*
T0*
_output_shapes
:2
dropout_2/dropout/Shape╥
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype020
.dropout_2/dropout/random_uniform/RandomUniformЙ
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2"
 dropout_2/dropout/GreaterEqual/yц
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         2 
dropout_2/dropout/GreaterEqualЭ
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         2
dropout_2/dropout/Castв
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*'
_output_shapes
:         2
dropout_2/dropout/Mul_1Я
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense/MatMul/ReadVariableOpЪ
dense/MatMulMatMuldropout_2/dropout/Mul_1:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense/MatMulЮ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOpЩ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense/BiasAdds
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense/Sigmoid╔
IdentityIdentitydense/Sigmoid:y:08^batch_normalization/AssignMovingAvg/AssignSubVariableOp:^batch_normalization/AssignMovingAvg_1/AssignSubVariableOp:^batch_normalization_1/AssignMovingAvg/AssignSubVariableOp<^batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp:^batch_normalization_2/AssignMovingAvg/AssignSubVariableOp<^batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp:^batch_normalization_3/AssignMovingAvg/AssignSubVariableOp<^batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*└
_input_shapesо
л:         н:         ::::::::::::::::::::::::::::::2r
7batch_normalization/AssignMovingAvg/AssignSubVariableOp7batch_normalization/AssignMovingAvg/AssignSubVariableOp2v
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp2v
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOp9batch_normalization_1/AssignMovingAvg/AssignSubVariableOp2z
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp2v
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOp9batch_normalization_2/AssignMovingAvg/AssignSubVariableOp2z
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp2v
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOp9batch_normalization_3/AssignMovingAvg/AssignSubVariableOp2z
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp:^ Z
4
_output_shapes"
 :         н
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
¤
H
,__inference_dropout_1_layer_call_fn_11220067

inputs
identityз
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:         ╚* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_1_layer_call_and_return_conditional_losses_112180052
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ╚2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ╚:P L
(
_output_shapes
:         ╚
 
_user_specified_nameinputs
ъ
м
D__inference_layer1_layer_call_and_return_conditional_losses_11220031

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Г
╚*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:╚*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2	
BiasAddV
EluEluBiasAdd:output:0*
T0*(
_output_shapes
:         ╚2
Eluf
IdentityIdentityElu:activations:0*
T0*(
_output_shapes
:         ╚2

Identity"
identityIdentity:output:0*/
_input_shapes
:         Г
:::P L
(
_output_shapes
:         Г

 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
о	
л
8__inference_batch_normalization_1_layer_call_fn_11219474

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallг
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*N
_output_shapes<
::8                                    *$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_112171312
StatefulPartitionedCall╡
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8                                    2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:8                                    ::::22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8                                    
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
░	
л
8__inference_batch_normalization_3_layer_call_fn_11219887

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallе
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*N
_output_shapes<
::8                                    *&
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*\
fWRU
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_112174882
StatefulPartitionedCall╡
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8                                    2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:8                                    ::::22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8                                    
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ў
}
(__inference_dense_layer_call_fn_11220134

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCall╘
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_112180862
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Ъ
А
+__inference_conv3d_2_layer_call_fn_11217197

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCall■
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*N
_output_shapes<
::8                                    *$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_conv3d_2_layer_call_and_return_conditional_losses_112171872
StatefulPartitionedCall╡
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8                                    2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8                                    ::22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8                                    
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Ъ
А
+__inference_conv3d_1_layer_call_fn_11217035

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCall■
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*N
_output_shapes<
::8                                    *$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_conv3d_1_layer_call_and_return_conditional_losses_112170252
StatefulPartitionedCall╡
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8                                    2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8                                    ::22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8                                    
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Е
c
*__inference_dropout_layer_call_fn_11220002

inputs
identityИвStatefulPartitionedCall╜
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:         А
* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_112179272
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А
2

Identity"
identityIdentity:output:0*'
_input_shapes
:         А
22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А

 
_user_specified_nameinputs
╟+
╨
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_11217748

inputs
assignmovingavg_11217723
assignmovingavg_1_11217729)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpЩ
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indicesЫ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/meanИ
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
:2
moments/StopGradient░
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*3
_output_shapes!
:         2
moments/SquaredDifferenceб
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices╛
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/varianceГ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/SqueezeЛ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1а
AssignMovingAvg/decayConst*+
_class!
loc:@AssignMovingAvg/11217723*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayХ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_11217723*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp┼
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*+
_class!
loc:@AssignMovingAvg/11217723*
_output_shapes
:2
AssignMovingAvg/sub╝
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg/11217723*
_output_shapes
:2
AssignMovingAvg/mulЕ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_11217723AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/11217723*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpж
AssignMovingAvg_1/decayConst*-
_class#
!loc:@AssignMovingAvg_1/11217729*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayЫ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_11217729*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╧
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/11217729*
_output_shapes
:2
AssignMovingAvg_1/sub╞
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/11217729*
_output_shapes
:2
AssignMovingAvg_1/mulС
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_11217729AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/11217729*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulВ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:         2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subС
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:         2
batchnorm/add_1┴
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*3
_output_shapes!
:         2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:         ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:[ W
3
_output_shapes!
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
║
s
I__inference_concatenate_layer_call_and_return_conditional_losses_11217952

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisА
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:         Г
2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:         Г
2

Identity"
identityIdentity:output:0*:
_input_shapes)
':         А
:         :P L
(
_output_shapes
:         А

 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs
ъ
м
D__inference_layer1_layer_call_and_return_conditional_losses_11217972

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Г
╚*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:╚*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2	
BiasAddV
EluEluBiasAdd:output:0*
T0*(
_output_shapes
:         ╚2
Eluf
IdentityIdentityElu:activations:0*
T0*(
_output_shapes
:         ╚2

Identity"
identityIdentity:output:0*/
_input_shapes
:         Г
:::P L
(
_output_shapes
:         Г

 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Д
о
F__inference_conv3d_1_layer_call_and_return_conditional_losses_11217025

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identityИЩ
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype02
Conv3D/ReadVariableOp─
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8                                    *
paddingVALID*
strides	
2
Conv3DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpз
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8                                    2	
BiasAdd|
EluEluBiasAdd:output:0*
T0*N
_output_shapes<
::8                                    2
EluМ
IdentityIdentityElu:activations:0*
T0*N
_output_shapes<
::8                                    2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8                                    :::v r
N
_output_shapes<
::8                                    
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
└
й
6__inference_batch_normalization_layer_call_fn_11219369

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallИ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*3
_output_shapes!
:         *&
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_112175762
StatefulPartitionedCallЪ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:         2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:         ::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
┬
л
8__inference_batch_normalization_3_layer_call_fn_11219956

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallИ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*3
_output_shapes!
:         *$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*\
fWRU
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_112178442
StatefulPartitionedCallЪ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:         2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:         ::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
∙
F
*__inference_dropout_layer_call_fn_11220007

inputs
identityе
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:         А
* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_112179322
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А
2

Identity"
identityIdentity:output:0*'
_input_shapes
:         А
:P L
(
_output_shapes
:         А

 
_user_specified_nameinputs
ц
л
C__inference_dense_layer_call_and_return_conditional_losses_11220125

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:         2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         :::O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
щ,
╨
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_11217131

inputs
assignmovingavg_11217106
assignmovingavg_1_11217112)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpЩ
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indicesЫ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/meanИ
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
:2
moments/StopGradient╦
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*N
_output_shapes<
::8                                    2
moments/SquaredDifferenceб
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices╛
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/varianceГ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/SqueezeЛ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1а
AssignMovingAvg/decayConst*+
_class!
loc:@AssignMovingAvg/11217106*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayХ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_11217106*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp┼
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*+
_class!
loc:@AssignMovingAvg/11217106*
_output_shapes
:2
AssignMovingAvg/sub╝
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg/11217106*
_output_shapes
:2
AssignMovingAvg/mulЕ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_11217106AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/11217106*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpж
AssignMovingAvg_1/decayConst*-
_class#
!loc:@AssignMovingAvg_1/11217112*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayЫ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_11217112*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╧
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/11217112*
_output_shapes
:2
AssignMovingAvg_1/sub╞
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/11217112*
_output_shapes
:2
AssignMovingAvg_1/mulС
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_11217112AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/11217112*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulЭ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8                                    2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subм
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8                                    2
batchnorm/add_1▄
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*N
_output_shapes<
::8                                    2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:8                                    ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:v r
N
_output_shapes<
::8                                    
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
╩
e
G__inference_dropout_2_layer_call_and_return_conditional_losses_11220104

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:         2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
К
d
E__inference_dropout_layer_call_and_return_conditional_losses_11219992

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU╒?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         А
2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╡
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А
*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠>2
dropout/GreaterEqual/y┐
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А
2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А
2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         А
2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         А
2

Identity"
identityIdentity:output:0*'
_input_shapes
:         А
:P L
(
_output_shapes
:         А

 
_user_specified_nameinputs
Ж

м
D__inference_conv3d_layer_call_and_return_conditional_losses_11216863

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identityИЪ
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource*+
_output_shapes
:н*
dtype02
Conv3D/ReadVariableOp─
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8                                    *
paddingVALID*
strides	
2
Conv3DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpз
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8                                    2	
BiasAddЛ
IdentityIdentityBiasAdd:output:0*
T0*N
_output_shapes<
::8                                    2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:9                                    н:::w s
O
_output_shapes=
;:9                                    н
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
╩
P
4__inference_average_pooling3d_layer_call_fn_11217511

inputs
identity▐
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*W
_output_shapesE
C:A                                             * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_average_pooling3d_layer_call_and_return_conditional_losses_112175052
PartitionedCallЬ
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A                                             2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:A                                             : {
W
_output_shapesE
C:A                                             
 
_user_specified_nameinputs
МY
В
C__inference_model_layer_call_and_return_conditional_losses_11218419

inputs
inputs_1
conv3d_11218341
conv3d_11218343 
batch_normalization_11218346 
batch_normalization_11218348 
batch_normalization_11218350 
batch_normalization_11218352
conv3d_1_11218355
conv3d_1_11218357"
batch_normalization_1_11218360"
batch_normalization_1_11218362"
batch_normalization_1_11218364"
batch_normalization_1_11218366
conv3d_2_11218369
conv3d_2_11218371"
batch_normalization_2_11218374"
batch_normalization_2_11218376"
batch_normalization_2_11218378"
batch_normalization_2_11218380
conv3d_3_11218383
conv3d_3_11218385"
batch_normalization_3_11218388"
batch_normalization_3_11218390"
batch_normalization_3_11218392"
batch_normalization_3_11218394
layer1_11218401
layer1_11218403
layer2_11218407
layer2_11218409
dense_11218413
dense_11218415
identityИв+batch_normalization/StatefulPartitionedCallв-batch_normalization_1/StatefulPartitionedCallв-batch_normalization_2/StatefulPartitionedCallв-batch_normalization_3/StatefulPartitionedCallвconv3d/StatefulPartitionedCallв conv3d_1/StatefulPartitionedCallв conv3d_2/StatefulPartitionedCallв conv3d_3/StatefulPartitionedCallвdense/StatefulPartitionedCallвlayer1/StatefulPartitionedCallвlayer2/StatefulPartitionedCall¤
conv3d/StatefulPartitionedCallStatefulPartitionedCallinputsconv3d_11218341conv3d_11218343*
Tin
2*
Tout
2*3
_output_shapes!
:         *$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_conv3d_layer_call_and_return_conditional_losses_112168632 
conv3d/StatefulPartitionedCallЯ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv3d/StatefulPartitionedCall:output:0batch_normalization_11218346batch_normalization_11218348batch_normalization_11218350batch_normalization_11218352*
Tin	
2*
Tout
2*3
_output_shapes!
:         *&
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_112175762-
+batch_normalization/StatefulPartitionedCall╡
 conv3d_1/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0conv3d_1_11218355conv3d_1_11218357*
Tin
2*
Tout
2*3
_output_shapes!
:         *$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_conv3d_1_layer_call_and_return_conditional_losses_112170252"
 conv3d_1/StatefulPartitionedCallп
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv3d_1/StatefulPartitionedCall:output:0batch_normalization_1_11218360batch_normalization_1_11218362batch_normalization_1_11218364batch_normalization_1_11218366*
Tin	
2*
Tout
2*3
_output_shapes!
:         *&
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_112176722/
-batch_normalization_1/StatefulPartitionedCall╖
 conv3d_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0conv3d_2_11218369conv3d_2_11218371*
Tin
2*
Tout
2*3
_output_shapes!
:         *$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_conv3d_2_layer_call_and_return_conditional_losses_112171872"
 conv3d_2/StatefulPartitionedCallп
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv3d_2/StatefulPartitionedCall:output:0batch_normalization_2_11218374batch_normalization_2_11218376batch_normalization_2_11218378batch_normalization_2_11218380*
Tin	
2*
Tout
2*3
_output_shapes!
:         *&
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_112177682/
-batch_normalization_2/StatefulPartitionedCall╖
 conv3d_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0conv3d_3_11218383conv3d_3_11218385*
Tin
2*
Tout
2*3
_output_shapes!
:         *$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_conv3d_3_layer_call_and_return_conditional_losses_112173492"
 conv3d_3/StatefulPartitionedCallп
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv3d_3/StatefulPartitionedCall:output:0batch_normalization_3_11218388batch_normalization_3_11218390batch_normalization_3_11218392batch_normalization_3_11218394*
Tin	
2*
Tout
2*3
_output_shapes!
:         *&
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*\
fWRU
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_112178642/
-batch_normalization_3/StatefulPartitionedCallО
!average_pooling3d/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:         * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_average_pooling3d_layer_call_and_return_conditional_losses_112175052#
!average_pooling3d/PartitionedCall┘
flatten/PartitionedCallPartitionedCall*average_pooling3d/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:         А
* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_112179072
flatten/PartitionedCall╧
dropout/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:         А
* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_112179322
dropout/PartitionedCallц
concatenate/PartitionedCallPartitionedCall dropout/PartitionedCall:output:0inputs_1*
Tin
2*
Tout
2*(
_output_shapes
:         Г
* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_concatenate_layer_call_and_return_conditional_losses_112179522
concatenate/PartitionedCallР
layer1/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0layer1_11218401layer1_11218403*
Tin
2*
Tout
2*(
_output_shapes
:         ╚*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_layer1_layer_call_and_return_conditional_losses_112179722 
layer1/StatefulPartitionedCall▄
dropout_1/PartitionedCallPartitionedCall'layer1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:         ╚* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_1_layer_call_and_return_conditional_losses_112180052
dropout_1/PartitionedCallН
layer2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0layer2_11218407layer2_11218409*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_layer2_layer_call_and_return_conditional_losses_112180292 
layer2/StatefulPartitionedCall█
dropout_2/PartitionedCallPartitionedCall'layer2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_2_layer_call_and_return_conditional_losses_112180622
dropout_2/PartitionedCallИ
dense/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_11218413dense_11218415*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_112180862
dense/StatefulPartitionedCallд
IdentityIdentity&dense/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall^conv3d/StatefulPartitionedCall!^conv3d_1/StatefulPartitionedCall!^conv3d_2/StatefulPartitionedCall!^conv3d_3/StatefulPartitionedCall^dense/StatefulPartitionedCall^layer1/StatefulPartitionedCall^layer2/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*└
_input_shapesо
л:         н:         ::::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2@
conv3d/StatefulPartitionedCallconv3d/StatefulPartitionedCall2D
 conv3d_1/StatefulPartitionedCall conv3d_1/StatefulPartitionedCall2D
 conv3d_2/StatefulPartitionedCall conv3d_2/StatefulPartitionedCall2D
 conv3d_3/StatefulPartitionedCall conv3d_3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2@
layer2/StatefulPartitionedCalllayer2/StatefulPartitionedCall:\ X
4
_output_shapes"
 :         н
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
·
┬
(__inference_model_layer_call_fn_11218334
input_1
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28
identityИвStatefulPartitionedCall╓
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28*+
Tin$
"2 *
Tout
2*'
_output_shapes
:         *8
_read_only_resource_inputs
	*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_112182712
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*└
_input_shapesо
л:         н:         ::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
4
_output_shapes"
 :         н
!
_user_specified_name	input_1:PL
'
_output_shapes
:         
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
┬
л
8__inference_batch_normalization_1_layer_call_fn_11219556

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallИ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*3
_output_shapes!
:         *$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_112176522
StatefulPartitionedCallЪ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:         2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:         ::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
║
Ц
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_11219661

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИТ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulВ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:         2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subС
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:         2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0*
T0*3
_output_shapes!
:         2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:         :::::[ W
3
_output_shapes!
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ф
м
D__inference_layer2_layer_call_and_return_conditional_losses_11220078

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	╚*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddU
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:         2
Elue
IdentityIdentityElu:activations:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ╚:::P L
(
_output_shapes
:         ╚
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Г
f
G__inference_dropout_2_layer_call_and_return_conditional_losses_11220099

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape┤
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2
dropout/GreaterEqual/y╛
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
М
f
G__inference_dropout_1_layer_call_and_return_conditional_losses_11220052

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         ╚2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╡
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         ╚*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/y┐
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ╚2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ╚2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         ╚2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         ╚2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ╚:P L
(
_output_shapes
:         ╚
 
_user_specified_nameinputs
─
л
8__inference_batch_normalization_1_layer_call_fn_11219569

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallК
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*3
_output_shapes!
:         *&
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_112176722
StatefulPartitionedCallЪ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:         2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:         ::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
╕
Ф
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_11219343

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИТ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulВ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:         2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subС
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:         2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0*
T0*3
_output_shapes!
:         2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:         :::::[ W
3
_output_shapes!
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Г
f
G__inference_dropout_2_layer_call_and_return_conditional_losses_11218057

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape┤
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2
dropout/GreaterEqual/y╛
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
┼
a
E__inference_flatten_layer_call_and_return_conditional_losses_11219975

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         А
2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         А
2

Identity"
identityIdentity:output:0*2
_input_shapes!
:         :[ W
3
_output_shapes!
:         
 
_user_specified_nameinputs
╟+
╨
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_11219923

inputs
assignmovingavg_11219898
assignmovingavg_1_11219904)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpЩ
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indicesЫ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/meanИ
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
:2
moments/StopGradient░
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*3
_output_shapes!
:         2
moments/SquaredDifferenceб
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices╛
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/varianceГ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/SqueezeЛ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1а
AssignMovingAvg/decayConst*+
_class!
loc:@AssignMovingAvg/11219898*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayХ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_11219898*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp┼
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*+
_class!
loc:@AssignMovingAvg/11219898*
_output_shapes
:2
AssignMovingAvg/sub╝
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg/11219898*
_output_shapes
:2
AssignMovingAvg/mulЕ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_11219898AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/11219898*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpж
AssignMovingAvg_1/decayConst*-
_class#
!loc:@AssignMovingAvg_1/11219904*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayЫ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_11219904*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╧
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/11219904*
_output_shapes
:2
AssignMovingAvg_1/sub╞
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/11219904*
_output_shapes
:2
AssignMovingAvg_1/mulС
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_11219904AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/11219904*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulВ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:         2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subС
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:         2
batchnorm/add_1┴
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*3
_output_shapes!
:         2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:         ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:[ W
3
_output_shapes!
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ц
л
C__inference_dense_layer_call_and_return_conditional_losses_11218086

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:         2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         :::O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
└
Ф
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_11217002

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИТ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulЭ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8                                    2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subм
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8                                    2
batchnorm/add_1О
IdentityIdentitybatchnorm/add_1:z:0*
T0*N
_output_shapes<
::8                                    2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:8                                    :::::v r
N
_output_shapes<
::8                                    
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
П
F
*__inference_flatten_layer_call_fn_11219980

inputs
identityе
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:         А
* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_112179072
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А
2

Identity"
identityIdentity:output:0*2
_input_shapes!
:         :[ W
3
_output_shapes!
:         
 
_user_specified_nameinputs
ци
И$
!__inference__traced_save_11220405
file_prefix,
(savev2_conv3d_kernel_read_readvariableop*
&savev2_conv3d_bias_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop.
*savev2_conv3d_1_kernel_read_readvariableop,
(savev2_conv3d_1_bias_read_readvariableop:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop.
*savev2_conv3d_2_kernel_read_readvariableop,
(savev2_conv3d_2_bias_read_readvariableop:
6savev2_batch_normalization_2_gamma_read_readvariableop9
5savev2_batch_normalization_2_beta_read_readvariableop@
<savev2_batch_normalization_2_moving_mean_read_readvariableopD
@savev2_batch_normalization_2_moving_variance_read_readvariableop.
*savev2_conv3d_3_kernel_read_readvariableop,
(savev2_conv3d_3_bias_read_readvariableop:
6savev2_batch_normalization_3_gamma_read_readvariableop9
5savev2_batch_normalization_3_beta_read_readvariableop@
<savev2_batch_normalization_3_moving_mean_read_readvariableopD
@savev2_batch_normalization_3_moving_variance_read_readvariableop,
(savev2_layer1_kernel_read_readvariableop*
&savev2_layer1_bias_read_readvariableop,
(savev2_layer2_kernel_read_readvariableop*
&savev2_layer2_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop3
/savev2_adam_conv3d_kernel_m_read_readvariableop1
-savev2_adam_conv3d_bias_m_read_readvariableop?
;savev2_adam_batch_normalization_gamma_m_read_readvariableop>
:savev2_adam_batch_normalization_beta_m_read_readvariableop5
1savev2_adam_conv3d_1_kernel_m_read_readvariableop3
/savev2_adam_conv3d_1_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_1_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_1_beta_m_read_readvariableop5
1savev2_adam_conv3d_2_kernel_m_read_readvariableop3
/savev2_adam_conv3d_2_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_2_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_2_beta_m_read_readvariableop5
1savev2_adam_conv3d_3_kernel_m_read_readvariableop3
/savev2_adam_conv3d_3_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_3_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_3_beta_m_read_readvariableop3
/savev2_adam_layer1_kernel_m_read_readvariableop1
-savev2_adam_layer1_bias_m_read_readvariableop3
/savev2_adam_layer2_kernel_m_read_readvariableop1
-savev2_adam_layer2_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop3
/savev2_adam_conv3d_kernel_v_read_readvariableop1
-savev2_adam_conv3d_bias_v_read_readvariableop?
;savev2_adam_batch_normalization_gamma_v_read_readvariableop>
:savev2_adam_batch_normalization_beta_v_read_readvariableop5
1savev2_adam_conv3d_1_kernel_v_read_readvariableop3
/savev2_adam_conv3d_1_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_1_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_1_beta_v_read_readvariableop5
1savev2_adam_conv3d_2_kernel_v_read_readvariableop3
/savev2_adam_conv3d_2_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_2_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_2_beta_v_read_readvariableop5
1savev2_adam_conv3d_3_kernel_v_read_readvariableop3
/savev2_adam_conv3d_3_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_3_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_3_beta_v_read_readvariableop3
/savev2_adam_layer1_kernel_v_read_readvariableop1
-savev2_adam_layer1_bias_v_read_readvariableop3
/savev2_adam_layer2_kernel_v_read_readvariableop1
-savev2_adam_layer2_bias_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop
savev2_1_const

identity_1ИвMergeV2CheckpointsвSaveV2вSaveV2_1П
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
ConstН
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_6c761ca2c1ce4fc3b8f897abc733b15b/part2	
Const_1Л
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardж
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename─-
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:Q*
dtype0*╓,
value╠,B╔,QB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_namesн
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:Q*
dtype0*╖
valueнBкQB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices╝"
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv3d_kernel_read_readvariableop&savev2_conv3d_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop*savev2_conv3d_1_kernel_read_readvariableop(savev2_conv3d_1_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop*savev2_conv3d_2_kernel_read_readvariableop(savev2_conv3d_2_bias_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop*savev2_conv3d_3_kernel_read_readvariableop(savev2_conv3d_3_bias_read_readvariableop6savev2_batch_normalization_3_gamma_read_readvariableop5savev2_batch_normalization_3_beta_read_readvariableop<savev2_batch_normalization_3_moving_mean_read_readvariableop@savev2_batch_normalization_3_moving_variance_read_readvariableop(savev2_layer1_kernel_read_readvariableop&savev2_layer1_bias_read_readvariableop(savev2_layer2_kernel_read_readvariableop&savev2_layer2_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop/savev2_adam_conv3d_kernel_m_read_readvariableop-savev2_adam_conv3d_bias_m_read_readvariableop;savev2_adam_batch_normalization_gamma_m_read_readvariableop:savev2_adam_batch_normalization_beta_m_read_readvariableop1savev2_adam_conv3d_1_kernel_m_read_readvariableop/savev2_adam_conv3d_1_bias_m_read_readvariableop=savev2_adam_batch_normalization_1_gamma_m_read_readvariableop<savev2_adam_batch_normalization_1_beta_m_read_readvariableop1savev2_adam_conv3d_2_kernel_m_read_readvariableop/savev2_adam_conv3d_2_bias_m_read_readvariableop=savev2_adam_batch_normalization_2_gamma_m_read_readvariableop<savev2_adam_batch_normalization_2_beta_m_read_readvariableop1savev2_adam_conv3d_3_kernel_m_read_readvariableop/savev2_adam_conv3d_3_bias_m_read_readvariableop=savev2_adam_batch_normalization_3_gamma_m_read_readvariableop<savev2_adam_batch_normalization_3_beta_m_read_readvariableop/savev2_adam_layer1_kernel_m_read_readvariableop-savev2_adam_layer1_bias_m_read_readvariableop/savev2_adam_layer2_kernel_m_read_readvariableop-savev2_adam_layer2_bias_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop/savev2_adam_conv3d_kernel_v_read_readvariableop-savev2_adam_conv3d_bias_v_read_readvariableop;savev2_adam_batch_normalization_gamma_v_read_readvariableop:savev2_adam_batch_normalization_beta_v_read_readvariableop1savev2_adam_conv3d_1_kernel_v_read_readvariableop/savev2_adam_conv3d_1_bias_v_read_readvariableop=savev2_adam_batch_normalization_1_gamma_v_read_readvariableop<savev2_adam_batch_normalization_1_beta_v_read_readvariableop1savev2_adam_conv3d_2_kernel_v_read_readvariableop/savev2_adam_conv3d_2_bias_v_read_readvariableop=savev2_adam_batch_normalization_2_gamma_v_read_readvariableop<savev2_adam_batch_normalization_2_beta_v_read_readvariableop1savev2_adam_conv3d_3_kernel_v_read_readvariableop/savev2_adam_conv3d_3_bias_v_read_readvariableop=savev2_adam_batch_normalization_3_gamma_v_read_readvariableop<savev2_adam_batch_normalization_3_beta_v_read_readvariableop/savev2_adam_layer1_kernel_v_read_readvariableop-savev2_adam_layer1_bias_v_read_readvariableop/savev2_adam_layer2_kernel_v_read_readvariableop-savev2_adam_layer2_bias_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *_
dtypesU
S2Q	2
SaveV2Г
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shardм
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1в
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_namesО
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices╧
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1у
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesм
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

IdentityБ

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*╓
_input_shapes─
┴: :н::::::::::::::::::::::::
Г
╚:╚:	╚:::: : : : : : : :н::::::::::::::::
Г
╚:╚:	╚::::н::::::::::::::::
Г
╚:╚:	╚:::: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:1-
+
_output_shapes
:н: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::0,
*
_output_shapes
:: 

_output_shapes
:: 	

_output_shapes
:: 


_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::0,
*
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::0,
*
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::&"
 
_output_shapes
:
Г
╚:!

_output_shapes	
:╚:%!

_output_shapes
:	╚: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :1&-
+
_output_shapes
:н: '

_output_shapes
:: (

_output_shapes
:: )

_output_shapes
::0*,
*
_output_shapes
:: +

_output_shapes
:: ,

_output_shapes
:: -

_output_shapes
::0.,
*
_output_shapes
:: /

_output_shapes
:: 0

_output_shapes
:: 1

_output_shapes
::02,
*
_output_shapes
:: 3

_output_shapes
:: 4

_output_shapes
:: 5

_output_shapes
::&6"
 
_output_shapes
:
Г
╚:!7

_output_shapes	
:╚:%8!

_output_shapes
:	╚: 9

_output_shapes
::$: 

_output_shapes

:: ;

_output_shapes
::1<-
+
_output_shapes
:н: =

_output_shapes
:: >

_output_shapes
:: ?

_output_shapes
::0@,
*
_output_shapes
:: A

_output_shapes
:: B

_output_shapes
:: C

_output_shapes
::0D,
*
_output_shapes
:: E

_output_shapes
:: F

_output_shapes
:: G

_output_shapes
::0H,
*
_output_shapes
:: I

_output_shapes
:: J

_output_shapes
:: K

_output_shapes
::&L"
 
_output_shapes
:
Г
╚:!M

_output_shapes	
:╚:%N!

_output_shapes
:	╚: O

_output_shapes
::$P 

_output_shapes

:: Q

_output_shapes
::R

_output_shapes
: 
┬
Ц
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_11217164

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИТ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulЭ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8                                    2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subм
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8                                    2
batchnorm/add_1О
IdentityIdentitybatchnorm/add_1:z:0*
T0*N
_output_shapes<
::8                                    2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:8                                    :::::v r
N
_output_shapes<
::8                                    
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
╠
k
O__inference_average_pooling3d_layer_call_and_return_conditional_losses_11217505

inputs
identity╦
	AvgPool3D	AvgPool3Dinputs*
T0*W
_output_shapesE
C:A                                             *
ksize	
*
paddingVALID*
strides	
2
	AvgPool3DЦ
IdentityIdentityAvgPool3D:output:0*
T0*W
_output_shapesE
C:A                                             2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:A                                             : {
W
_output_shapesE
C:A                                             
 
_user_specified_nameinputs
╕
Ф
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_11217576

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИТ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulВ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:         2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subС
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:         2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0*
T0*3
_output_shapes!
:         2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:         :::::[ W
3
_output_shapes!
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Д
о
F__inference_conv3d_2_layer_call_and_return_conditional_losses_11217187

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identityИЩ
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype02
Conv3D/ReadVariableOp─
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8                                    *
paddingVALID*
strides	
2
Conv3DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpз
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8                                    2	
BiasAdd|
EluEluBiasAdd:output:0*
T0*N
_output_shapes<
::8                                    2
EluМ
IdentityIdentityElu:activations:0*
T0*N
_output_shapes<
::8                                    2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8                                    :::v r
N
_output_shapes<
::8                                    
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
╥]
ь
C__inference_model_layer_call_and_return_conditional_losses_11218271

inputs
inputs_1
conv3d_11218193
conv3d_11218195 
batch_normalization_11218198 
batch_normalization_11218200 
batch_normalization_11218202 
batch_normalization_11218204
conv3d_1_11218207
conv3d_1_11218209"
batch_normalization_1_11218212"
batch_normalization_1_11218214"
batch_normalization_1_11218216"
batch_normalization_1_11218218
conv3d_2_11218221
conv3d_2_11218223"
batch_normalization_2_11218226"
batch_normalization_2_11218228"
batch_normalization_2_11218230"
batch_normalization_2_11218232
conv3d_3_11218235
conv3d_3_11218237"
batch_normalization_3_11218240"
batch_normalization_3_11218242"
batch_normalization_3_11218244"
batch_normalization_3_11218246
layer1_11218253
layer1_11218255
layer2_11218259
layer2_11218261
dense_11218265
dense_11218267
identityИв+batch_normalization/StatefulPartitionedCallв-batch_normalization_1/StatefulPartitionedCallв-batch_normalization_2/StatefulPartitionedCallв-batch_normalization_3/StatefulPartitionedCallвconv3d/StatefulPartitionedCallв conv3d_1/StatefulPartitionedCallв conv3d_2/StatefulPartitionedCallв conv3d_3/StatefulPartitionedCallвdense/StatefulPartitionedCallвdropout/StatefulPartitionedCallв!dropout_1/StatefulPartitionedCallв!dropout_2/StatefulPartitionedCallвlayer1/StatefulPartitionedCallвlayer2/StatefulPartitionedCall¤
conv3d/StatefulPartitionedCallStatefulPartitionedCallinputsconv3d_11218193conv3d_11218195*
Tin
2*
Tout
2*3
_output_shapes!
:         *$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_conv3d_layer_call_and_return_conditional_losses_112168632 
conv3d/StatefulPartitionedCallЭ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv3d/StatefulPartitionedCall:output:0batch_normalization_11218198batch_normalization_11218200batch_normalization_11218202batch_normalization_11218204*
Tin	
2*
Tout
2*3
_output_shapes!
:         *$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_112175562-
+batch_normalization/StatefulPartitionedCall╡
 conv3d_1/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0conv3d_1_11218207conv3d_1_11218209*
Tin
2*
Tout
2*3
_output_shapes!
:         *$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_conv3d_1_layer_call_and_return_conditional_losses_112170252"
 conv3d_1/StatefulPartitionedCallн
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv3d_1/StatefulPartitionedCall:output:0batch_normalization_1_11218212batch_normalization_1_11218214batch_normalization_1_11218216batch_normalization_1_11218218*
Tin	
2*
Tout
2*3
_output_shapes!
:         *$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_112176522/
-batch_normalization_1/StatefulPartitionedCall╖
 conv3d_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0conv3d_2_11218221conv3d_2_11218223*
Tin
2*
Tout
2*3
_output_shapes!
:         *$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_conv3d_2_layer_call_and_return_conditional_losses_112171872"
 conv3d_2/StatefulPartitionedCallн
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv3d_2/StatefulPartitionedCall:output:0batch_normalization_2_11218226batch_normalization_2_11218228batch_normalization_2_11218230batch_normalization_2_11218232*
Tin	
2*
Tout
2*3
_output_shapes!
:         *$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_112177482/
-batch_normalization_2/StatefulPartitionedCall╖
 conv3d_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0conv3d_3_11218235conv3d_3_11218237*
Tin
2*
Tout
2*3
_output_shapes!
:         *$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_conv3d_3_layer_call_and_return_conditional_losses_112173492"
 conv3d_3/StatefulPartitionedCallн
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv3d_3/StatefulPartitionedCall:output:0batch_normalization_3_11218240batch_normalization_3_11218242batch_normalization_3_11218244batch_normalization_3_11218246*
Tin	
2*
Tout
2*3
_output_shapes!
:         *$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*\
fWRU
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_112178442/
-batch_normalization_3/StatefulPartitionedCallО
!average_pooling3d/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:         * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_average_pooling3d_layer_call_and_return_conditional_losses_112175052#
!average_pooling3d/PartitionedCall┘
flatten/PartitionedCallPartitionedCall*average_pooling3d/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:         А
* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_112179072
flatten/PartitionedCallч
dropout/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:         А
* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_112179272!
dropout/StatefulPartitionedCallю
concatenate/PartitionedCallPartitionedCall(dropout/StatefulPartitionedCall:output:0inputs_1*
Tin
2*
Tout
2*(
_output_shapes
:         Г
* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_concatenate_layer_call_and_return_conditional_losses_112179522
concatenate/PartitionedCallР
layer1/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0layer1_11218253layer1_11218255*
Tin
2*
Tout
2*(
_output_shapes
:         ╚*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_layer1_layer_call_and_return_conditional_losses_112179722 
layer1/StatefulPartitionedCallЦ
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall'layer1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*(
_output_shapes
:         ╚* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_1_layer_call_and_return_conditional_losses_112180002#
!dropout_1/StatefulPartitionedCallХ
layer2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0layer2_11218259layer2_11218261*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_layer2_layer_call_and_return_conditional_losses_112180292 
layer2/StatefulPartitionedCallЧ
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall'layer2/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_2_layer_call_and_return_conditional_losses_112180572#
!dropout_2/StatefulPartitionedCallР
dense/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_11218265dense_11218267*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_112180862
dense/StatefulPartitionedCallО
IdentityIdentity&dense/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall^conv3d/StatefulPartitionedCall!^conv3d_1/StatefulPartitionedCall!^conv3d_2/StatefulPartitionedCall!^conv3d_3/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall^layer1/StatefulPartitionedCall^layer2/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*└
_input_shapesо
л:         н:         ::::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2@
conv3d/StatefulPartitionedCallconv3d/StatefulPartitionedCall2D
 conv3d_1/StatefulPartitionedCall conv3d_1/StatefulPartitionedCall2D
 conv3d_2/StatefulPartitionedCall conv3d_2/StatefulPartitionedCall2D
 conv3d_3/StatefulPartitionedCall conv3d_3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2@
layer2/StatefulPartitionedCalllayer2/StatefulPartitionedCall:\ X
4
_output_shapes"
 :         н
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
║
Ц
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_11217864

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИТ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulВ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:         2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subС
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:         2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0*
T0*3
_output_shapes!
:         2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:         :::::[ W
3
_output_shapes!
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ф
м
D__inference_layer2_layer_call_and_return_conditional_losses_11218029

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	╚*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddU
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:         2
Elue
IdentityIdentityElu:activations:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ╚:::P L
(
_output_shapes
:         ╚
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
╟+
╨
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_11217844

inputs
assignmovingavg_11217819
assignmovingavg_1_11217825)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpЩ
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indicesЫ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/meanИ
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
:2
moments/StopGradient░
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*3
_output_shapes!
:         2
moments/SquaredDifferenceб
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices╛
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/varianceГ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/SqueezeЛ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1а
AssignMovingAvg/decayConst*+
_class!
loc:@AssignMovingAvg/11217819*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayХ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_11217819*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp┼
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*+
_class!
loc:@AssignMovingAvg/11217819*
_output_shapes
:2
AssignMovingAvg/sub╝
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg/11217819*
_output_shapes
:2
AssignMovingAvg/mulЕ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_11217819AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/11217819*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpж
AssignMovingAvg_1/decayConst*-
_class#
!loc:@AssignMovingAvg_1/11217825*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayЫ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_11217825*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╧
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/11217825*
_output_shapes
:2
AssignMovingAvg_1/sub╞
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/11217825*
_output_shapes
:2
AssignMovingAvg_1/mulС
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_11217825AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/11217825*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulВ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:         2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subС
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:         2
batchnorm/add_1┴
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*3
_output_shapes!
:         2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:         ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:[ W
3
_output_shapes!
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
м	
й
6__inference_batch_normalization_layer_call_fn_11219287

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallг
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*N
_output_shapes<
::8                                    *&
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_112170022
StatefulPartitionedCall╡
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8                                    2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:8                                    ::::22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8                                    
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
И
─
(__inference_model_layer_call_fn_11219169
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28
identityИвStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28*+
Tin$
"2 *
Tout
2*'
_output_shapes
:         *@
_read_only_resource_inputs"
 	
*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_112184192
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*└
_input_shapesо
л:         н:         ::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :         н
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
║
Ц
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_11219943

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИТ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulВ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:         2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subС
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:         2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0*
T0*3
_output_shapes!
:         2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:         :::::[ W
3
_output_shapes!
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: "пL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Є
serving_default▐
H
input_1=
serving_default_input_1:0         н
;
input_20
serving_default_input_2:0         9
dense0
StatefulPartitionedCall:0         tensorflow/serving/predict:· 
╬е
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer_with_weights-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer_with_weights-8
layer-14
layer-15
layer_with_weights-9
layer-16
layer-17
layer_with_weights-10
layer-18
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
+Р&call_and_return_all_conditional_losses
С_default_save_signature
Т__call__"рЯ
_tf_keras_model┼Я{"class_name": "Model", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 24, 24, 24, 173]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv3D", "config": {"name": "conv3d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 24, 24, 24, 173]}, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [1, 1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [4], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization", "inbound_nodes": [[["conv3d", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 24, 24, 24, 20]}, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_1", "inbound_nodes": [[["batch_normalization", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [4], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_1", "inbound_nodes": [[["conv3d_1", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 22, 22, 22, 20]}, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [4, 4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_2", "inbound_nodes": [[["batch_normalization_1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [4], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_2", "inbound_nodes": [[["conv3d_2", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d_3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 19, 19, 19, 30]}, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [4, 4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_3", "inbound_nodes": [[["batch_normalization_2", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [4], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_3", "inbound_nodes": [[["conv3d_3", 0, 0, {}]]]}, {"class_name": "AveragePooling3D", "config": {"name": "average_pooling3d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [4, 4, 4]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [4, 4, 4]}, "data_format": "channels_last"}, "name": "average_pooling3d", "inbound_nodes": [[["batch_normalization_3", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["average_pooling3d", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["dropout", 0, 0, {}], ["input_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "layer1", "trainable": true, "dtype": "float32", "units": 200, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "MaxNorm", "config": {"max_value": 4, "axis": 0}}, "bias_constraint": {"class_name": "MaxNorm", "config": {"max_value": 4, "axis": 0}}}, "name": "layer1", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["layer1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "layer2", "trainable": true, "dtype": "float32", "units": 20, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "MaxNorm", "config": {"max_value": 4, "axis": 0}}, "bias_constraint": {"class_name": "MaxNorm", "config": {"max_value": 4, "axis": 0}}}, "name": "layer2", "inbound_nodes": [[["dropout_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_2", "inbound_nodes": [[["layer2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["dropout_2", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0], ["input_2", 0, 0]], "output_layers": [["dense", 0, 0]]}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 24, 24, 24, 173]}, {"class_name": "TensorShape", "items": [null, 3]}], "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 24, 24, 24, 173]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv3D", "config": {"name": "conv3d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 24, 24, 24, 173]}, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [1, 1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [4], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization", "inbound_nodes": [[["conv3d", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 24, 24, 24, 20]}, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_1", "inbound_nodes": [[["batch_normalization", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [4], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_1", "inbound_nodes": [[["conv3d_1", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 22, 22, 22, 20]}, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [4, 4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_2", "inbound_nodes": [[["batch_normalization_1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [4], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_2", "inbound_nodes": [[["conv3d_2", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d_3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 19, 19, 19, 30]}, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [4, 4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_3", "inbound_nodes": [[["batch_normalization_2", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [4], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_3", "inbound_nodes": [[["conv3d_3", 0, 0, {}]]]}, {"class_name": "AveragePooling3D", "config": {"name": "average_pooling3d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [4, 4, 4]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [4, 4, 4]}, "data_format": "channels_last"}, "name": "average_pooling3d", "inbound_nodes": [[["batch_normalization_3", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["average_pooling3d", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["dropout", 0, 0, {}], ["input_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "layer1", "trainable": true, "dtype": "float32", "units": 200, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "MaxNorm", "config": {"max_value": 4, "axis": 0}}, "bias_constraint": {"class_name": "MaxNorm", "config": {"max_value": 4, "axis": 0}}}, "name": "layer1", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["layer1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "layer2", "trainable": true, "dtype": "float32", "units": 20, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "MaxNorm", "config": {"max_value": 4, "axis": 0}}, "bias_constraint": {"class_name": "MaxNorm", "config": {"max_value": 4, "axis": 0}}}, "name": "layer2", "inbound_nodes": [[["dropout_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_2", "inbound_nodes": [[["layer2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["dropout_2", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0], ["input_2", 0, 0]], "output_layers": [["dense", 0, 0]]}}, "training_config": {"loss": "binary_crossentropy", "metrics": null, "weighted_metrics": null, "loss_weights": null, "sample_weight_mode": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
Е"В
_tf_keras_input_layerт{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 24, 24, 24, 173]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 24, 24, 24, 173]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
и

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
+У&call_and_return_all_conditional_losses
Ф__call__"Б

_tf_keras_layerч	{"class_name": "Conv3D", "name": "conv3d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 24, 24, 24, 173]}, "stateful": false, "config": {"name": "conv3d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 24, 24, 24, 173]}, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [1, 1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {"-1": 173}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24, 24, 24, 173]}}
Щ	
 axis
	!gamma
"beta
#moving_mean
$moving_variance
%trainable_variables
&	variables
'regularization_losses
(	keras_api
+Х&call_and_return_all_conditional_losses
Ц__call__"├
_tf_keras_layerй{"class_name": "BatchNormalization", "name": "batch_normalization", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [4], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {"4": 20}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24, 24, 24, 20]}}
е

)kernel
*bias
+trainable_variables
,	variables
-regularization_losses
.	keras_api
+Ч&call_and_return_all_conditional_losses
Ш__call__"■	
_tf_keras_layerф	{"class_name": "Conv3D", "name": "conv3d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 24, 24, 24, 20]}, "stateful": false, "config": {"name": "conv3d_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 24, 24, 24, 20]}, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {"-1": 20}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24, 24, 24, 20]}}
Э	
/axis
	0gamma
1beta
2moving_mean
3moving_variance
4trainable_variables
5	variables
6regularization_losses
7	keras_api
+Щ&call_and_return_all_conditional_losses
Ъ__call__"╟
_tf_keras_layerн{"class_name": "BatchNormalization", "name": "batch_normalization_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [4], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {"4": 20}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 22, 22, 22, 20]}}
е

8kernel
9bias
:trainable_variables
;	variables
<regularization_losses
=	keras_api
+Ы&call_and_return_all_conditional_losses
Ь__call__"■	
_tf_keras_layerф	{"class_name": "Conv3D", "name": "conv3d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 22, 22, 22, 20]}, "stateful": false, "config": {"name": "conv3d_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 22, 22, 22, 20]}, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [4, 4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {"-1": 20}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 22, 22, 22, 20]}}
Э	
>axis
	?gamma
@beta
Amoving_mean
Bmoving_variance
Ctrainable_variables
D	variables
Eregularization_losses
F	keras_api
+Э&call_and_return_all_conditional_losses
Ю__call__"╟
_tf_keras_layerн{"class_name": "BatchNormalization", "name": "batch_normalization_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [4], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {"4": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 19, 19, 19, 30]}}
е

Gkernel
Hbias
Itrainable_variables
J	variables
Kregularization_losses
L	keras_api
+Я&call_and_return_all_conditional_losses
а__call__"■	
_tf_keras_layerф	{"class_name": "Conv3D", "name": "conv3d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 19, 19, 19, 30]}, "stateful": false, "config": {"name": "conv3d_3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 19, 19, 19, 30]}, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [4, 4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {"-1": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 19, 19, 19, 30]}}
Э	
Maxis
	Ngamma
Obeta
Pmoving_mean
Qmoving_variance
Rtrainable_variables
S	variables
Tregularization_losses
U	keras_api
+б&call_and_return_all_conditional_losses
в__call__"╟
_tf_keras_layerн{"class_name": "BatchNormalization", "name": "batch_normalization_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [4], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {"4": 20}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 16, 20]}}
ь
Vtrainable_variables
W	variables
Xregularization_losses
Y	keras_api
+г&call_and_return_all_conditional_losses
д__call__"█
_tf_keras_layer┴{"class_name": "AveragePooling3D", "name": "average_pooling3d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "average_pooling3d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [4, 4, 4]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [4, 4, 4]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {}}}}
┴
Ztrainable_variables
[	variables
\regularization_losses
]	keras_api
+е&call_and_return_all_conditional_losses
ж__call__"░
_tf_keras_layerЦ{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
└
^trainable_variables
_	variables
`regularization_losses
a	keras_api
+з&call_and_return_all_conditional_losses
и__call__"п
_tf_keras_layerХ{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}}
щ"ц
_tf_keras_input_layer╞{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}
й
btrainable_variables
c	variables
dregularization_losses
e	keras_api
+й&call_and_return_all_conditional_losses
к__call__"Ш
_tf_keras_layer■{"class_name": "Concatenate", "name": "concatenate", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1280]}, {"class_name": "TensorShape", "items": [null, 3]}]}
╔

fkernel
gbias
htrainable_variables
i	variables
jregularization_losses
k	keras_api
+л&call_and_return_all_conditional_losses
м__call__"в
_tf_keras_layerИ{"class_name": "Dense", "name": "layer1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "layer1", "trainable": true, "dtype": "float32", "units": 200, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "MaxNorm", "config": {"max_value": 4, "axis": 0}}, "bias_constraint": {"class_name": "MaxNorm", "config": {"max_value": 4, "axis": 0}}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1283}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1283]}}
─
ltrainable_variables
m	variables
nregularization_losses
o	keras_api
+н&call_and_return_all_conditional_losses
о__call__"│
_tf_keras_layerЩ{"class_name": "Dropout", "name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
╞

pkernel
qbias
rtrainable_variables
s	variables
tregularization_losses
u	keras_api
+п&call_and_return_all_conditional_losses
░__call__"Я
_tf_keras_layerЕ{"class_name": "Dense", "name": "layer2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "layer2", "trainable": true, "dtype": "float32", "units": 20, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "MaxNorm", "config": {"max_value": 4, "axis": 0}}, "bias_constraint": {"class_name": "MaxNorm", "config": {"max_value": 4, "axis": 0}}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 200}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 200]}}
─
vtrainable_variables
w	variables
xregularization_losses
y	keras_api
+▒&call_and_return_all_conditional_losses
▓__call__"│
_tf_keras_layerЩ{"class_name": "Dropout", "name": "dropout_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
═

zkernel
{bias
|trainable_variables
}	variables
~regularization_losses
	keras_api
+│&call_and_return_all_conditional_losses
┤__call__"ж
_tf_keras_layerМ{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 20}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20]}}
Р
	Аiter
Бbeta_1
Вbeta_2

Гdecay
Дlearning_ratemфmх!mц"mч)mш*mщ0mъ1mы8mь9mэ?mю@mяGmЁHmёNmЄOmєfmЇgmїpmЎqmўzm°{m∙v·v√!v№"v¤)v■*v 0vА1vБ8vВ9vГ?vД@vЕGvЖHvЗNvИOvЙfvКgvЛpvМqvНzvО{vП"
	optimizer
╞
0
1
!2
"3
)4
*5
06
17
88
99
?10
@11
G12
H13
N14
O15
f16
g17
p18
q19
z20
{21"
trackable_list_wrapper
Ж
0
1
!2
"3
#4
$5
)6
*7
08
19
210
311
812
913
?14
@15
A16
B17
G18
H19
N20
O21
P22
Q23
f24
g25
p26
q27
z28
{29"
trackable_list_wrapper
 "
trackable_list_wrapper
╙
Еlayer_metrics
trainable_variables
Жmetrics
	variables
 Зlayer_regularization_losses
regularization_losses
Иnon_trainable_variables
Йlayers
Т__call__
С_default_save_signature
+Р&call_and_return_all_conditional_losses
'Р"call_and_return_conditional_losses"
_generic_user_object
-
╡serving_default"
signature_map
,:*н2conv3d/kernel
:2conv3d/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
Кlayer_metrics
Лmetrics
trainable_variables
	variables
 Мlayer_regularization_losses
regularization_losses
Нnon_trainable_variables
Оlayers
Ф__call__
+У&call_and_return_all_conditional_losses
'У"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
':%2batch_normalization/gamma
&:$2batch_normalization/beta
/:- (2batch_normalization/moving_mean
3:1 (2#batch_normalization/moving_variance
.
!0
"1"
trackable_list_wrapper
<
!0
"1
#2
$3"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
Пlayer_metrics
Рmetrics
%trainable_variables
&	variables
 Сlayer_regularization_losses
'regularization_losses
Тnon_trainable_variables
Уlayers
Ц__call__
+Х&call_and_return_all_conditional_losses
'Х"call_and_return_conditional_losses"
_generic_user_object
-:+2conv3d_1/kernel
:2conv3d_1/bias
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
Фlayer_metrics
Хmetrics
+trainable_variables
,	variables
 Цlayer_regularization_losses
-regularization_losses
Чnon_trainable_variables
Шlayers
Ш__call__
+Ч&call_and_return_all_conditional_losses
'Ч"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'2batch_normalization_1/gamma
(:&2batch_normalization_1/beta
1:/ (2!batch_normalization_1/moving_mean
5:3 (2%batch_normalization_1/moving_variance
.
00
11"
trackable_list_wrapper
<
00
11
22
33"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
Щlayer_metrics
Ъmetrics
4trainable_variables
5	variables
 Ыlayer_regularization_losses
6regularization_losses
Ьnon_trainable_variables
Эlayers
Ъ__call__
+Щ&call_and_return_all_conditional_losses
'Щ"call_and_return_conditional_losses"
_generic_user_object
-:+2conv3d_2/kernel
:2conv3d_2/bias
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
Юlayer_metrics
Яmetrics
:trainable_variables
;	variables
 аlayer_regularization_losses
<regularization_losses
бnon_trainable_variables
вlayers
Ь__call__
+Ы&call_and_return_all_conditional_losses
'Ы"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'2batch_normalization_2/gamma
(:&2batch_normalization_2/beta
1:/ (2!batch_normalization_2/moving_mean
5:3 (2%batch_normalization_2/moving_variance
.
?0
@1"
trackable_list_wrapper
<
?0
@1
A2
B3"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
гlayer_metrics
дmetrics
Ctrainable_variables
D	variables
 еlayer_regularization_losses
Eregularization_losses
жnon_trainable_variables
зlayers
Ю__call__
+Э&call_and_return_all_conditional_losses
'Э"call_and_return_conditional_losses"
_generic_user_object
-:+2conv3d_3/kernel
:2conv3d_3/bias
.
G0
H1"
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
иlayer_metrics
йmetrics
Itrainable_variables
J	variables
 кlayer_regularization_losses
Kregularization_losses
лnon_trainable_variables
мlayers
а__call__
+Я&call_and_return_all_conditional_losses
'Я"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'2batch_normalization_3/gamma
(:&2batch_normalization_3/beta
1:/ (2!batch_normalization_3/moving_mean
5:3 (2%batch_normalization_3/moving_variance
.
N0
O1"
trackable_list_wrapper
<
N0
O1
P2
Q3"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
нlayer_metrics
оmetrics
Rtrainable_variables
S	variables
 пlayer_regularization_losses
Tregularization_losses
░non_trainable_variables
▒layers
в__call__
+б&call_and_return_all_conditional_losses
'б"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
▓layer_metrics
│metrics
Vtrainable_variables
W	variables
 ┤layer_regularization_losses
Xregularization_losses
╡non_trainable_variables
╢layers
д__call__
+г&call_and_return_all_conditional_losses
'г"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
╖layer_metrics
╕metrics
Ztrainable_variables
[	variables
 ╣layer_regularization_losses
\regularization_losses
║non_trainable_variables
╗layers
ж__call__
+е&call_and_return_all_conditional_losses
'е"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
╝layer_metrics
╜metrics
^trainable_variables
_	variables
 ╛layer_regularization_losses
`regularization_losses
┐non_trainable_variables
└layers
и__call__
+з&call_and_return_all_conditional_losses
'з"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
┴layer_metrics
┬metrics
btrainable_variables
c	variables
 ├layer_regularization_losses
dregularization_losses
─non_trainable_variables
┼layers
к__call__
+й&call_and_return_all_conditional_losses
'й"call_and_return_conditional_losses"
_generic_user_object
!:
Г
╚2layer1/kernel
:╚2layer1/bias
.
f0
g1"
trackable_list_wrapper
.
f0
g1"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
╞layer_metrics
╟metrics
htrainable_variables
i	variables
 ╚layer_regularization_losses
jregularization_losses
╔non_trainable_variables
╩layers
м__call__
+л&call_and_return_all_conditional_losses
'л"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
╦layer_metrics
╠metrics
ltrainable_variables
m	variables
 ═layer_regularization_losses
nregularization_losses
╬non_trainable_variables
╧layers
о__call__
+н&call_and_return_all_conditional_losses
'н"call_and_return_conditional_losses"
_generic_user_object
 :	╚2layer2/kernel
:2layer2/bias
.
p0
q1"
trackable_list_wrapper
.
p0
q1"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
╨layer_metrics
╤metrics
rtrainable_variables
s	variables
 ╥layer_regularization_losses
tregularization_losses
╙non_trainable_variables
╘layers
░__call__
+п&call_and_return_all_conditional_losses
'п"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
╒layer_metrics
╓metrics
vtrainable_variables
w	variables
 ╫layer_regularization_losses
xregularization_losses
╪non_trainable_variables
┘layers
▓__call__
+▒&call_and_return_all_conditional_losses
'▒"call_and_return_conditional_losses"
_generic_user_object
:2dense/kernel
:2
dense/bias
.
z0
{1"
trackable_list_wrapper
.
z0
{1"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
┌layer_metrics
█metrics
|trainable_variables
}	variables
 ▄layer_regularization_losses
~regularization_losses
▌non_trainable_variables
▐layers
┤__call__
+│&call_and_return_all_conditional_losses
'│"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_dict_wrapper
(
▀0"
trackable_list_wrapper
 "
trackable_list_wrapper
X
#0
$1
22
33
A4
B5
P6
Q7"
trackable_list_wrapper
о
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18"
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
.
#0
$1"
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
.
20
31"
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
.
A0
B1"
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
.
P0
Q1"
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
┐

рtotal

сcount
т	variables
у	keras_api"Д
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
:  (2total
:  (2count
0
р0
с1"
trackable_list_wrapper
.
т	variables"
_generic_user_object
1:/н2Adam/conv3d/kernel/m
:2Adam/conv3d/bias/m
,:*2 Adam/batch_normalization/gamma/m
+:)2Adam/batch_normalization/beta/m
2:02Adam/conv3d_1/kernel/m
 :2Adam/conv3d_1/bias/m
.:,2"Adam/batch_normalization_1/gamma/m
-:+2!Adam/batch_normalization_1/beta/m
2:02Adam/conv3d_2/kernel/m
 :2Adam/conv3d_2/bias/m
.:,2"Adam/batch_normalization_2/gamma/m
-:+2!Adam/batch_normalization_2/beta/m
2:02Adam/conv3d_3/kernel/m
 :2Adam/conv3d_3/bias/m
.:,2"Adam/batch_normalization_3/gamma/m
-:+2!Adam/batch_normalization_3/beta/m
&:$
Г
╚2Adam/layer1/kernel/m
:╚2Adam/layer1/bias/m
%:#	╚2Adam/layer2/kernel/m
:2Adam/layer2/bias/m
#:!2Adam/dense/kernel/m
:2Adam/dense/bias/m
1:/н2Adam/conv3d/kernel/v
:2Adam/conv3d/bias/v
,:*2 Adam/batch_normalization/gamma/v
+:)2Adam/batch_normalization/beta/v
2:02Adam/conv3d_1/kernel/v
 :2Adam/conv3d_1/bias/v
.:,2"Adam/batch_normalization_1/gamma/v
-:+2!Adam/batch_normalization_1/beta/v
2:02Adam/conv3d_2/kernel/v
 :2Adam/conv3d_2/bias/v
.:,2"Adam/batch_normalization_2/gamma/v
-:+2!Adam/batch_normalization_2/beta/v
2:02Adam/conv3d_3/kernel/v
 :2Adam/conv3d_3/bias/v
.:,2"Adam/batch_normalization_3/gamma/v
-:+2!Adam/batch_normalization_3/beta/v
&:$
Г
╚2Adam/layer1/kernel/v
:╚2Adam/layer1/bias/v
%:#	╚2Adam/layer2/kernel/v
:2Adam/layer2/bias/v
#:!2Adam/dense/kernel/v
:2Adam/dense/bias/v
┌2╫
C__inference_model_layer_call_and_return_conditional_losses_11219037
C__inference_model_layer_call_and_return_conditional_losses_11218103
C__inference_model_layer_call_and_return_conditional_losses_11218185
C__inference_model_layer_call_and_return_conditional_losses_11218912└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ц2У
#__inference__wrapped_model_11216852ы
Л▓З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *[вX
VЪS
.К+
input_1         н
!К
input_2         
ю2ы
(__inference_model_layer_call_fn_11218482
(__inference_model_layer_call_fn_11219103
(__inference_model_layer_call_fn_11219169
(__inference_model_layer_call_fn_11218334└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
▒2о
D__inference_conv3d_layer_call_and_return_conditional_losses_11216863х
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *EвB
@К=9                                    н
Ц2У
)__inference_conv3d_layer_call_fn_11216873х
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *EвB
@К=9                                    н
Ж2Г
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_11219343
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_11219261
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_11219241
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_11219323┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ъ2Ч
6__inference_batch_normalization_layer_call_fn_11219287
6__inference_batch_normalization_layer_call_fn_11219274
6__inference_batch_normalization_layer_call_fn_11219356
6__inference_batch_normalization_layer_call_fn_11219369┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
▓2п
F__inference_conv3d_1_layer_call_and_return_conditional_losses_11217025ф
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *DвA
?К<8                                    
Ч2Ф
+__inference_conv3d_1_layer_call_fn_11217035ф
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *DвA
?К<8                                    
О2Л
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_11219441
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_11219461
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_11219543
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_11219523┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
в2Я
8__inference_batch_normalization_1_layer_call_fn_11219487
8__inference_batch_normalization_1_layer_call_fn_11219474
8__inference_batch_normalization_1_layer_call_fn_11219556
8__inference_batch_normalization_1_layer_call_fn_11219569┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
▓2п
F__inference_conv3d_2_layer_call_and_return_conditional_losses_11217187ф
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *DвA
?К<8                                    
Ч2Ф
+__inference_conv3d_2_layer_call_fn_11217197ф
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *DвA
?К<8                                    
О2Л
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_11219743
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_11219661
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_11219641
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_11219723┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
в2Я
8__inference_batch_normalization_2_layer_call_fn_11219756
8__inference_batch_normalization_2_layer_call_fn_11219674
8__inference_batch_normalization_2_layer_call_fn_11219687
8__inference_batch_normalization_2_layer_call_fn_11219769┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
▓2п
F__inference_conv3d_3_layer_call_and_return_conditional_losses_11217349ф
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *DвA
?К<8                                    
Ч2Ф
+__inference_conv3d_3_layer_call_fn_11217359ф
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *DвA
?К<8                                    
О2Л
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_11219841
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_11219923
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_11219943
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_11219861┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
в2Я
8__inference_batch_normalization_3_layer_call_fn_11219969
8__inference_batch_normalization_3_layer_call_fn_11219887
8__inference_batch_normalization_3_layer_call_fn_11219874
8__inference_batch_normalization_3_layer_call_fn_11219956┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
─2┴
O__inference_average_pooling3d_layer_call_and_return_conditional_losses_11217505э
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *MвJ
HКEA                                             
й2ж
4__inference_average_pooling3d_layer_call_fn_11217511э
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *MвJ
HКEA                                             
я2ь
E__inference_flatten_layer_call_and_return_conditional_losses_11219975в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╘2╤
*__inference_flatten_layer_call_fn_11219980в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╚2┼
E__inference_dropout_layer_call_and_return_conditional_losses_11219992
E__inference_dropout_layer_call_and_return_conditional_losses_11219997┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Т2П
*__inference_dropout_layer_call_fn_11220002
*__inference_dropout_layer_call_fn_11220007┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
є2Ё
I__inference_concatenate_layer_call_and_return_conditional_losses_11220014в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╪2╒
.__inference_concatenate_layer_call_fn_11220020в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ю2ы
D__inference_layer1_layer_call_and_return_conditional_losses_11220031в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╙2╨
)__inference_layer1_layer_call_fn_11220040в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╠2╔
G__inference_dropout_1_layer_call_and_return_conditional_losses_11220052
G__inference_dropout_1_layer_call_and_return_conditional_losses_11220057┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ц2У
,__inference_dropout_1_layer_call_fn_11220067
,__inference_dropout_1_layer_call_fn_11220062┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ю2ы
D__inference_layer2_layer_call_and_return_conditional_losses_11220078в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╙2╨
)__inference_layer2_layer_call_fn_11220087в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╠2╔
G__inference_dropout_2_layer_call_and_return_conditional_losses_11220099
G__inference_dropout_2_layer_call_and_return_conditional_losses_11220104┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ц2У
,__inference_dropout_2_layer_call_fn_11220114
,__inference_dropout_2_layer_call_fn_11220109┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
э2ъ
C__inference_dense_layer_call_and_return_conditional_losses_11220125в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╥2╧
(__inference_dense_layer_call_fn_11220134в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
<B:
&__inference_signature_wrapper_11218702input_1input_2▐
#__inference__wrapped_model_11216852╢$!#")*302189B?A@GHQNPOfgpqz{eвb
[вX
VЪS
.К+
input_1         н
!К
input_2         
к "-к*
(
denseК
dense         М
O__inference_average_pooling3d_layer_call_and_return_conditional_losses_11217505╕_в\
UвR
PКM
inputsA                                             
к "UвR
KКH
0A                                             
Ъ ф
4__inference_average_pooling3d_layer_call_fn_11217511л_в\
UвR
PКM
inputsA                                             
к "HКEA                                             И
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_11219441░2301ZвW
PвM
GКD
inputs8                                    
p
к "LвI
BК?
08                                    
Ъ И
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_11219461░3021ZвW
PвM
GКD
inputs8                                    
p 
к "LвI
BК?
08                                    
Ъ ╤
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_11219523z2301?в<
5в2
,К)
inputs         
p
к "1в.
'К$
0         
Ъ ╤
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_11219543z3021?в<
5в2
,К)
inputs         
p 
к "1в.
'К$
0         
Ъ р
8__inference_batch_normalization_1_layer_call_fn_11219474г2301ZвW
PвM
GКD
inputs8                                    
p
к "?К<8                                    р
8__inference_batch_normalization_1_layer_call_fn_11219487г3021ZвW
PвM
GКD
inputs8                                    
p 
к "?К<8                                    й
8__inference_batch_normalization_1_layer_call_fn_11219556m2301?в<
5в2
,К)
inputs         
p
к "$К!         й
8__inference_batch_normalization_1_layer_call_fn_11219569m3021?в<
5в2
,К)
inputs         
p 
к "$К!         ╤
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_11219641zAB?@?в<
5в2
,К)
inputs         
p
к "1в.
'К$
0         
Ъ ╤
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_11219661zB?A@?в<
5в2
,К)
inputs         
p 
к "1в.
'К$
0         
Ъ И
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_11219723░AB?@ZвW
PвM
GКD
inputs8                                    
p
к "LвI
BК?
08                                    
Ъ И
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_11219743░B?A@ZвW
PвM
GКD
inputs8                                    
p 
к "LвI
BК?
08                                    
Ъ й
8__inference_batch_normalization_2_layer_call_fn_11219674mAB?@?в<
5в2
,К)
inputs         
p
к "$К!         й
8__inference_batch_normalization_2_layer_call_fn_11219687mB?A@?в<
5в2
,К)
inputs         
p 
к "$К!         р
8__inference_batch_normalization_2_layer_call_fn_11219756гAB?@ZвW
PвM
GКD
inputs8                                    
p
к "?К<8                                    р
8__inference_batch_normalization_2_layer_call_fn_11219769гB?A@ZвW
PвM
GКD
inputs8                                    
p 
к "?К<8                                    И
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_11219841░PQNOZвW
PвM
GКD
inputs8                                    
p
к "LвI
BК?
08                                    
Ъ И
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_11219861░QNPOZвW
PвM
GКD
inputs8                                    
p 
к "LвI
BК?
08                                    
Ъ ╤
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_11219923zPQNO?в<
5в2
,К)
inputs         
p
к "1в.
'К$
0         
Ъ ╤
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_11219943zQNPO?в<
5в2
,К)
inputs         
p 
к "1в.
'К$
0         
Ъ р
8__inference_batch_normalization_3_layer_call_fn_11219874гPQNOZвW
PвM
GКD
inputs8                                    
p
к "?К<8                                    р
8__inference_batch_normalization_3_layer_call_fn_11219887гQNPOZвW
PвM
GКD
inputs8                                    
p 
к "?К<8                                    й
8__inference_batch_normalization_3_layer_call_fn_11219956mPQNO?в<
5в2
,К)
inputs         
p
к "$К!         й
8__inference_batch_normalization_3_layer_call_fn_11219969mQNPO?в<
5в2
,К)
inputs         
p 
к "$К!         Ж
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_11219241░#$!"ZвW
PвM
GКD
inputs8                                    
p
к "LвI
BК?
08                                    
Ъ Ж
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_11219261░$!#"ZвW
PвM
GКD
inputs8                                    
p 
к "LвI
BК?
08                                    
Ъ ╧
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_11219323z#$!"?в<
5в2
,К)
inputs         
p
к "1в.
'К$
0         
Ъ ╧
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_11219343z$!#"?в<
5в2
,К)
inputs         
p 
к "1в.
'К$
0         
Ъ ▐
6__inference_batch_normalization_layer_call_fn_11219274г#$!"ZвW
PвM
GКD
inputs8                                    
p
к "?К<8                                    ▐
6__inference_batch_normalization_layer_call_fn_11219287г$!#"ZвW
PвM
GКD
inputs8                                    
p 
к "?К<8                                    з
6__inference_batch_normalization_layer_call_fn_11219356m#$!"?в<
5в2
,К)
inputs         
p
к "$К!         з
6__inference_batch_normalization_layer_call_fn_11219369m$!#"?в<
5в2
,К)
inputs         
p 
к "$К!         ╙
I__inference_concatenate_layer_call_and_return_conditional_losses_11220014Е[вX
QвN
LЪI
#К 
inputs/0         А

"К
inputs/1         
к "&в#
К
0         Г

Ъ к
.__inference_concatenate_layer_call_fn_11220020x[вX
QвN
LЪI
#К 
inputs/0         А

"К
inputs/1         
к "К         Г
ї
F__inference_conv3d_1_layer_call_and_return_conditional_losses_11217025к)*VвS
LвI
GКD
inputs8                                    
к "LвI
BК?
08                                    
Ъ ═
+__inference_conv3d_1_layer_call_fn_11217035Э)*VвS
LвI
GКD
inputs8                                    
к "?К<8                                    ї
F__inference_conv3d_2_layer_call_and_return_conditional_losses_11217187к89VвS
LвI
GКD
inputs8                                    
к "LвI
BК?
08                                    
Ъ ═
+__inference_conv3d_2_layer_call_fn_11217197Э89VвS
LвI
GКD
inputs8                                    
к "?К<8                                    ї
F__inference_conv3d_3_layer_call_and_return_conditional_losses_11217349кGHVвS
LвI
GКD
inputs8                                    
к "LвI
BК?
08                                    
Ъ ═
+__inference_conv3d_3_layer_call_fn_11217359ЭGHVвS
LвI
GКD
inputs8                                    
к "?К<8                                    Ї
D__inference_conv3d_layer_call_and_return_conditional_losses_11216863лWвT
MвJ
HКE
inputs9                                    н
к "LвI
BК?
08                                    
Ъ ╠
)__inference_conv3d_layer_call_fn_11216873ЮWвT
MвJ
HКE
inputs9                                    н
к "?К<8                                    г
C__inference_dense_layer_call_and_return_conditional_losses_11220125\z{/в,
%в"
 К
inputs         
к "%в"
К
0         
Ъ {
(__inference_dense_layer_call_fn_11220134Oz{/в,
%в"
 К
inputs         
к "К         й
G__inference_dropout_1_layer_call_and_return_conditional_losses_11220052^4в1
*в'
!К
inputs         ╚
p
к "&в#
К
0         ╚
Ъ й
G__inference_dropout_1_layer_call_and_return_conditional_losses_11220057^4в1
*в'
!К
inputs         ╚
p 
к "&в#
К
0         ╚
Ъ Б
,__inference_dropout_1_layer_call_fn_11220062Q4в1
*в'
!К
inputs         ╚
p
к "К         ╚Б
,__inference_dropout_1_layer_call_fn_11220067Q4в1
*в'
!К
inputs         ╚
p 
к "К         ╚з
G__inference_dropout_2_layer_call_and_return_conditional_losses_11220099\3в0
)в&
 К
inputs         
p
к "%в"
К
0         
Ъ з
G__inference_dropout_2_layer_call_and_return_conditional_losses_11220104\3в0
)в&
 К
inputs         
p 
к "%в"
К
0         
Ъ 
,__inference_dropout_2_layer_call_fn_11220109O3в0
)в&
 К
inputs         
p
к "К         
,__inference_dropout_2_layer_call_fn_11220114O3в0
)в&
 К
inputs         
p 
к "К         з
E__inference_dropout_layer_call_and_return_conditional_losses_11219992^4в1
*в'
!К
inputs         А

p
к "&в#
К
0         А

Ъ з
E__inference_dropout_layer_call_and_return_conditional_losses_11219997^4в1
*в'
!К
inputs         А

p 
к "&в#
К
0         А

Ъ 
*__inference_dropout_layer_call_fn_11220002Q4в1
*в'
!К
inputs         А

p
к "К         А

*__inference_dropout_layer_call_fn_11220007Q4в1
*в'
!К
inputs         А

p 
к "К         А
о
E__inference_flatten_layer_call_and_return_conditional_losses_11219975e;в8
1в.
,К)
inputs         
к "&в#
К
0         А

Ъ Ж
*__inference_flatten_layer_call_fn_11219980X;в8
1в.
,К)
inputs         
к "К         А
ж
D__inference_layer1_layer_call_and_return_conditional_losses_11220031^fg0в-
&в#
!К
inputs         Г

к "&в#
К
0         ╚
Ъ ~
)__inference_layer1_layer_call_fn_11220040Qfg0в-
&в#
!К
inputs         Г

к "К         ╚е
D__inference_layer2_layer_call_and_return_conditional_losses_11220078]pq0в-
&в#
!К
inputs         ╚
к "%в"
К
0         
Ъ }
)__inference_layer2_layer_call_fn_11220087Ppq0в-
&в#
!К
inputs         ╚
к "К         ■
C__inference_model_layer_call_and_return_conditional_losses_11218103╢#$!")*230189AB?@GHPQNOfgpqz{mвj
cв`
VЪS
.К+
input_1         н
!К
input_2         
p

 
к "%в"
К
0         
Ъ ■
C__inference_model_layer_call_and_return_conditional_losses_11218185╢$!#")*302189B?A@GHQNPOfgpqz{mвj
cв`
VЪS
.К+
input_1         н
!К
input_2         
p 

 
к "%в"
К
0         
Ъ А
C__inference_model_layer_call_and_return_conditional_losses_11218912╕#$!")*230189AB?@GHPQNOfgpqz{oвl
eвb
XЪU
/К,
inputs/0         н
"К
inputs/1         
p

 
к "%в"
К
0         
Ъ А
C__inference_model_layer_call_and_return_conditional_losses_11219037╕$!#")*302189B?A@GHQNPOfgpqz{oвl
eвb
XЪU
/К,
inputs/0         н
"К
inputs/1         
p 

 
к "%в"
К
0         
Ъ ╓
(__inference_model_layer_call_fn_11218334й#$!")*230189AB?@GHPQNOfgpqz{mвj
cв`
VЪS
.К+
input_1         н
!К
input_2         
p

 
к "К         ╓
(__inference_model_layer_call_fn_11218482й$!#")*302189B?A@GHQNPOfgpqz{mвj
cв`
VЪS
.К+
input_1         н
!К
input_2         
p 

 
к "К         ╪
(__inference_model_layer_call_fn_11219103л#$!")*230189AB?@GHPQNOfgpqz{oвl
eвb
XЪU
/К,
inputs/0         н
"К
inputs/1         
p

 
к "К         ╪
(__inference_model_layer_call_fn_11219169л$!#")*302189B?A@GHQNPOfgpqz{oвl
eвb
XЪU
/К,
inputs/0         н
"К
inputs/1         
p 

 
к "К         Є
&__inference_signature_wrapper_11218702╟$!#")*302189B?A@GHQNPOfgpqz{vвs
в 
lкi
9
input_1.К+
input_1         н
,
input_2!К
input_2         "-к*
(
denseК
dense         