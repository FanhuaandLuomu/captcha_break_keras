#coding:utf-8
from __future__ import division
from captcha.image import ImageCaptcha
from keras.utils.visualize_util import plot
import numpy as np
import random
import os
from PIL import Image
from keras import backend as K
import matplotlib.pyplot as plt

def ctc_lambda_func(args):
	y_pred,labels,input_length,label_length=args
	y_pred=y_pred[:,2:,:]
	return K.ctc_batch_cost(labels,y_pred,input_length,label_length)

import string
characters=string.digits+string.ascii_uppercase
print characters

width,height,n_len,n_class=170,80,4,len(characters)

from keras.models import *
from keras.layers import *
rnn_size=128

input_tensor=Input((3,width,height))
x=input_tensor
for i in range(3):
	x=Convolution2D(32,3,3,activation='relu')(x)
	x=Convolution2D(32,3,3,activation='relu')(x)
	x=MaxPooling2D(pool_size=(2,2))(x)

# conv_shape=x.output_shape
conv_shape=K.int_shape(x)
# conv_shape=np.array([17,6,32])
print conv_shape,conv_shape[2],conv_shape[3],conv_shape[1]

x=Reshape(target_shape=(int(conv_shape[2]),int(conv_shape[3]*conv_shape[1])))(x)

x=Dense(32,activation='relu')(x)

gru1=GRU(rnn_size,return_sequences=True,init='he_normal',name='gru1')(x)
gru1_b=GRU(rnn_size,return_sequences=True,init='he_normal',
			go_backwards=True,name='gru1_b')(x)
gru1_merged=merge([gru1,gru1_b],mode='sum')

gru2=GRU(rnn_size,return_sequences=True,init='he_normal',name='gru2')(gru1_merged)
gru2_b=GRU(rnn_size,return_sequences=True,init='he_normal',
			go_backwards=True,name='gru2_b')(gru1_merged)
x=merge([gru2,gru2_b],mode='concat')

x=Dropout(0.25)(x)
x=Dense(n_class,init='he_normal',activation='softmax')(x)
base_model=Model(input=input_tensor,output=x)

labels=Input(name='the_labels',shape=[n_len],dtype='float32')
input_length=Input(name='input_length',shape=[1],dtype='int64')
label_length=Input(name='label_length',shape=[1],dtype='int64')
loss_out=Lambda(ctc_lambda_func,output_shape=(1,),
				name='ctc')([x,labels,input_length,label_length])

model=Model(input=[input_tensor,labels,input_length,label_length],
			output=[loss_out])

model.compile(loss={'ctc':lambda y_true,y_pred:y_pred},optimizer='adadelta')

# 模型可视化
plot(model,to_file='model_ctc.png',show_shapes=True)




def gen(batch_size=128):
	X=np.zeros((batch_size,height,width,3),dtype=np.uint8)
	y=np.zeros((batch_size,n_len),dtype=np.uint8)
	while True:
		generator=ImageCaptcha(width=width,height=height)
		for i in range(batch_size):
			random_str=''.join([random.choice(characters) for j in range(4)])
			X[i]=np.array(generator.generate_image(random_str))
			# X[i]=np.transpose(X[i],(2,1,0))
			# 长n_len  下标表示
			y[i]=[characters.find(x) for x in random_str]	
		yield [X.transpose((0,3,2,1)),y,np.ones(batch_size)*int(conv_shape[1]-2),
				np.ones(batch_size)*n_len],np.ones(batch_size)

x,y=next(gen(1))
print x[0].shape,x[1].shape,x[2].shape,x[3].shape,y.shape

def evaluate(model,batch_num=10):
	batch_acc=0
	generator=gen()
	for i in range(batch_num):
		[X_test,y_test,_,_],_=next(generator)
		y_pred=base_model.predict(X_test)
		shape=y_pred[:,2:,:].shape
		ctc_decode=K.ctc_decode(y_pred[:,2:,:],
						input_length=np.ones(shape[0])*shape[1])[0][0]
		out=K.get_value(ctc_decode)[:,:4]
		if out.shape==4:
			batch_acc+=((y_test==out).sum(axis=1)==4).mean()
	return batch_acc/batch_num


from keras.callbacks import *

class Evaluate(Callback):
	def __init__(self):
		self.accs=[]
	def on_epoch_end(self,epoch,logs=None):
		acc=evaluate(base_model)*100
		self.accs.append(acc)
		print 
		print 'acc:%f%%' %acc

evaluator=Evaluate()


model.fit_generator(gen(128),samples_per_epoch=51200,nb_epoch=1,
			callbacks=[EarlyStopping(patience=10),evaluator],
			validation_data=gen(),nb_val_samples=1280)
