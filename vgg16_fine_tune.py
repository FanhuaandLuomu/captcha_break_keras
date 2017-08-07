#coding:utf-8
# 使用vgg16+全连接分类层进行验证码识别
# 冻结vgg16的卷积部分，训练顶部的全连接softmax分类层

from __future__ import division
from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from PIL import Image

import string
# 验证码的字符来源
characters=string.digits+string.ascii_uppercase
print characters

from keras.models import *
from keras.layers import *
from keras.applications.vgg16 import VGG16
from keras.utils.visualize_util import plot

# 验证码的大小  长度4个字符
width,height,n_len,n_class=170,80,4,len(characters)

# generator=ImageCaptcha(width=width,height=height)
# random_str=''.join([random.choice(characters) for j in range(4)])
# img=generator.generate_image(random_str)

# plt.show(img)
# plt.title(random_str)

# 生成器 每次产生batch_size个样本
# 注意后端使用的是Theano    (3,height,width)  th
# 原项目中使用Tensorflow  (height,width,3)  tf
def gen(batch_size=32):
	# 
	X=np.zeros((batch_size,height,width,3),dtype=np.uint8)
	# one-hot 
	y=[np.zeros((batch_size,n_class),dtype=np.uint8) for i in range(n_len)]
	generator=ImageCaptcha(width=width,height=height)
	while True:
		for i in range(batch_size):
			random_str=''.join([random.choice(characters) for j in range(n_len)])
			X[i]=generator.generate_image(random_str)
			for j,ch in enumerate(random_str):
				y[j][i,:]=0
				y[j][i,characters.find(ch)]=1
		# 将图片输入转为th格式输入
		# X=np.transpose(X,(0,3,1,2))
		yield np.transpose(X,(0,3,1,2)),y


# 解码 得到预测字符串
def decode(y):
	y=np.argmax(np.array(y),axis=2)[:,0]
	return ''.join([characters[x] for x in y]) 

# x,y=next(gen(1))
# print x
# print y

# vgg no top model
model=VGG16(weights='imagenet',include_top=False,input_tensor=Input((3,height,width)))
# input_shape:(None,3,80,170)
# output_shape:(None,512,2,5)
print model.summary()

# top model
input1=Input((512,2,5))
flatten1=Flatten()(input1)
dropout1=Dropout(0.25)(flatten1)
x=[Dense(n_class,activation='softmax',name='c%d' %(i+1))(dropout1) for i in range(4)]

top_model=Model(input1,x)
print top_model.summary()
# input_shape:(None,512,2,5)
# output_shape:[(None,36),(None,36),(None,36),(None,36)]

# 合并 model
vgg_model=Model(model.input,top_model(model.output))


for layer in vgg_model.layers[:19]:
	layer.trainable=False

print vgg_model.summary()
plot(vgg_model,to_file='vgg_fine_tune.png',show_shapes=True)

vgg_model.compile(loss='categorical_crossentropy',
				optimizer='adam',
				metrics=['accuracy'])

# # 训练模型
vgg_model.fit_generator(gen(),samples_per_epoch=51200,nb_epoch=15,
					validation_data=gen(),nb_val_samples=1280)

# 保存权重
vgg_model.save_weights('vgg_fine_tune_model_weights.h5')
					
# 测试
err=0

# 测试集2000
ppath='pic'
fnames=os.listdir(ppath)
for fname in fnames:
	img=Image.open(ppath+os.sep+fname)
	img=img.resize((170,80))
	img=img.convert('RGB')
	arr=np.asarray(img,dtype='float32')
	# arr=[arr]  (1,170,80,3)
	arr=np.expand_dims(arr,axis=0)
	# (1,3,170,80)
	arr=np.transpose(arr,(0,3,1,2))

	y_pred = model.predict(arr)
	y_pred=decode(y_pred)

	y_real=fname.split('.')[0]

	if y_real!=y_pred:
		err+=1
		print y_real,y_pred

print err
print 'acc:',(len(fnames)-err)/len(fnames)

