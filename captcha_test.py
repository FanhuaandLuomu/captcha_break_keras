#coding:utf-8
# 使用卷积神经网络（CNN）识别验证码(数字+字母)
# 2000 个测试集上 acc 在94% 左右
# 错误分析：0 与 O较难区分

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

# th (3,height,width)
input_tensor=Input((3,height,width))
x=input_tensor
for i in range(4):
	x=Convolution2D(32*2**i,3,3,activation='relu')(x)
	x=Convolution2D(32*2**i,3,3,activation='relu')(x)
	x=MaxPooling2D((2,2))(x)

x=Flatten()(x)
x=Dropout(0.25)(x)
x=[Dense(n_class,activation='softmax',name='c%d' %(i+1))(x) for i in range(4)]

model=Model(input=input_tensor,output=x)

model.compile(loss='categorical_crossentropy',
				optimizer='adadelta',
				metrics=['accuracy'])

# model.load_weights('my_model_weights.h5')

'''
# 模型可视化
plot(model,to_file='model.png',show_shapes=True)

# 训练模型
model.fit_generator(gen(),samples_per_epoch=51200,nb_epoch=15,
					validation_data=gen(),nb_val_samples=1280)
# 保存权重
model.save_weights('my_model_weights.h5')
'''

# 测试  载入权重
model.load_weights('my_model_weights.h5')

'''
X, y = next(gen(1))
y_pred = model.predict(X)

print decode(y)
print decode(y_pred)
'''

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

'''
from tqdm import tqdm

def evaluate(model,batch_num=1000):
	batch_acc=0
	generator=gen()
	for i in tqdm(range(batch_num)):
		X,y=next(generator)   #  一次产生32个样本
		y_pred=model.predict(X)
		y_pred=np.argmax(y_pred,axis=2).T
		y_true=np.argmax(y,axis=2).T
		batch_acc+=np.mean(map(np.array_equal,y_true,y_pred))
	return batch_acc/batch_num

print evaluate(model)
'''