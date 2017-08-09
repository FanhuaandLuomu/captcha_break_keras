#coding:utf-8
# tencent 验证码 大小写字母
import matplotlib.pyplot as plt
import random
import os
from PIL import Image
import shutil

import glob
samples=glob.glob('sample/*.jpg')
print len(samples)

# 打乱训练样本
import numpy as np
np.random.shuffle(samples)

from keras.models import *
from keras.layers import *
from keras.utils.visualize_util import plot

nb_train=90000
train_samples=samples[:90000]
test_samples=samples[90000:100000]

# 单独抽出测试集 用于观测预测结果
test_path='test_samples3'
if not os.path.exists(test_path):
	os.mkdir(test_path)

	for fname in test_samples:
		shutil.copyfile(fname,test_path+os.sep+fname[-8:])


print len(train_samples),len(test_samples)


# 验证码的大小  长度4个字符
width,height,n_len,n_class=129,53,4,26

def data_generator(data,batch_size):
	while True:
		batch=np.random.choice(data,batch_size)
		x=[]
		# y是一个长度为4的列表  因为4个输出
		y=[np.zeros((batch_size,n_class),dtype=np.uint8) for i in range(n_len)]
		for i,img in enumerate(batch):
			im=Image.open(img).resize((width,height))
			x.append(np.array(im))
			# y.append([ord(i)-ord('a') for i in img[-8:-4]])
			for j,ch in enumerate(img[-8:-4]):
				y[j][i,:]=0
				y[j][i,ord(ch)-ord('a')]=1

		x=np.array(x).astype(float)
		# y=np.array(y)

		yield np.transpose(x,(0,3,1,2)),y

# yield 生成器 节约内存
# x,y=next(data_generator(train_samples,32))
# print x.shape,y.shape

# th (3,height,width)
input_tensor=Input((3,height,width))
x=input_tensor
for i in range(3):
	x=Convolution2D(32*2**i,3,3,activation='relu')(x)
	x=Convolution2D(32*2**i,3,3,activation='relu')(x)
	x=MaxPooling2D((2,2))(x)

f=Flatten()(x)
d=Dropout(0.25)(f)
x=[Dense(n_class,activation='softmax',name='c%d' %(i+1))(d) for i in range(4)]

model=Model(input_tensor,x)

plot(model,to_file='tencent_captcha_cnn.png',show_shapes=True)

model.compile(loss='categorical_crossentropy',
				optimizer='adam',
				metrics=['accuracy'])

# load权重
model.load_weights('tencent_captcha_weight.h5')

# 训练模型
'''
model.fit_generator(data_generator(train_samples,128),samples_per_epoch=90000,
					nb_epoch=2,validation_data=data_generator(test_samples,100),
					nb_val_samples=10000)
# 保存权重
model.save_weights('tencent_captcha_weight.h5')
'''


# 测试模块
from tqdm import tqdm
total=0.
right=0.
index=0

def get_test_samples(test_path):

	test_samples=os.listdir(test_path)
	
	for i,img in enumerate(test_samples): 

		x=[]
		y=[np.zeros((1,n_class),dtype=np.uint8) for i in range(n_len)]

		im=Image.open(test_path+os.sep+img).resize((width,height))
		x.append(np.array(im))
		for j,ch in enumerate(img[-8:-4]):
			y[j][0,:]=0
			y[j][0,ord(ch)-ord('a')]=1
		yield np.transpose(x,(0,3,1,2)),y

		



for x,y in get_test_samples(test_path):
	index+=1

	# 真实label
	y=np.array([i.argmax(axis=1) for i in y]).T
	# print y

	real_label=[[chr(ord('a')+c) for c in item] for item in y]
	# print 'real_label_%d:' %(index),''.join(real_label[0])

	# 预测label
	_=model.predict(x)
	_=np.array([i.argmax(axis=1) for i in _]).T
	# print _
	pred_label=[[chr(ord('a')+c) for c in item] for item in _]
	# print 'pred_label_%d:' %(index),''.join(pred_label[0])
	
	total+=len(x)

	# right+=((_==y).sum(axis=1)==4).sum()

	if pred_label==real_label:
		right+=1
	else:
		# 打印识别错的验证码
		print 'real_label_%d:' %(index),''.join(real_label[0])
		print 'pred_label_%d:' %(index),''.join(pred_label[0])

print 'model acc:',right/total







