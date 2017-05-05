#coding:utf-8
# 生成2000个随机的验证码作为测试集

from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import numpy as np
import random
import os

import string
characters=string.digits+string.ascii_uppercase
print characters

width,height,n_len,n_class=170,80,4,len(characters)

generator=ImageCaptcha(width=width,height=height)

test_path='pic'
if not os.path.exists(test_path):
	os.mkdir(test_path)

for i in range(2000):
	random_str=''.join([random.choice(characters) for j in range(4)])
	img=generator.generate_image(random_str)

	img.save(test_path+os.sep+'%s.jpg' %(random_str))