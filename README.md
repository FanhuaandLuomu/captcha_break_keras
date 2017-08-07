# captcha_break_keras
keras theano  验证码破解  字母+数字
# 1. python get_capthch.py
生成2000个随机验证码，作为固定测试集
# 2. python captcha_test.py
使用生成器gen每次生成batch_size个样本，fit_generator训练，节约内存。
(模型预训练权重已给出，可直接使用)
# 3. python captcha_test_ctc_loss
ctc loss
# 4. python vgg_16_fine_tune.py
使用vgg16进行特征提取，训练顶部分类层
