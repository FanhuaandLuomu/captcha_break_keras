# captcha_break_keras
# keras theano  验证码破解  字母+数字
# 1. python get_capthch.py
生成2000个随机验证码，作为固定测试集 
# 2. python captcha_test.py
使用生成器gen每次生成batch_size个样本，fit_generator训练，节约内存。
在测试集上测试，正确率达到0.95~
(模型预训练权重已给出，可直接使用)
# 3. python captcha_test_ctc_loss
ctc loss
# 4. python vgg_16_fine_tune.py
使用vgg16进行特征提取，训练顶部分类层

# ===============================================
# 使用腾讯验证码(英文大小写字母)--验证码下载：http://pan.baidu.com/s/1miMaFna
python tencent_captcha_test.py
共10w已标注验证码，取9w训练，1w测试。
可在测试中打印预测标签和真实标签（测试样本1w也会单独抽出，存于test_samples3文件夹中，方便对比）
也可打印错误样本，看看是哪些字符识别的不好（一般是o-p,b-h等）。
1w测试样本的正确率是 0.7873
