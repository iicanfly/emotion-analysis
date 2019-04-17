神经网络：
源代码均为Python文件,且python环境为python2.7

文件夹分为/2_types文件夹和/FlaskWebProject1

其中/2_types中主要是文本的处理和神经网络训练的代码,/FlaskWebProject1则主要为CS架构下进行在线情感分析判断的代码

/2_types下文件的作用
--------/2文件夹:
---------------------test_data.txt: testData.txt处理后的文件,如去除停止词和标点符号等
---------------------testData.txt: 存放预测集原文本
---------------------(注:因为找到的数据预测集没有给答案,所以预测集后来是将训练集分割,使用训练集的一部分作为预测集)
---------------------train_data.txt: traintxt处理后的文件,如去除停止词和标点符号等
---------------------train_data_last.txt: 对train_data.txt使用tf-idf方法进行词库提取后的文本
---------------------trainData.txt: 存放训练集原文本
---------------------trainLabel.txt: 训练集的标签

--------/parameter文件夹: 存放每一次迭代训练的参数

--------dictOrg.txt: 使用tf-idf方法提取出来的词库

--------filehandle_train.py:文本处理代码

--------NeuralNetwork.py:神经网络代码

--------tf-idf.py:使用tf-idf方法进行词库提取的代码

--------程序运行记录.txt: 存放训练过程中的训练信息
  
  
/FlaskWebProject1: 使用python的flask框架进行网站的编写,用户可以在线进行文本的情感分析
--------runserver.py: 启动网站服务器

--------views.py: 业务逻辑代码

--------home.html: 网页的html代码

--------/statics: 存放css文件和图标等
  
结果的简要介绍:
神经网络参数如下：
输入：one-hot(23151维)
架构：23151(输入)*64*64*64(隐层)*1(输出)
批(batch)：128
学习率：0.01
迭代次数:40次
训练集：验证集=23500 : 499

准确率最高的一次训练如下:(在程序运行记录.txt当中可以进行查看)
Epoch 9 training now
Epoch 9 training complete
Cost on training data: 0.236970892156
Accuracy on training data: 21326 / 23500
Cost on evaluation data: 0.311158956432
Accuracy on evaluation data: 440 / 499








