# coding=utf-8
# python train.py cfh flower
import os
import sys
import numpy as np
import tensorflow as tf
import keras
from PIL import Image

# 从文件夹读取图片和标签到numpy数组中


def read_train_data(data_dir):
    datas = []
    labels = []
    fpaths = []
    for Pokemon_index in os.listdir(data_dir):  # train_class为图片文件夹的下层目录的内容
        if not os.path.isdir(os.path.join(data_dir, Pokemon_index)):  # 判断该是否是文件夹,如果不是就跳过后面的for循环
            continue
        for pic in os.listdir(os.path.join(data_dir, Pokemon_index)):  # 获取每个子目录下的图片名
            fpath = os.path.join(data_dir, Pokemon_index, pic)  # 拼接得到完整的图片路径
            if (fpath[-4:].upper() == '.JPG'):  # 如果图片是jpg格式的
                fpaths.append(fpath)  # 把完整的图片路径添加到fpath列表中
                image = Image.open(fpath)  # 利用image模块打开图片
                image = image.resize((32, 32))  # 把图片像素调小
                data = np.array(image) / 255.0  # 把范围为0-255的RGB值缩小到0-1
                label = int(Pokemon_index)
                datas.append(data)
                labels.append(label)  # 把图片的上级目录设为标签
    datas = np.array(datas)  # 把所有的数组放到一个array中
    labels = np.array(labels)
    print("shape of datas: {}\tshape of labels: {}".format(
        datas.shape, labels.shape))  # {}中的内容由format括号内的元素代替
    return fpaths, datas, labels


def cnn_net(datas_placeholder, dropout_placeholdr, num_classes):
    # 定义卷积层, 20个卷积核, 卷积核大小为5，用Relu激活
    conv0 = tf.layers.conv2d(datas_placeholder, 20, 5, activation=tf.nn.relu)
    # 定义max-pooling层，pooling窗口为2x2，步长为2x2
    pool0 = tf.layers.max_pooling2d(conv0, [2, 2], [2, 2])
    # 定义卷积层, 40个卷积核, 卷积核大小为4，用Relu激活
    conv1 = tf.layers.conv2d(pool0, 40, 4, activation=tf.nn.relu)
    # 定义max-pooling层，pooling窗口为2x2，步长为2x2
    pool1 = tf.layers.max_pooling2d(conv1, [2, 2], [2, 2])
    # 将3维特征转换为1维向量
    flatten = tf.layers.flatten(pool1)
    # 全连接层，转换为长度为100的特征向量
    fc = tf.layers.dense(flatten, 400, activation=tf.nn.relu)
    # 加上DropOut，防止过拟合
    dropout_fc = tf.layers.dropout(fc, dropout_placeholdr)
    # 未激活的输出层
    logits = tf.layers.dense(dropout_fc, num_classes)
    return logits


if __name__ == '__main__':
    # 数据文件夹
    data_dir = sys.argv[1]  # 训练数据的文件夹
    # 模型文件路径
    if not os.path.exists(os.path.join('.', 'model')):
        os.makedirs(os.path.join('.', 'model'))
    model_path = os.path.join("model", sys.argv[2])  # 在本地当前目录下创建/model/模型名子目录
    # 载入图片
    fpaths, datas, labels = read_train_data(data_dir)  # 从图片文件夹读入图片
    # 计算有多少类图片
    num_classes = len(set(labels))
    # 定义Placeholder，存放输入和标签，神经网络构建graph的时在模型中的占位符
    datas_placeholder = tf.placeholder(tf.float32, [None, 32, 32, 3])  # 图片数据为4维
    labels_placeholder = tf.placeholder(tf.int32, [None])  # 类别为1维
    # 存放DropOut参数的容器，因为本身样本少，训练时为0.5，测试时为0
    dropout_placeholdr = tf.placeholder(tf.float32)
    # 获取模型输出层
    logits = cnn_net(datas_placeholder, dropout_placeholdr, num_classes)
    # 利用交叉熵定义损失，但求出来的结果是向量形式的
    losses = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.one_hot(labels_placeholder, num_classes),
        logits=logits
    )  # 参数分别为实际的标签和预测结果标签，然后计算两者的交叉熵
    # 平均损失
    mean_loss = tf.reduce_mean(losses)
    # 定义优化器，指定要优化的损失函数
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(losses)
    # 用于保存和载入模型
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # 如果是训练，初始化参数
        sess.run(tf.global_variables_initializer())  # 变量要在会话启动、初始化之后才能启动
        # 定义输入和Label以填充容器，训练时dropout为0.5
        train_feed_dict = {
            datas_placeholder: datas,
            labels_placeholder: labels,
            dropout_placeholdr: 0.5
        }
        for step in range(200):
            _, mean_loss_val = sess.run(
                [optimizer, mean_loss], feed_dict=train_feed_dict)

            if step % 10 == 0:
                print("step = {}\tmean loss = {}".format(step, mean_loss_val))  # 每十次显示一下模型误差
        saver.save(sess, model_path)
        print("训练结束，保存模型到{}".format(model_path))