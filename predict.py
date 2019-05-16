# coding=utf-8
# python predict.py test flower cfh.csv
import os
import sys
import numpy as np
import tensorflow as tf
import keras
from train import cnn_net
from PIL import Image
import csv
import tkinter as tk
from PIL import ImageTk

# 从文件夹读取图片和标签到numpy数组中


def read_test_data(data_dir):
    datas = []
    fpaths = []
    for pic in os.listdir(data_dir):
        fpath = os.path.join(data_dir, pic)
        if (fpath[-4:].upper() == '.JPG'):
            fpaths.append(fpath)
            image = Image.open(fpath)
            image = image.resize((32, 32))
            data = np.array(image) / 255.0
            datas.append(data)
    datas = np.array(datas)  # 获取图像矩阵
    return fpaths, datas


if __name__ == '__main__':
    filepath = []
    texts = []
    # 数据文件夹
    data_dir = sys.argv[1]
    # 模型文件路径
    model_path = os.path.join("model", sys.argv[2])
    # 载入标签字典
    label_dic = {}
    csv_file_r = open(sys.argv[3], 'r', newline='', encoding="utf-8")
    csv_reader = csv.reader(csv_file_r, delimiter=',', quotechar='"')
    next(csv_reader)
    for row in csv_reader:
        label_dic[row[0]] = row[1]  # 读取csv文件中每行的第二个值，即精灵名称
    # 载入图片
    fpaths, datas = read_test_data(data_dir)
    # 计算有多少类图片
    num_classes = len(label_dic)
    # 定义Placeholder，存放输入和标签
    datas_placeholder = tf.placeholder(tf.float32, [None, 32, 32, 3])
    # 存放DropOut参数的容器，训练时为0.5，测试时为0
    dropout_placeholdr = tf.placeholder(tf.float32)
    # 获取模型输出层
    logits = cnn_net(datas_placeholder, dropout_placeholdr, num_classes)
    # 预测predicted_labels的容器
    predicted_labels = tf.math.argmax(logits, 1)
    # 用于保存和载入模型
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # 如果是测试，载入参数
        saver.restore(sess, model_path)
        print("从{}载入模型".format(model_path))
        # 定义输入和Label以填充容器，测试时dropout为0
        test_feed_dict = {
            datas_placeholder: datas,
            dropout_placeholdr: 0  # 测试样本不需要删减
        }
        predicted_labels_val = sess.run(
            predicted_labels, feed_dict=test_feed_dict)  # 得到预测标签
        for fpath, predicted_label in zip(fpaths, predicted_labels_val):
            filepath.append('D:\Python\pokemon/' + fpath)
            texts.append(label_dic[str(predicted_label)])# print("{} => {}".format(fpath, label_dic[str(predicted_label)]))

# 图形界面部分
window = tk.Tk()
window.title('Pokemon识别结果')
window.geometry('1400x280')
# l = tk.Label(window, text='Pokemon', font=('微软雅黑', 14), width=30, height=2)


img = Image.open(filepath[0])
img = ImageTk.PhotoImage(img)
label_img = tk.Label(window, image = img)
label_img.pack(side='left')
text = tk.Label(window, text=texts[0], font=('微软雅黑', 12))
text.place(x=40, y=200)


img1 = Image.open(filepath[1])
img1 = ImageTk.PhotoImage(img1)
label_img1 = tk.Label(window, image = img1)
label_img1.pack(side='left')
text = tk.Label(window, text=texts[1], font=('微软雅黑', 12))
text.place(x=220, y=200)

img2 = Image.open(filepath[2])
img2 = ImageTk.PhotoImage(img2)
label_img2 = tk.Label(window, image = img2)
label_img2.pack(side='left')
text = tk.Label(window, text=texts[2], font=('微软雅黑', 12))
text.place(x=370, y=200)

img3 = Image.open(filepath[3])
img3 = ImageTk.PhotoImage(img3)
label_img3 = tk.Label(window, image = img3)
label_img3.pack(side='left')
text = tk.Label(window, text=texts[3], font=('微软雅黑', 12))
text.place(x=540, y=200)

img4 = Image.open(filepath[4])
img4 = ImageTk.PhotoImage(img4)
label_img4 = tk.Label(window, image = img4)
label_img4.pack(side='left')
text = tk.Label(window, text=texts[4], font=('微软雅黑', 12))
text.place(x=705, y=200)

img5 = Image.open(filepath[5])
img5 = ImageTk.PhotoImage(img5)
label_img5 = tk.Label(window, image = img5)
label_img5.pack(side='left')
text = tk.Label(window, text=texts[5], font=('微软雅黑', 12))
text.place(x=865, y=200)

img6 = Image.open(filepath[6])
img6 = ImageTk.PhotoImage(img6)
label_img6 = tk.Label(window, image = img6)
label_img6.pack(side='left')
text = tk.Label(window, text=texts[6], font=('微软雅黑', 12))
text.place(x=1030, y=200)

img7 = Image.open(filepath[7])
img7 = ImageTk.PhotoImage(img7)
label_img7 = tk.Label(window, image = img7)
label_img7.pack(side='left')
text = tk.Label(window, text=texts[7], font=('微软雅黑', 12))
text.place(x=1190, y=200)

window.mainloop()
