# -*- coding:utf-8 -*-
# made by YY 2019.4.16
import os
import re
import requests
import csv
import sys


class DownloadImages:
    def __init__(self, download_id, key_word, download_max):
        self.download_sum = 0
        self.download_max = download_max
        self.key_word = key_word
        self.save_path = './pic/' + download_id  # 每次下载一种宝可梦的图片都建立相应的目录

    def start_download(self):
        self.download_sum = 0
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)  # 确保创建好了目录
        while self.download_sum < self.download_max:
            try:
                str_pn = str(self.download_sum)
                url = 'http://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&' \
                      'word=' + self.key_word + '&pn=' + str_pn
                print(url)
                result = requests.get(url)
                self.download_images(result.text)
            except Exception:
                self.download_sum += 1
                continue
        print('下载完成')

    def download_images(self, html):
        img_urls = re.findall('"objURL":"(.*?)",', html, re.S)  # 正则匹配获取当前页面的图片地址
        print('找到关键词:' + self.key_word + '的图片，现在开始下载图片...')
        for img_url in img_urls:
            print('正在下载第' + str(self.download_sum + 1) +
                  '张图片，图片地址:' + str(img_url))
            try:
                pic = requests.get(img_url, timeout=50)
                pic_name = self.save_path + '/' + \
                    str(self.download_sum + 1) + '.jpg'
                with open(pic_name, 'wb') as f:
                    f.write(pic.content)  # 把图片文件写入相应的目录
                self.download_sum += 1
                if self.download_sum >= self.download_max:
                    break
            except Exception:
                self.download_sum += 1
                continue


if __name__ == '__main__':
    if len(sys.argv) > 1:
        csv_file_r = open(sys.argv[1], 'r', newline='', encoding="utf-8")
        csv_reader = csv.reader(csv_file_r, delimiter=',', quotechar='"')
        next(csv_reader)
        for row in csv_reader:
            downloadImages = DownloadImages(row[0], row[1], int(row[2]))  # 获取编号 名称 下载数量
            downloadImages.start_download()