# -*- coding: utf-8 -*-
# list datalab file
# !ls datalab/1736/file/
# #把上传的tt.csv转换为MP4文件（实际上是一个MP4文件但是限于不能上传视频数据集所以。。。）
# !mv datalab/1736/file/tt.csv datalab/1736/file/test.mp4
# #创建一个image目录  用于待会儿保存从视频提取的图片
# !mkdir datalab/1736/file/image
# !ls datalab/1736/file
def extrac_image(vc,save_path,mp4_file):
    c = 1
    rval = False
    if vc.isOpened():  # 判断是否正常打开
       rval, frame = vc.read()
    else:
       rval = False
    # 一般MP4每秒30帧
    timeF =50
    while rval:# 存取视频第6秒的图片
        rows, cols, channel = frame.shape
        frame = cv2.resize(frame, (224, 224), fx=0, fy=0, interpolation=cv2.INTER_AREA)
        if(c % timeF==0):
            #print("--write-to-file---: " )
            cv2.imwrite(save_path + mp4_file.replace('mp4','jpg') , frame)  # 存储为图像
            break
        c = c + 1
        rval, frame = vc.read()
    vc.release()

import cv2
import os
path='E:\\data\\video\\VQADatasetA_20180815\\test\\'
save_path="E:\\data\\video\\VQADatasetA_20180815\\image\\test\\"
files=os.listdir(path)
for mp4_file in files:
            mp4_file_dir = os.path.join(path, mp4_file)
            vc = cv2.VideoCapture( mp4_file_dir)  # 读入视频文件
            extrac_image(vc, save_path,mp4_file)

