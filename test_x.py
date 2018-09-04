# creat train data
import string

import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model, Sequential, load_model
from keras.preprocessing import image
from keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np

path = 'E:\\data\\video\\VQADatasetA_20180815\\'
data_test = pd.read_csv(path + 'test.txt', header=None)
#data_test = data_test#[:2]  # !!!!!!!!!注意这里我只上传了2个图片，线下需修改
length = len(data_test) * 15//3
x_img = np.zeros((length, 224, 224, 3))
x_q = []

for i in range(len(data_test)):
    lable, q1, a11, a12, a13, q2, a21, a22, a23, q3, a31, a32, a33, q4, a41, a42, a43, q5, a51, a52, a53 = \
    data_test.loc[i]
    img = image.load_img(path + 'image\\test\\' + lable + '.jpg')
    i = i * 15//3
    for j in range(15//3):
        x_img[i + j] = image.img_to_array(img)
    x_q.append(str(q1))
    x_q.append(str(q2))
    x_q.append(str(q3))
    x_q.append(str(q4))
    x_q.append(str(q5))

print("creat data end.")

tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_q)
encoded = tokenizer.texts_to_sequences(x_q)
size = len(tokenizer.word_index) + 1

test_q = np.zeros((len(encoded), 100))
for i in range(len(encoded)):
    test_q[i][:len(encoded[i])] = encoded[i]

print('question encode size = ' + str(size))
print(test_q)


from keras.utils import to_categorical
#Reload model
model = load_model('E:\\data\\video\\VQADatasetA_20180815\\mode.h5')
result =model.predict([x_img,test_q])

#########################################整理格式
y_di= pd.read_csv(path + 'y_di.csv', header=None)#这里需要替换
y=y_di[:][0]
res=[]
for i in range(len(data_test)):
    tem_res=[]
    lable, q1, a11, a12, a13, q2, a21, a22, a23, q3, a31, a32, a33, q4, a41, a42, a43, q5, a51, a52, a53 = \
    data_test.loc[i]
    tem_res.append(lable)
    for j in range (5):
        if j==0:
            tem_res.append(str(q1))
        if j==1:
            tem_res.append(str(q2))
        if j == 2:
            tem_res.append(str(q3))
        if j==3:
            tem_res.append(str(q4))
        if j==4:
            tem_res.append(str(q5))
        tem=list(result[5*i+j])
        test_a=y[tem.index(max(tem))+1]#对应到字典
        tem_res.append(str(test_a[2:-1]))
    res.append(tem_res)
    # res.append(','.join([str(x) for x in tem_res]))
pd.DataFrame(res).to_csv(path+'res.txt', index = None, header = None)
