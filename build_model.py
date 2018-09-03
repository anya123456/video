import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Input, LSTM, Embedding, Dense,Dropout
from keras.models import Model, Sequential
from keras.preprocessing import image
from keras.preprocessing.text import Tokenizer
from keras.applications.vgg16 import VGG16
from keras.layers.convolutional import  ZeroPadding2D
#vision_model = VGG16()
# 首先，让我们用 Sequential 来定义一个视觉模型。
# # 这个模型会把一张图像编码为向量。
# vision_model = Sequential()
# vision_model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)))
# vision_model.add(Conv2D(64, (3, 3), activation='relu'))
# vision_model.add(MaxPooling2D((2, 2)))
# vision_model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
# vision_model.add(Conv2D(128, (3, 3), activation='relu'))
# vision_model.add(MaxPooling2D((2, 2)))
# vision_model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
# vision_model.add(Conv2D(256, (3, 3), activation='relu'))
# vision_model.add(Conv2D(256, (3, 3), activation='relu'))
# vision_model.add(MaxPooling2D((2, 2)))
# vision_model.add(Flatten())
model = Sequential()
model.add(ZeroPadding2D((1, 1), input_shape=(224, 224,3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(128,(3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(128,(3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(256,(3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1000, activation='softmax'))


# 现在让我们用视觉模型来得到一个输出张量：
image_input = Input(shape=(224, 224, 3))
encoded_image = model(image_input)
print("build vidion model end")

# 接下来，定义一个语言模型来将问题编码成一个向量。
# 每个问题最长 100 个词，词的索引从 1 到 9999.
question_input = Input(shape=(100,), dtype='int32')
embedded_question = Embedding(input_dim=10000, output_dim=256, input_length=100)(question_input)
encoded_question = LSTM(256)(embedded_question)
print("build encoded_question end")
# 连接问题向量和图像向量：
merged = keras.layers.concatenate([encoded_question, encoded_image])

# 然后在上面训练一个 1000 词的逻辑回归模型：
output = Dense(1000, activation='softmax')(merged)

# 最终模型：
vqa_model = Model(inputs=[image_input, question_input], outputs=output)
vqa_model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
vqa_model.summary()

########################################################################################
# creat train data
import pandas as pd
import numpy as np

path = 'E:\\data\\video\\VQADatasetA_20180815\\'
data_train = pd.read_csv(path + 'train.txt', header=None)
#data_train = data_train#[:2]  # !!!!!!!!!注意这里我只上传了2个图片，线下需修改
length = len(data_train) * 15//3
x_img = np.zeros((length, 224, 224, 3))
x_q = []
x_a = []

for i in range(len(data_train)):
    lable, q1, a11, a12, a13, q2, a21, a22, a23, q3, a31, a32, a33, q4, a41, a42, a43, q5, a51, a52, a53 = \
    data_train.loc[i]#给问题和答案标签
    img = image.load_img(path + 'image\\train\\' + lable + '.jpg')
    i = i * 15//3
    for j in range(15//3):
        x_img[i + j] = image.img_to_array(img)
    x_q.append(str(q1))
    x_q.append(str(q2))
    x_q.append(str(q3))
    x_q.append(str(q4))
    x_q.append(str(q5))

    x_a.append(a11)
    x_a.append(a21)
    x_a.append(a31)
    x_a.append(a41)
    x_a.append(a51)
    # [x_q.append(str(q1)) for j in range(3)]
    # [x_q.append(str(q2)) for j in range(3)]
    # [x_q.append(str(q3)) for j in range(3)]
    # [x_q.append(str(q4)) for j in range(3)]
    # [x_q.append(str(q5)) for j in range(3)]
    #
    # x_a.append(a11)
    # x_a.append(a12)
    # x_a.append(a13)
    # x_a.append(a21)
    # x_a.append(a22)
    # x_a.append(a23)
    # x_a.append(a31)
    # x_a.append(a32)
    # x_a.append(a33)
    # x_a.append(a41)
    # x_a.append(a42)
    # x_a.append(a43)
    # x_a.append(a51)
    # x_a.append(a52)
    # x_a.append(a53)

print("creat data end.")
from nltk.probability import FreqDist
from collections import Counter

#答案编码
tmp = x_a.copy()
fdist1 = FreqDist(tmp)
y_dist = fdist1.most_common(999)
y_di = [ytemp[0] for ytemp in y_dist]
train_a = []
for name in x_a:
    if name in y_di:
        train_a.append(y_di.index(name) + 1)
    else:
        train_a.append(0)

#问题编码
tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_q)
encoded = tokenizer.texts_to_sequences(x_q)
size = len(tokenizer.word_index) + 1

train_q = np.zeros((len(encoded), 1000))
for i in range(len(encoded)):
    train_q[i][:len(encoded[i])] = encoded[i]

print('question encode size = ' + str(size))
print(train_q)
print(y_dist)
#np.savetxt('E:\\data\\video\\VQADatasetA_20180815\\y_di.csv',y_di,delimiter=',')


#train the model ##自己线下跑时训练轮数多写点
from keras.utils import to_categorical
Y = to_categorical(train_a, num_classes= 1000)
h = vqa_model.fit([x_img,train_q], Y, epochs=3, batch_size=32)

vqa_model.save(path + "model_vgg.h5")
##########################################################
