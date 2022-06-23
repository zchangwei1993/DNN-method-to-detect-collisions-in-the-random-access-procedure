#Class the received RAPs into four categories, i.e., no MTD choose the RAP, one MTD choose the RAP, Two MTD choose the RAP and three MTDs choose the RAP.

import tensorflow as tf
import scipy.io as scio
import numpy as np
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import  Dense,Input,Flatten,Dropout,Reshape
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ReduceLROnPlateau


Total_point = 1118

# 数据读取
data = np.load('D:\data_set_-5db.npz')
Data_X = data['Data_X']
Data_Y = data['Data_Y']

# 预处理
Data_X = np.hstack((Data_X.real, Data_X.imag))
Data_X = Data_X[:, : :3]
x_mean = np.mean(Data_X, axis = 0)
x_std = np.std(Data_X, axis = 0)
Data_X = [(x - x_mean)/x_std for x in Data_X]
Data_X = np.reshape(Data_X, (40000, -1))

# 训练集测试集
Train_x = Data_X[:30000, :]
Train_y = Data_Y[:30000, :]
Test_x = Data_X[30000:, :]
Test_y = Data_Y[30000:, :]


input = Input(shape=(Total_point,))
#input11 = Dropout(0.1)(input)
Dense1 = Dense(Total_point,activation='relu',kernel_initializer='he_normal')(input)
#Dense11 = Dropout(0.3)(Dense1)
BN1 = BatchNormalization(axis=-1)(Dense1)
Dense2 = Dense(Total_point,activation='relu')(BN1)
#Dense22 = Dropout(0.3)(Dense2)
BN2 = BatchNormalization(axis=-1)(Dense2)
Dense3 = Dense(Total_point,activation='relu')(BN2)
#Dense33 = Dropout(0.3)(Dense3)
BN3 = BatchNormalization(axis=-1)(Dense3)
Dense4 = Dense(Total_point,activation='relu')(BN3)
#Dense44 = Dropout(0.3)(Dense4)
BN4 = BatchNormalization(axis=-1)(Dense4)
Dense5 = Dense(Total_point,activation='relu')(BN4)
Dense6 = Dense(Total_point,activation='relu')(Dense5)
Output = Dense(4,activation='softmax')(Dense6)
# BN3 = BatchNormalization(axis=-1)(Dense1)(Dense3)
model = Model(inputs=input, outputs=Output)

adam = Adam(lr=0.01, decay=1e-6)
model.compile(optimizer='adagrad', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
#reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')
model.fit(Train_x, Train_y,
          epochs = 80, batch_size = 200,
          validation_data=(Test_x,Test_y))

model.save_weights('model0.h5')
model_json = model.to_json()
with open('model0.json','w') as json_file:
    json_file.write(model_json)
json_file.close()

preds = model.evaluate(Test_x, Test_y)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

# net = network4.Network([3354,2000,1000,100, 4], cost=networ+k4.CrossEntropyCost)
# net.large_weight_initializer()
# net.SGD(training_data, 30, 10, 0.1, lmbda = 5.0,evaluation_data=validation_data,
#     monitor_evaluation_accuracy=True)
