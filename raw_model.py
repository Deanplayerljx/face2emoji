import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten,Dropout, BatchNormalization
from keras.optimizers import Adam
import numpy as np

features = np.load('./features.npy')
labels = np.load('./labels.npy')
features = np.array(features, dtype=np.float32)
labels = np.array(labels, dtype=int)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)
print(len(X_train))
print(len(X_val))
print(len(X_test))

inputs = Input(X_train.shape[1:])
x = Conv2D(kernel_size=3, filters=32, strides=1, padding='same', activation='relu')(inputs)
x = BatchNormalization()(x)
x = Conv2D(kernel_size=3, filters=32, strides=1, padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = Conv2D(kernel_size=3, filters=32, strides=1, padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPool2D(pool_size=2, strides=2)(x)

x = Conv2D(kernel_size=3, filters=64, strides=1, padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = Conv2D(kernel_size=3, filters=64, strides=1, padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = Conv2D(kernel_size=3, filters=64, strides=1, padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPool2D(pool_size=2, strides=2)(x)

x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.4)(x)
outputs = Dense(7, activation='softmax')(x)
model = Model(inputs, outputs)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer= Adam(),metrics=['categorical_accuracy'])
model.fit(X_train, y_train, batch_size=128, epochs=40, validation_data=[X_val, y_val])
model.save('./cnn_model')
print (model.evaluate(x=X_test, y=y_test, batch_size=128))
