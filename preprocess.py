import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
df = pd.read_csv('./fer2013/fer2013.csv')
X_train = []
y_train = df['emotion'].as_matrix()
y_train = y_train.reshape(-1,1)
enc = OneHotEncoder(sparse=False)
y_train = enc.fit_transform(y_train)

print (y_train.shape)
print (y_train)

np.save('labels.npy', y_train)
for e in df['pixels']:
    X_train.append(np.array(e.split(' ')).reshape(48,48,1))
np.save('features.npy', np.array(X_train))

print (np.array(X_train).shape)