import pandas as pd

df = pd.read_csv('diabetes.csv')

dataset_x = df.iloc[:, :-1].to_numpy() 
dataset_y = df.iloc[:, -1].to_numpy() 

from sklearn.model_selection import train_test_split

training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x, dataset_y, random_state=1234)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import GlorotUniform

model = Sequential(name='Sample')
model.add(Dense(32, activation='relu', input_dim=dataset_x.shape[1], kernel_initializer=GlorotUniform(seed=1234), name='Hidden-1'))
model.add(Dense(16, activation='relu', kernel_initializer=GlorotUniform(seed=1234), name='Hidden-2'))
model.add(Dense(1, activation='sigmoid', kernel_initializer=GlorotUniform(seed=1234), name='Output'))

model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['binary_accuracy'])
model.fit(training_dataset_x, training_dataset_y, batch_size=32, epochs=200, validation_split=0.2, shuffle=False)

eval_result = model.evaluate(test_dataset_x, test_dataset_y)

for i in range(len(eval_result)):
    print(f'{model.metrics_names[i]}: {eval_result[i]}')

import numpy as np

predict_data  = np.array([[2,148,58,37,128,25.4,0.699,24], [7,114,67,0,0,32.8,0.258,42], 
[5,99,79,27,0,29,0.203,32], [0,110,85,30,0,39.5,0.855,39]])

predict_result = model.predict(predict_data)

for i in range(len(predict_result)):
    print(predict_result[i, 0])

for i in range(len(predict_result)):
    print("Şeker Hastası" if predict_result[i, 0] > 0.5 else "Şeket Hastası Değil")
