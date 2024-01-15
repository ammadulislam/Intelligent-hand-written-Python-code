import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

from trainedModel.EmnistModel import num_classes

train_df = pd.read_csv('C:\\Users\\HP\\PycharmProjects\\Fyp_Final\\trainedModel\\train.csv')  # Replace 'path_to_train.csv' with the actual path
test_df = pd.read_csv('C:\\Users\\HP\\PycharmProjects\\Fyp_Final\\trainedModel\\test.csv')  # Replace 'path_to_test.csv' with the actual path


num_classes = len(train_df['label'].unique())


X_train = np.array([np.fromstring(image, dtype=int, sep=' ') for image in train_df['image']])
y_train = to_categorical(train_df['label'], num_classes=num_classes)

X_test = np.array([np.fromstring(image, dtype=int, sep=' ') for image in test_df['image']])
y_test = to_categorical(test_df['label'], num_classes=num_classes)

X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


early_stopping = EarlyStopping(monitor='val_loss', patience=5)
model.fit(X_train, y_train, batch_size=128, epochs=50, validation_split=0.2, callbacks=[early_stopping])
model.save('D:\\FINAL YEAR Project\\Emnist\\Emnist_model.h5')

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
