import cv2
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping


def load_images_and_labels(base_dir):
    images = []
    labels = []
    class_folders = os.listdir(base_dir)
    label_encoder = LabelEncoder()

    for class_folder in class_folders:
        class_path = os.path.join(base_dir, class_folder)
        class_label = class_folder  # Use the folder name as the label
        label_encoded = label_encoder.fit_transform([class_label])[0]

        for image_file in os.listdir(class_path):
            image_path = os.path.join(class_path, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (28, 28))  # Resize the image to match the model's input size
            images.append(image)
            labels.append(label_encoded)

    return np.array(images), np.array(labels)


train_data_dir = 'D:\\dataset\\Train'
validation_data_dir = 'D:\\dataset\\Test'

X_train, y_train = load_images_and_labels(train_data_dir)
X_validation, y_validation = load_images_and_labels(validation_data_dir)


X_train = X_train / 255.0
X_validation = X_validation / 255.0

num_classes = len(np.unique(y_train))
y_train = to_categorical(y_train, num_classes)
y_validation = to_categorical(y_validation, num_classes)


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

model.fit(X_train, y_train, batch_size=128, epochs=50, validation_data=(X_validation, y_validation), callbacks=[early_stopping])


model.save('D:\\FINAL YEAR Project\\Emnist\\Emnist_model.h5')


score = model.evaluate(X_validation, y_validation, verbose=0)
print('Validation loss:', score[0])
print('Validation accuracy:', score[1])
