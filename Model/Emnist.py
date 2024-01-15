import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

data_folder = 'E:\\dataset'

preprocessed_images = []
labels = []

characters = '123ABCabc'

label_mapping = {char: i for i, char in enumerate(characters)}

for character_folder in os.listdir(data_folder):
    character_path = os.path.join(data_folder, character_folder)

    if os.path.isdir(character_path) and character_folder in label_mapping:
        label = label_mapping[character_folder]

        for image_file in os.listdir(character_path):
            image_path = os.path.join(character_path, image_file)

            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (28, 28))  # Remove the ", 1"
            image = image / 255.0

            preprocessed_images.append(image)
            labels.append(label)

train_data_folder = 'E:\\dataset\\Train'
validation_data_folder = 'E:\\dataset\\Test'

# Initialize lists for preprocessed images and labels for training and validation
train_preprocessed_images = []
train_labels = []
validation_preprocessed_images = []
validation_labels = []

for data_folder in [train_data_folder, validation_data_folder]:
    preprocessed_images = []
    labels = []

    for character_folder in os.listdir(data_folder):
        character_path = os.path.join(data_folder, character_folder)

        if os.path.isdir(character_path) and character_folder in label_mapping:
            label = label_mapping[character_folder]

            for image_file in os.listdir(character_path):
                image_path = os.path.join(character_path, image_file)

                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (28, 28))  # Remove the ", 1"
                image = image / 255.0

                if data_folder == train_data_folder:
                    train_preprocessed_images.append(image)
                    train_labels.append(label)
                else:
                    validation_preprocessed_images.append(image)
                    validation_labels.append(label)

train_preprocessed_images = np.array(train_preprocessed_images)
train_labels = np.array(train_labels)
validation_preprocessed_images = np.array(validation_preprocessed_images)
validation_labels = np.array(validation_labels)

X_train, X_valid, y_train, y_valid = train_test_split(train_preprocessed_images, train_labels, test_size=0.1,
                                                      random_state=42)

X_train = np.expand_dims(X_train, axis=-1)
X_valid = np.expand_dims(X_valid, axis=-1)

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,
    fill_mode='nearest'
)

datagen.fit(X_train)

model = keras.Sequential([
    keras.layers.Input(shape=(28, 28, 1)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(len(characters), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=10, validation_data=(X_valid, y_valid))

test_loss, test_acc = model.evaluate(X_valid, y_valid)
print(f'Validation accuracy: {test_acc * 100:.2f}%')

model.save('C:\\Users\\HP\\PycharmProjects\\Emnist_model.h5')
