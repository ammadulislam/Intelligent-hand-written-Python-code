import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from emnist import extract_training_samples, extract_test_samples


x_train, y_train = extract_training_samples('letters')
x_test, y_test = extract_test_samples('letters')


x_train, x_test = x_train / 255.0, x_test / 255.0


base_model = VGG16(weights='imagenet', include_top=False, input_shape=(28, 28, 3))


model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(62, activation='softmax')
])

for layer in base_model.layers:
    layer.trainable = False


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))


model.save('EMNIST_VGG16.h5')
