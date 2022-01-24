import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import app

batch_size = 8

training_data_generator = ImageDataGenerator(rescale = 1./255, zoom_range = 0.2, rotation_range = 15, width_shift_range= 0.05, height_shift_range = 0.05)

training_iterator = training_data_generator.flow_from_directory('augmented-data/train', color_mode = 'grayscale', class_mode = 'categorical', batch_size = batch_size)

validation_data_generator = ImageDataGenerator(rescale = 1./255)

validation_iterator = validation_data_generator.flow_from_directory('augmented-data/test', color_mode = 'grayscale', class_mode = 'sparse', batch_size = batch_size)

model = Sequential()

model.add(tf.keras.Input(shape = tf.shape(validation_iterator)[1:]))

model.add(layers.Conv2D(8, 3, 2, activation = 'relu'))
model.add(tf.keras.layers.MaxPooling2D(2,2))
model.add(tf.keras.layers.Conv2D(8, 3, 2, activation = 'relu'))
model.add(tf.keras.layers.MaxPooling2D(2,2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation="relu"))
model.add(tf.keras.layers.Dense(16, activation = 'relu'))

model.add(layers.Dense(3, activation = 'relu'))

print(model.summary())

model.compile(optimizer = 'adam', learning_rate = 0.01, loss = 'sparse_categorical_crossentropy', metric = [tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.AUC()])

history = model.fit(training_iterator, steps_per_epoch = len(training_iterator) / 5, epochs = 20, validation_data = validation_iterator, validation_steps = len(validation_iterator))

# Do Matplotlib extension below
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(history.history['categorical_accuracy'])
ax1.plot(history.history['val_categorical_accuracy'])
ax1.set_title('model accuracy')
ax1.set_xlabel('epoch')
ax1.set_ylabel('accuracy')
ax1.legend(['train', 'validation'], loc='upper left')
 
# plotting auc and validation auc over epochs
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(history.history['auc'])
ax2.plot(history.history['val_auc'])
ax2.set_title('model auc')
ax2.set_xlabel('epoch')
ax2.set_ylabel('auc')
ax2.legend(['train', 'validation'], loc='upper left')
 
# used to keep plots from overlapping
fig.tight_layout()

# use this savefig call at the end of your graph instead of using plt.show()
fig.show()
plt.savefig('static/images/my_plots.png')