# Set system settings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import dependencies
import numpy as np
# from keras import regularizers
from keras.models import Sequential
from keras.layers import Activation, Dropout, UpSampling2D, Conv2D, Conv2DTranspose, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pickle

# Constants
train_data_filename = "train_data.p"
train_labels_filename = "train_labels.p"
INDEX_RANGE_RATE = 0.25
TEST_SIZE = 0.1
BATCH_SIZE = 128
EPOCHS = 8
POOL_SIZE = (2, 2)

# Load training images and labels from pickle file, return as NumPy array
print("Loading training data/images...")
train_images = np.array(pickle.load(open(train_data_filename, 'rb')))
print("Loading training labels...")
train_labels = np.array(pickle.load(open(train_labels_filename, 'rb')))

# Shuffle data
print("Shuffling training data...")
train_images, train_labels = shuffle(train_images, train_labels)

# Only use limited amount of training data samples
print("Limiting data range to", int(train_images.shape[0] * INDEX_RANGE_RATE), "out of", train_images.shape[0], "samples...")
train_images = train_images[0:int(train_images.shape[0] * INDEX_RANGE_RATE)]
train_labels = train_labels[0:int(train_labels.shape[0] * INDEX_RANGE_RATE)]

# Normalize labels
print("Normalizing training data labels...")
train_labels = train_labels / 255

# Split training data into training and test data (test_size is amount as percentage)
print("Splitting training data into training and testing data...")
X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=TEST_SIZE)
input_shape = X_train.shape[1:]

# Define neural network architecture
print("Defining model structure...")
# Use sequential architecture
model = Sequential()

# Add layers
model.add(BatchNormalization(input_shape=input_shape))

model.add(Conv2D(1, (3, 3), padding='valid', strides=(1, 1), activation='relu'))
model.add(Dropout(0.25))
model.add(Conv2D(1, (3, 3), padding='valid', strides=(1, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=POOL_SIZE))

model.add(Conv2D(8, (3, 3), padding='valid', strides=(1, 1), activation='relu'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=POOL_SIZE))

model.add(Conv2D(16, (3, 3), padding='valid', strides=(1, 1), activation='relu'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=POOL_SIZE))

model.add(Conv2D(32, (3, 3), padding='valid', strides=(1, 1), activation='relu'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=POOL_SIZE))

model.add(UpSampling2D(size=POOL_SIZE))
model.add(Conv2DTranspose(32, (3, 3), padding='valid', strides=(1, 1), activation='relu'))
model.add(Dropout(0.25))

model.add(UpSampling2D(size=POOL_SIZE))
model.add(Conv2DTranspose(16, (3, 3), padding='valid', strides=(1, 1), activation='relu'))
model.add(Dropout(0.25))

model.add(UpSampling2D(size=POOL_SIZE))
model.add(Conv2DTranspose(8, (3, 3), padding='valid', strides=(1, 1), activation='relu'))
model.add(Dropout(0.25))

model.add(UpSampling2D(size=POOL_SIZE))
model.add(Conv2DTranspose(1, (3, 3), padding='valid', strides=(1, 1), activation='relu'))
model.add(Dropout(0.25))
model.add(Conv2DTranspose(1, (3, 3), padding='valid', strides=(1, 1), activation='relu'))

# Compile the model
model.compile(optimizer='Adam', loss='mean_squared_error')

# Add channel shift to training dataset
datagen = ImageDataGenerator(channel_shift_range=0.1)
datagen.fit(X_train)

# Train model
model.fit_generator(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    steps_per_epoch=len(X_train) / BATCH_SIZE,
    epochs=EPOCHS,
    verbose=1,
    validation_data=(X_val, y_val)
)

# Freeze layers since training is done, save model structure and weights
model.trainable = False
model.compile(optimizer='Adam', loss='mean_squared_error')
model.save('model.h5')

# Show summary of model
model.summary()

# Evaluate model
print(model.evaluate(X_val, y_val, batch_size=128))