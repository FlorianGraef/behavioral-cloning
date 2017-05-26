import csv
import random

import cv2
import numpy as np
import keras
from keras.models import Sequential
from keras import optimizers
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPool2D, Cropping2D, Dropout, ELU, MaxPooling2D, \
    Convolution2D

img_dim = (64, 64, 3)


# nvidia model
def model_nvidia():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=img_dim))
    model.add(Cropping2D(
        cropping=((int(img_dim[0] * 0.43), int(img_dim[0] * 0.15)), (0, 0))
    )
    )
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="elu"))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="elu"))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="elu"))
    model.add(Convolution2D(64, 3, 3, activation="elu"))
    model.add(Convolution2D(64, 3, 3, activation="elu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model


# build extended LeNet
def model_Lenet():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=img_dim))  # normalize image data
    model.add(Cropping2D(  # crop off sky and hood of the car
        cropping=((int(img_dim[0] * 0.43), int(img_dim[0] * 0.15)), (0, 0))
    )
    )
    model.add(Conv2D(32, 5, 5, activation='relu'))
    model.add(MaxPool2D())
    model.add(Conv2D(64, 5, 5, activation='relu'))
    model.add(MaxPool2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    return model


lines = []  # list for csv file lines


# read CSV file into lines list
def read_csv(file):
    global line
    with open(file) as csvfile:
        reader = csv.reader(csvfile, skipinitialspace=True)
        for line in reader:
            lines.append(line)


# use recorded training data from several training sessions
csv_files = ['F:\SDCarND\simulator\DATA\\basic.csv',
             'F:\SDCarND\simulator\\new_baseline\driving_log.csv',
             'F:\SDCarND\simulator\\bridge_train\\bridge.csv',
             'F:\SDCarND\simulator\\recovey\driving_log.csv']

# read all training data
for file in csv_files:
    read_csv(file)

images = []
steer_angles = []
angle_correction = 0.2  # correction offset for left and right camera images
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # prepare adaptive histogram equalization

# use data from all thre cameras
for line in lines:
    angle = float(line[3])
    speed = float(line[6])
    if speed <= 0.5:
        continue

    steer_angles.append(angle)
    # apply correction factorrs for images of side cameras
    steer_angles.append(angle + angle_correction)
    steer_angles.append(angle - angle_correction)

    for col_num in range(3):
        img_path = line[col_num]
        image = cv2.imread(img_path)

        # resize all images for faster training
        image = cv2.resize(image, (img_dim[0], img_dim[1]))

        # adaptive histogram equalization
        b, g, r = cv2.split(image)
        b = clahe.apply(b)
        g = clahe.apply(g)
        r = clahe.apply(r)

        image = cv2.merge([b, g, r])
        images.append(image)

augmented_images = []
augmented_steer_angles = []

# augment data + balance steering angle distribution by flipping images and steering angles
for image, steer_angle in zip(images, steer_angles):
    augmented_images.append(image)
    augmented_steer_angles.append(steer_angle)
    flipped_image = cv2.flip(image, 1)
    flipped_steer_angle = float(steer_angle) * -1.0
    augmented_images.append(flipped_image)
    augmented_steer_angles.append(flipped_steer_angle)

X_train = np.array(augmented_images)
y_train = np.array(augmented_steer_angles)

# statistics of the augmented dataset
print("Dataset length: ")
print(len(X_train))

# histogram of steering angles
hist, edges = np.histogram(y_train, 30, (-1., 1.))
import matplotlib.pyplot as plt

plt.hist(y_train, bins=40)
plt.title("Steering angle histogram")
plt.xlabel("Angle")
plt.ylabel("Frequency")
plt.show()

# build the neural network
model = model_Lenet()

model_name = "model_nv4"
adam = keras.optimizers.Adam(lr=0.0001)  # use admam optimizerer with lower, non-default, learning rate

# compile the model with custom adam, optimizer and  mean squared error loss function
model.compile(optimizer=adam, loss='mse')

# configure checkpoint callback function to save the model whenever it improves the validation accuracy
checkpoint_cb = keras.callbacks.ModelCheckpoint('.\\models\\' + model_name + '-{epoch:02d}-{val_loss:.4f}.h5',
                                                monitor='val_loss', verbose=0, save_best_only=True,
                                                save_weights_only=False, mode='auto', period=1)

# stop the training early to avoid overfitting and save training time
early_stop_cb = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')

# fit the model with 20% validation set, shuffle date and use the callbacks
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=20, verbose=2,
          callbacks=[checkpoint_cb, early_stop_cb])
model.save(model_name + '.h5')
