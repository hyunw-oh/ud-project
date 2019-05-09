import numpy as np
import matplotlib.pyplot as plt
import csv
import cv2

csv_path = 'driving_log.csv'  # my da
csv_path1 = 'drive/driving_log.csv'  # udacity data

images_db, center_db, left_db, right_db, steer_db = [], [], [], [], []
valid_center_db, valid_left_db, valid_right_db, valid_steer_db = [], [], [], []
Rows, Cols = 160, 320
offset = 0.2

# read csv file
with open(csv_path1) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        images_db.append([row['center'], row['left'], row['right'], float(row['steering'])])

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# test data : 85% / valid data : 15%
train_samples, validation_samples = train_test_split(images_db, test_size=0.15, shuffle=True)


for row in train_samples:
    center_db.append(row[0])
    left_db.append(row[1])
    right_db.append(row[2])
    steer_db.append(row[3])

for row in validation_samples:
    valid_center_db.append(row[0])
    valid_left_db.append(row[1])
    valid_right_db.append(row[2])
    valid_steer_db.append(row[3])

plt.figure(figsize=(10,4))
x = [range(len(images_db))]
x = np.squeeze(np.asarray(x))

y = np.asarray([x[3] for x in images_db])
plt.title('data distribution', fontsize=17)
plt.xlabel('frames')
plt.ylabel('steering angle')
plt.plot(x,y, 'g', linewidth=0.4)
plt.show()

plt.hist(y, bins= 50, color= 'orange', linewidth=0.1)
plt.title('angle histogram', fontsize=17)
plt.xlabel('steering angle')
plt.ylabel('counts')
plt.show()

images_db = None

#Load the all image binary to memory to imporve learning speed
def read_imgs(img_list):
    images = []
    for img in img_list:
        img = cv2.imread(img)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        images.append(img)
    return images

#load train images on memory all once
center_imgs, center_steerings = read_imgs(center_db), steer_db
left_imgs, left_steerings = read_imgs(left_db), [x+offset for x in steer_db]
right_imgs, right_steerings = read_imgs(right_db), [x-offset for x in steer_db]

#load valid images on memory all once
valid_center_imgs, valid_center_steerings = read_imgs(valid_center_db), valid_steer_db
valid_left_imgs, valid_left_steerings = read_imgs(valid_left_db), [x+offset for x in valid_steer_db]
valid_right_imgs, valid_right_steerings = read_imgs(valid_right_db), [x-offset for x in valid_steer_db]

# Algorithm for augment data
def shift_img(image, steer):
    max_shift = 30
    max_ang = 0.08  # ang_per_pixel = 0.0025

    rows, cols, _ = image.shape

    random_x = np.random.randint(-max_shift, max_shift + 1)
    dst_steer = steer + (random_x / max_shift) * max_ang
    if abs(dst_steer) > 1:
        dst_steer = -1 if (dst_steer < 0) else 1

    mat = np.float32([[1, 0, random_x], [0, 1, 0]])
    dst_img = cv2.warpAffine(image, mat, (cols, rows))
    return dst_img, dst_steer

def brightness_img(image):
    br_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    coin = np.random.randint(2)
    if coin == 0:
        random_bright = 0.2 + np.random.uniform(0.2, 0.6)
        br_img[:, :, 2] = br_img[:, :, 2] * random_bright
    br_img = cv2.cvtColor(br_img, cv2.COLOR_HSV2RGB)
    return br_img

def generate_shadow(image):
    dark_mul = np.random.uniform(0.3, 0.6)
    prob = np.random.randint(2)
    rows, cols,_ = image.shape
    if prob == 0:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        x = np.random.randint(0, cols)
        y = np.random.randint(0, rows)
        xoffset = np.random.randint(int(cols/2),cols)
        if(x+ xoffset > cols ):
            x = cols - x
        yoffset = np.random.randint(int(rows/2),rows)
        if(y + yoffset > rows):
            y = rows - y
        image[y:y+yoffset,x:x+xoffset,2] = image[y:y+yoffset,x:x+xoffset,2]*dark_mul
        image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    return image


def flip_img(image, steering):
    flip_image, flip_steering = cv2.flip(image, 1), -steering
    return flip_image, flip_steering

from sklearn.utils import shuffle
def augment_data3(center_imgs,left_imgs,right_imgs,center_steerings,left_steerings,right_steerings):
    augmented_images = []
    augmented_steers = []
    # Shuffle the data, so that the train is not affected by the order
    shuffle(center_imgs, center_steerings)
    shuffle(left_imgs, left_steerings)
    shuffle(right_imgs, right_steerings)

    length = len(center_imgs)
    for i in range(length):
        img, steer = center_imgs[i] , center_steerings[i]
        rand = np.random.randint(3)
        # augment center image
        if rand == 0:
            img= brightness_img(img)
        elif rand == 1:
            img= generate_shadow(img)
        augmented_images.append(img)
        augmented_steers.append(steer)
        img, steer = shift_img(center_imgs[i] , center_steerings[i])
        img, steer = flip_img(img, steer)
        augmented_images.append(img)
        augmented_steers.append(steer)
        # augment left image
        img, steer = left_imgs[i] , left_steerings[i]
        rand = np.random.randint(3)
        if rand == 0:
            img= brightness_img(img)
        elif rand == 1:
            img= generate_shadow(img)
        augmented_images.append(img)
        augmented_steers.append(steer)
        img, steer = shift_img(left_imgs[i] , left_steerings[i])
        img, steer = flip_img(img, steer)
        img = brightness_img(img)
        img = generate_shadow(img)
        augmented_images.append(img)
        augmented_steers.append(steer)
        # augment right image
        img, steer = right_imgs[i] , right_steerings[i]
        rand = np.random.randint(3)
        if rand == 0:
            img= brightness_img(img)
        elif rand == 1:
            img= generate_shadow(img)
        augmented_images.append(img)
        augmented_steers.append(steer)
        img, steer = shift_img(right_imgs[i], right_steerings[i])
        img, steer = flip_img(img, steer)
        img = brightness_img(img)
        img = generate_shadow(img)
        augmented_images.append(img)
        augmented_steers.append(steer)
    return augmented_images, augmented_steers

# It is executed every batch
def generator(imgs, steerings, batch_size):
    length = len(imgs)
    while 1:
        for offset in range(0, length, batch_size):
            lim = len(center_imgs)
            if(offset + batch_size < lim):
                yield sklearn.utils.shuffle(np.array(imgs[offset:offset+batch_size]),
                                        np.array(steerings[offset:offset+batch_size]))

# data augmantaion
train_X,train_y = augment_data3(center_imgs,left_imgs,right_imgs,center_steerings,left_steerings,
                          right_steerings)
valid_X, valid_y = augment_data3(valid_center_imgs,valid_left_imgs,valid_right_imgs,valid_center_steerings,valid_left_steerings
                          ,valid_right_steerings)

# batch logic for learning
train_data = generator(train_X,train_y,64)
valid_data = generator(valid_X,valid_y,64)

from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Activation, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers import Lambda, Cropping2D

# this network is based on nvidia network
model = Sequential()

# zero center the image
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

# trim image to only see section with road
model.add(Cropping2D(cropping=((70,25),(0,0))))

model.add(Convolution2D(24,5,5,subsample=(2,2)))
model.add(Activation('elu'))
model.add(Convolution2D(36,5,5,subsample=(2,2)))
model.add(Activation('elu'))
model.add(Convolution2D(48,5,5,subsample=(2,2)))
model.add(Activation('elu'))
model.add(Convolution2D(64,3,3))
model.add(Activation('elu'))
model.add(Convolution2D(64,3,3))
model.add(Activation('elu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Activation('elu'))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Activation('elu'))
model.add(Dense(10))
model.add(Activation('elu'))
model.add(Dense(1))
#output is steering angle

# Mean Squared Error : for Regression
# adam optimizer : good solution for most cases
model.compile(loss='mse',optimizer='adam')

# keras method to print the model summary
model.summary()

from sklearn.utils import shuffle
import sklearn
model.fit_generator(train_data,
                    samples_per_epoch=len(center_db),
                    validation_data=valid_data,
                    nb_val_samples=len(valid_center_db),
                    nb_epoch=1,
                    verbose=1)

# saving model
model.save('model.h5')
