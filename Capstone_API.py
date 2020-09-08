import pandas as pd
import numpy as np
from matplotlib import image
from skimage.transform import resize
from tensorflow import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#import os
#cwd = os.getcwd()
#print(cwd)
path_train = '../data/train_v2/'
data_root = '../data/'
masks = pd.read_csv(data_root + 'train_ship_segmentations_v2.csv')
print(masks.head())

#loading images
original_image_length = 768
divisions = 8
new_length = int(original_image_length/divisions)

#alocation of memory: creating numpy array of zeros for images and labels
chunk_size = 1000 #total number of images to be loaded
numpy_arr = np.zeros((chunk_size, new_length, new_length, 3))
numpy_labels = np.zeros((chunk_size, 1))

#loading images
counter_image = 0
counter_cycle = 0
initial_image = counter_cycle * chunk_size
final_image = initial_image + chunk_size

for i in range(initial_image,final_image):
    if i in range(0,final_image,10):
        print(i)
    image_ID = masks.iloc[i][0]
    image_FLAG = masks.iloc[i][1]  #this label us whether there is not a ship or where a ship is (through a list of pixels)
    image1 = image.imread(path_train + image_ID)
    image1 = resize(image1, (image1.shape[0] // divisions, image1.shape[1] // divisions, 3), anti_aliasing=True)
    numpy_arr[counter_image] = image1

    if not isinstance(image_FLAG, float):
        numpy_labels[counter_image] = 1
    counter_image += 1

#end of loading images
print(chunk_size, 'images loaded')

#defining the model (Convolutional Neural Network)
model = keras.Sequential([
    keras.layers.Conv2D(20, kernel_size=3, activation='tanh', input_shape=(new_length,new_length,3)),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.Conv2D(10, kernel_size=3, activation='tanh'),
    keras.layers.Flatten(),
#adding drop out to experiment with regularization on next layer:
    keras.layers.Dropout(0.2),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])

#compiling the model.
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])

#splitting the data
X_train, X_test, y_train, y_test = train_test_split(numpy_arr, numpy_labels, test_size=0.3)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)

#introducing an early stopping regime
my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=50)
]
#history = model.fit(X_train, y_train, batch_size=2048, epochs=500, validation_data=(X_val,y_val), callbacks=my_callbacks)
history = model.fit(X_train, y_train, batch_size=256, epochs=500, validation_data=(X_val,y_val), callbacks=my_callbacks)

#saving the model
model.save('../model/model1.h5')

#testing the accuracy on testing/training/validation sets
test_loss, test_acc = model.evaluate(X_test, y_test)
train_loss, train_acc = model.evaluate(X_train, y_train)
val_loss, val_acc = model.evaluate(X_val, y_val)
print("Test loss: %.2f , test accuracy: %.2f"%(test_loss,test_acc))
print("train loss: %.2f , train accuracy: %.2f"%(train_loss,train_acc))
print("validation loss: %.2f , validation accuracy: %.2f"%(val_loss,val_acc))

#plotting evolution of losses with epochs
plt.figure(figsize=(10,5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training loss', 'validation loss'])
plt.xlabel('epoch')
plt.show()

#introducing confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score

predicted = model.predict(X_test)
predicted2 = np.zeros(y_test.shape)

# for i in range(len(y_test[:,0])):
#     if predicted[i][0] < predicted[i][1]:
#         predicted2[i] = 1
#
# cm = confusion_matrix(y_test, predicted2)
# fig, ax = plt.subplots()
# colorbar = ax.matshow(cm)
# fig.colorbar(colorbar)
# fig.savefig('sklearn.png')

print('________y_test________')
print(y_test[10])
print('________predicted________')
print(predicted[10])
