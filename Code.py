# Importing libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import time

######################################################################################

# Loading train and test dataset

(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

######################################################################################

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

######################################################################################

# Transformacija skupa podataka u odgovarajuci skup podataka za TensorFlow
# Ovo je potrebno uraditi jer TensorFlow obezbedjuje skup funkcija i operacija koje omogucavaju laksu manipulaciju i modifikaciju podataka

train_ds = tf.data.Dataset.from_tensor_slices((train_images,train_labels))
test_ds = tf.data.Dataset.from_tensor_slices((test_images,test_labels))

######################################################################################

# Visualization of CIFAR10 dataset

plt.figure(figsize=(30, 30))
for i, (image, label) in enumerate(train_ds.take(20)):
    ax=plt.subplot(5, 5, i+1)
    plt.imshow(image)
    plt.title(CLASS_NAMES[label.numpy()[0]])
    plt.axis('off')
    
######################################################################################

# Preprocessing data

def process_image(image, label):
    image = tf.image.per_image_standardization(image)
    image = tf.image.resize(image, (64, 64))
    return image, label

######################################################################################

# Data Pipeline

train_ds_size=tf.data.experimental.cardinality(train_ds).numpy()
test_ds_size=tf.data.experimental.cardinality(test_ds).numpy()
print('Train size:',train_ds_size)
print('Test size:',test_ds_size)

# batch_size predstavlja parametar koji je jednak broju ulaza koji se propustaju kroz mrezu,
# pre nego sto se uradi back propagation (podesavanje tezinskih koeficijenata i pragova aktivacije). Moguce vrednosti su 
# 1, 16, 32, 256, 1024,... (batch_size >= 1 AND batch_size <= sizeOfTraningSet)
# Sto je vrednost manja, vece je vreme obuke, veca je racunarska snaga, ali je i tacnost veca.

train_ds=(train_ds
          .map(process_image)
          .shuffle(buffer_size=train_ds_size)
          .batch(batch_size=32,drop_remainder=True)
         )

test_ds=(test_ds
          .map(process_image)
          .shuffle(buffer_size=test_ds_size)
          .batch(batch_size=32,drop_remainder=True)
         )

######################################################################################

# Model Implementation

# BatchNormalization - use for normalize and standardize the input values
# Normalize pixel - vrednost svakog piksela se deli brojem 255, tj. vrednost svakog piskela se svodi s opsega 0-255 na opseg od 0-1
# Standardize pixel - od trenutne vrednosti piksela oduzima se srednja vrednost piksela. a zatim se nova vrednost piksela deli vrednošću standradne devijacije piksela

numOfClasses = len(CLASS_NAMES)
 
model = keras.models.Sequential([
    keras.layers.Conv2D(filters=128, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(64,64,3)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(2,2)),
    keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3)),
    keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(1024, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1024, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(numOfClasses, activation='softmax')
    ])

######################################################################################

# Compiling

# loss - funkcija gubitka (troskova).
#      - Koristi se za izracunavanje vrednosti poznate kao gubitak. 
#      - Sto je gubitak manji, to su performanse bolje.
#      - Unakrsna entropija (cross entropy) je jedna od najpozantijih f-ja gubitaka.
# lr - learning rate - koliko brzo mreza uci.
#    - Od ovog parametra zavisi gubitak.
#    - Ako je ova vrednost previsoka, tezine veza i pragovi aktivacije ce se prilagodjavati na svakom prolazu.
#    - Ako je ova vrednost preniska, prilagodjavanje ce biti veoma sporo.
#    - Bilo koji uslov od gornja dva moze dovesti da mreza nikada ne konvergira (da ne pronalazi rezultat).

# Neki od nacina pronalazenja najbolje vrednosti za lr:
#   - Prvi nacin je koriscenje konstantne vrednosti za stopu ucenja.
#   - Drugi nacin je smanjivanje vrednosti za stopu ucenja tokom vremena. Na primer, za prvih 10 epoha,
#     moze se koristiti vrednost 0.001 za stopu ucenja. U epohi 11 spusta se stopa ucenja na 0.0001. U epohi
#     20 pada na 0.00001. Razlog za smanjenje brzine ucenja tokom vremena je taj sto prilagodjavanja tezina postaju
#     manje promenljiva kako mreza uci. To je kao kad se sisas. Frizer prvo odsece dugu kosu. Zatim vrsi neka podesavanja 
#     i sece krace duzine. Na kraju, pravi vrlo male rezove oko usiju za konacno poliranje.
#   - Treci nacin je da uzmemo visoku stopu ucenja od 0.1, a zatim nisku od 1e-7. Stope ucenja obicno variraju izmedju 0.1 i 1e-7,
#     ali najcesce nisu ove krajnje vrednosti nego neka izmedju. Zatim mozemo pokusati sa vrednoscu 0.01 i 1e-6 itd. Uporediti 
#     rezultate i pronaci najbolji rezultat.

# Dva najkoriscenija algoritma za optimizaciju hiperparametara su Astochastic Gradient Descent (SGD) optimizator i Adam optimizator.

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.optimizers.SGD(lr=0.001),
    metrics=['accuracy']    
)

print("Model Summary of AlexNet:")
print(model.summary())

# Training

# epochs - predstavlja parametar koji kaze koliko ce se puta proci kroz ceo skup podataka za obuku
# (epochs >= 1 AND epochs <= infinity)

# Pr. Pretpostavimo da imamo skup podataka od 200 uzoraka i da je batch_size=5 i epochs=1000.
#     To znači da će skup podataka biti podeljen u 40 grupa, svaka po 5 uzoraka. Težinski koef. i pragovi aktivacije će
#     biti ažurirani nakon svake serije od pet uzoraka. To takodje znaci da ce jedna epoha ukljucivati 40 serija ili 40 azuriranja modela.
#     Sa 1000 epoha, model ce proci kroz ceo skup podataka 1000 puta. To je ukupno 40000 serija tokom celog procesa obuke.

history=model.fit(train_ds, epochs=25, validation_data=test_ds, validation_freq=1)

######################################################################################

# Visualization of training over the different epochs

f,ax=plt.subplots(2,1,figsize=(10,10)) 

#Assigning the first subplot to graph training loss and validation loss
ax[0].plot(model.history.history['loss'],color='b',label='Training Loss')
ax[0].plot(model.history.history['val_loss'],color='r',label='Validation Loss')

#Plotting the training accuracy and validation accuracy
ax[1].plot(model.history.history['accuracy'],color='b',label='Training  Accuracy')
ax[1].plot(model.history.history['val_accuracy'],color='r',label='Validation Accuracy')

plt.legend()

######################################################################################

# Accuracy

print('Accuracy Score = ',np.max(history.history['val_accuracy']))