import numpy as np
from tensorflow import keras
from keras import layers, models
from keras.datasets import cifar10
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from tensorflow.python.keras.callbacks import TensorBoard

# ucitaj CIFAR-10 podatkovni skup
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

#X_train=X_train[:25000]
#y_train=y_train[:25000]
# pripremi podatke (skaliraj ih na raspon [0,1]])
X_train_n = X_train.astype('float32')/ 255.0
X_test_n = X_test.astype('float32')/ 255.0

# 1-od-K kodiranje
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# CNN mreza
model = keras.Sequential()
model.add(layers.Input(shape=(32,32,3)))
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
#model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
#model.add(layers.MaxPooling2D(pool_size=(2, 2)))
#model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
#model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(500, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

# definiraj listu s funkcijama povratnog poziva
my_callbacks = [
    keras.callbacks.TensorBoard(log_dir = 'logs/cnn_es_bez_slojeva',
                                update_freq = 100),

    keras.callbacks.EarlyStopping ( monitor = "val_loss" ,
                                        patience = 12 ,
                                        verbose = 1 )
]

model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

model.fit(X_train_n,
            y_train,
            epochs = 40,
            batch_size = 64,
            callbacks = my_callbacks,
            validation_split = 0.1)


score = model.evaluate(X_test_n, y_test, verbose=0)
print(f'Tocnost na testnom skupu podataka: {100.0*score[1]:.2f}')

#zad1)
#ako je batch mali(5) svaka epoka dulje traje (50-60s),te se puno sporije dolazi do krajnjih rezultata te se nece doci do earlystoppinga
#Epoch 21: early stopping
#Tocnost na testnom skupu podataka: 69.08
#accuracy: 0.8380 - loss: 0.4814 - val_accuracy: 0.7140 - val_loss: 1.1557
#ako je batch velik(150) svaka epoha traje krace (9s), te se puno brze dolazi do krajnjih rezultata, te je tocnost puno veca
#Epoch 22: early stopping
#Tocnost na testnom skupu podataka: 74.77


#zad2)
#oroginalin optimizator adam

#adagrad optimizator
#Tocnost na testnom skupu podataka: 53.50
#ovaj optimizator je loš jer pre brzo smanji stopu učenja, takoder procjecno vrijeme epohe sa 11s na doslo na 23s

#adamW optimizator
#Epoch 19: early stopping
#Tocnost na testnom skupu podataka: 74.32
#prosjecno vrijeme epohe 23s 
#bolji je od adagrada, slica adamu

#zad3)
#maknio sam 2 konvolucijska sloja i 2 maxpooling sloja
#prosjecno vrijeme epohe 14s
#Epoch 19: early stopping
#Tocnost na testnom skupu podataka: 66.98
#tocnost se smanjuje sa gubitkom slojeva sa 0.74 na 0.67
#te je veci gubitak sa 0.78 na 1.1

#zad4)
#ako uzmemo 50% podataka za treniranje 
#prosjecno vrijeme po epohi 12s
#Tocnost na testnom skupu podataka:69.55
#Epoch 19: early stopping
#manja tocnost nego sa vise podataka, te veci gubitci