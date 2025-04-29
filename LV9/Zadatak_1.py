import numpy as np
from tensorflow import keras
from keras import layers, models
from keras.datasets import cifar10
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from tensorflow.python.keras.callbacks import TensorBoard

# ucitaj CIFAR-10 podatkovni skup
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# prikazi 9 slika iz skupa za ucenje
plt.figure()
for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.xticks([]),plt.yticks([])
    plt.imshow(X_train[i])

plt.show()


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
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(500, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

# definiraj listu s funkcijama povratnog poziva
my_callbacks = [
    keras.callbacks.TensorBoard(log_dir = 'logs/cnn',
                                update_freq = 100)
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
#imamo 1 122 758 parametara
#cnn mreza se sastoji od 3 konvulucijska sloja, 3 sloja sazimanja i 2 skrivena sloja od 500 i 10 neurona
#zad3)
#tocnost na testnom skupu podataka=72.95

#za podatke na train skupu
#tocnost je naglo rasla do 0.95 nakon cega je postepeno rasla, te je najveca tocnost iznosila 0.9912 
#funkcija gubitka se snazno smanjivala do 0.1 te je nakon toga postepeno padala, najniza vrijednost je bila 0.026

#za podatke na validaciji
#tocnost na pocetku naglo raste do te dolazi do tocnosto od 0.75 a nakon toga postepeno pada, te je na kraju iznosila 0.744
#gubitak naglo pada te dolazi do 0.8 , a nakon toga strmoglao raste , te je na kraju iznosilo 2.36

#python -m tensorboard.main --logdir=logs