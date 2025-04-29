import numpy as np
from tensorflow import keras
from keras import layers, models
from keras.datasets import cifar10
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from tensorflow.python.keras.callbacks import TensorBoard

# ucitaj CIFAR-10 podatkovni skup
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

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
model.add(layers.Dropout(0.3))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

# definiraj listu s funkcijama povratnog poziva
my_callbacks = [
    keras.callbacks.TensorBoard(log_dir = 'logs/cnn_dropout',
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

#Tocnost na testnom skupu podataka: 73.09

# tocnost se malo poboljsala kada smo dodali dropout spoj
#na trening uskupu tocnost naglo raste do 0.93, te nastavlja rasti do 0.977 sto je malo manje nego bez dropout spoja
# na validacijskom skupu tocnost raste do 0.76, te krece padati do 0.75, a tocnost sa dropout slojem je vece nego bez njega

#gubitak na trening skupu je naglo pada do 0.19, a najniza vrijednost je 0.077, sa dropout slojem gubitak je veci
#gubitak na testnom skupu naglo pada do 0.772, te krece nakon toga rasti te je na kraju 1.77, gubitak sa dropout slojem je manja nego bez njega