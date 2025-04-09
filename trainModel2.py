import os  
import numpy as np 
import cv2 
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

is_init = False
dictionary = {}
c = 0

for i in os.listdir():
    if i.endswith(".npy") and not i.startswith("labels"):  
        class_name = i.split('.')[0]  
        
        if class_name not in dictionary:
            dictionary[class_name] = c
            c += 1

        data = np.load(i)
        num_samples = data.shape[0]  
        labels = np.full((num_samples, 1), dictionary[class_name])

        if not is_init:
            is_init = True
            X = data
            y = labels
        else:
            X = np.concatenate((X, data))
            y = np.concatenate((y, labels))

label = list(dictionary.keys())  
num_classes = len(np.unique(y))  
y = to_categorical(y, num_classes=num_classes)

# unique, counts = np.unique(np.argmax(y, axis=1), return_counts=True)
# print(dict(zip(unique, counts)))
# exit()

'''
When some dance forms have significantly more samples than others, the model may become biased.
Here the output was:
{0: 6577, 1: 7975, 2: 14138}

Thus using class weights
'''

class_weights = compute_class_weight('balanced', classes=np.unique(np.argmax(y, axis=1)), y=np.argmax(y, axis=1))
class_weights = dict(enumerate(class_weights))


# Normalize features for better training stability
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Split dataset into training (70%), validation (15%), and testing (15%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Shuffle dataset
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X, y = X[indices], y[indices]

ip = Input(shape=(X.shape[1],))
m = Dense(128, activation="relu")(ip)
m = Dropout(0.3)(m)
m = Dense(64, activation="relu")(m)
m = Dropout(0.3)(m)
op = Dense(num_classes, activation="softmax")(m)

model = Model(inputs=ip, outputs=op)
model.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=['accuracy'])

# Use early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

# model.fit(X, y, epochs=100, batch_size=32, callbacks=[early_stopping])
'''
Updated the model.fit with class weights because the classes count were imbalanced and 
even split the data into train, validation and split
'''
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val),
          callbacks=[early_stopping], class_weight=class_weights)


model.save("model.h5")
np.save("labels.npy", np.array(label))
