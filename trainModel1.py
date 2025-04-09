import os  
import numpy as np 
import cv2 
from tensorflow.keras.utils import to_categorical

from keras.layers import Input, Dense 
from keras.models import Model
 
is_init = False
size = -1

label = []
dictionary = {}
c = 0

# datacount = 0
# for i in os.listdir():
# 	if i.endswith(".npy") and not i.startswith("labels"):
# 		data = np.load(i)
# 		print(f"File: {i}, Shape: {data.shape}")  # Check individual .npy files
# 		datacount += data.shape[0]
# print(datacount)
# exit()

'''
Output:
File: Bharatanatyam.npy, Shape: (6577, 66)
File: Kathak.npy, Shape: (7975, 66)
File: Mohiniyatam.npy, Shape: (14138, 66)
28690

Thus, this ensure that the number of labels matches the number of features.
'''
# print("All .npy files:", os.listdir())
# exit() 

is_init = False
dictionary = {}
c = 0

for i in os.listdir():
    if i.endswith(".npy") and not i.startswith("labels"):  
        class_name = i.split('.')[0]  # Extract class name

        # Ensure the class is assigned a numeric label
        if class_name not in dictionary:
            dictionary[class_name] = c
            c += 1

        data = np.load(i)
        num_samples = data.shape[0]  # Number of feature vectors

        # Assign the corresponding numeric label to each feature row
        labels = np.full((num_samples, 1), dictionary[class_name])
        # print(labels)

        # Properly initialize or concatenate `X` and `y`
        if not is_init:
            is_init = True
            X = data
            y = labels
        else:
            X = np.concatenate((X, data))
            y = np.concatenate((y, labels))



label = list(dictionary.keys())  # Store class labels



# Assume y contains class labels as integers (e.g., 0, 1, 2, ...)
num_classes = len(np.unique(y))  # Get the number of unique classes
y = to_categorical(y, num_classes=num_classes)


# for i in range(y.shape[0]):
# 	y[i, 0] = dictionary.get(str(y[i, 0]), -1)  # Convert to string and handle missing keys
# y = np.array(y, dtype="int32")

# for i in range(y.shape[0]):
#     class_name = str(y[i, 0])  # Ensure class_name is a string

#     if class_name in dictionary:
#         y[i, 0] = dictionary[class_name]  # Assign correct numeric label
#     else:
#         print(f"Warning: Class '{class_name}' not found in dictionary! Setting to -1")
#         y[i, 0] = -1  # Debugging step

'''
The above code wasn't required 
as, the numerical categories where assigned to each row of features simultaneously as they were extracted
'''
# print("X shape:", X.shape)
# print("y shape:", y.shape)
# print(y)
# exit()


X_new = X.copy()
y_new = y.copy()
counter = 0 

cnt = np.arange(X.shape[0])
np.random.shuffle(cnt)

for i in cnt: 
	X_new[counter] = X[i]
	y_new[counter] = y[i]
	counter = counter + 1


ip = Input(shape=(X.shape[1],))

'''
ValueError: Cannot convert '66' to a shape.
Before: ip = Input(shape=(X.shape[1]))
After: ip = Input(shape=(X.shape[1],))
Issue: In TensorFlow/Keras, when specifying the shape for an Input layer, you need to provide it as a tuple. 
If the input is a single-dimensional feature vector (e.g., 66 features), it should be (66,), not just 66.
'''

m = Dense(128, activation="tanh")(ip)
m = Dense(64, activation="tanh")(m)

# op = Dense(y.shape[1], activation="softmax")(m) 
op = Dense(num_classes, activation="softmax")(m) 


model = Model(inputs=ip, outputs=op)

model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc'])

model.fit(X_new, y_new, epochs=80)


model.save("model.h5")
np.save("labels.npy", np.array(label))