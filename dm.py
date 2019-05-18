import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers
import csv

FILENAME = "Data/Train_data.csv"

print("Start loading data")

operateList = []
with open(FILENAME) as f:
    reader = csv.reader(f)
    for line in reader:
        tempLine = []
        for element in line:
            tempLine.append(float(element))
        operateList.append(tempLine)

inputList = []
for line in operateList:
    tempLine = []
    for element in line[:8]:
        tempLine.append(element)
    inputList.append(tempLine)

outputList = []
for line in operateList:
    tempLine = []
    tempLine.append(line[9])
    outputList.append(tempLine)

inputData = np.array(inputList)
outputData = np.array(outputList)

print("Start modeling the NN")

model = tf.keras.Sequential()

# for input layer
model.add(layers.Dense(9, activation="relu"))

# for hidden layer
model.add(layers.Dense(18, activation="relu", bias_initializer=tf.keras.initializers.constant(1.0)))
model.add(layers.Dense(18, activation="relu", bias_initializer=tf.keras.initializers.constant(1.0)))
# model.add(layers.Dense(18, activation="relu", bias_initializer=tf.keras.initializers.constant(1.0)))
# model.add(layers.Dense(18, activation="relu", bias_initializer=tf.keras.initializers.constant(1.0)))
# model.add(layers.Dense(18, activation="relu", bias_initializer=tf.keras.initializers.constant(1.0)))

# for output layer
model.add(layers.Dense(1, activation="relu", bias_initializer=tf.keras.initializers.constant(1.0)))


# compile the model
#model.compile(optimizer=keras.optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True), loss='mse', metrics=['mae'])
#model.compile(optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), loss='mse', metrics=['mae'])
#model.compile(optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True), loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), loss='mse',metrics=['mae'])


# fit the model
# 
model.fit(inputData, outputData, epochs=10, batch_size=3)

# evaluate the result
print("Priting the evaluate result")
model.evaluate(inputData, outputData, batch_size=3)
