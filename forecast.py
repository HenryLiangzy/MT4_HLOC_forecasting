import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import csv

print("Loading data")

FILENAME = "Data/CADCHF1440.csv"

exportFileName = "Data/AUDCAD30_input.csv"

opearaList = []

with open(FILENAME, 'r') as f_read:
    data = csv.reader(f_read)
    time = 0
    index = 0
    for line in data:
        temp = []
        for element in line[2:6]:
            temp.append(element)
        
        opearaList.append(temp)


print("Start proccess")
trainningList = []

#print(opearaList[0])
totalSize = len(opearaList)
print("Total size:", totalSize)

# for line in opearaList:
#     tempElement = []
#     index = opearaList.index(line)
#     # print(index)
#     if(index+9<=totalSize):
#         for shift in range(9):
#             tempElement = tempElement + opearaList[index+shift]
    
#     trainningList.append(tempElement)

inputList = []

# inputData = np.zero((totalSize, 36))


for index in range(totalSize):
    if(index+9<totalSize):
        tempElement = []
        for shift in range(9):
            for element in opearaList[index+shift]:
                tempElement.append(float(element))

        inputList.append(tempElement)


outputList = []

for line in opearaList[9:]:
    tempLine = []
    for element in line:
        tempLine.append(float(element))

    outputList.append(tempLine)

print("TrainList:", len(inputList))
print("OutputList:", len(outputList))



predictList =[]

tempPredict = []
for line in opearaList[totalSize-9:]:
    for element in line:
        tempPredict.append(float(element))

predictList.append(tempPredict)


# # translate data
# for line in inputList:
#     for element in line:
#         element = float(element)

# for line in outputList:
#     for element in line:
#         element = float(element)

# for line in predictList:
#     for element in line:
#         element = float(element)


# print(np.random.random((100, 36)))

inputData = np.array(inputList)
outputData = np.array(outputList)
predictData = np.array(predictList)

print("Start modeling")

model = tf.keras.Sequential()

# for input layer
model.add(layers.Dense(36, activation="relu"))

# for hidden layer
model.add(layers.Dense(18, activation="relu", bias_initializer=tf.keras.initializers.constant(1.0)))
# model.add(layers.Dense(18, activation="relu", bias_initializer=tf.keras.initializers.constant(1.0)))
# model.add(layers.Dense(18, activation="relu", bias_initializer=tf.keras.initializers.constant(1.0)))
# model.add(layers.Dense(18, activation="relu", bias_initializer=tf.keras.initializers.constant(1.0)))
# model.add(layers.Dense(18, activation="relu", bias_initializer=tf.keras.initializers.constant(1.0)))

# for output layer
model.add(layers.Dense(4, activation="relu", bias_initializer=tf.keras.initializers.constant(1.0)))


# compile the model
#model.compile(optimizer=keras.optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True), loss='mse', metrics=['mae'])
model.compile(optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), loss='mse', metrics=['mae'])
#model.compile(optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True), loss='categorical_crossentropy', metrics=['accuracy'])


# fit the model
# 
model.fit(inputData, outputData, epochs=10, batch_size=18)

# evaluate the result
print("Priting the evaluate result")
model.evaluate(inputData, outputData, batch_size=18)


# result predict
result = model.predict(predictData, batch_size=18)
print(result.shape)
print(result)


# Save entire model to a HDF5 file
model.save('audcad_model.h5')

# Recreate the exact same model, including weights and optimizer.
# model = tf.keras.models.load_model('my_model.h5')

# export the result
with open("result.csv", "w") as fw:
    writer = csv.writer(fw)
    writer.writerow(result)
    