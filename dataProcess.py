import numpy as np
import csv

print("Loading data")

FILENAME = "Data/CADCHF1440.csv"

exportFileName = "Data/CADCHF1440_input.csv"

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

print(inputData)