import csv

filename = "Data/AUDCAD30.csv"

train = []
output = []


with open(filename) as f:
    data = csv.reader(f)
    time = 0;
    tempList = []
    for line in data:
        print(data.line_num, line)