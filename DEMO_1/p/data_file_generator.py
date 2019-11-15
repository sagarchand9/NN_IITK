from numpy import *
import csv

fields = []
data = []

try:
    with open("Vx_Vy_sonar_reading_March1_5000.csv", 'r') as rawData:
        reader = csv.reader(rawData)
        fields = reader.next()

        for row in reader:
            data.append(row)

        n = reader.line_num - 1
except IOError as e:
    print "raw data file absent in current directory"

print n
m = len(fields)

cleanData = empty((n, m + 2))
cleanData[0][m] = 0.0
cleanData[0][m + 1] = 0.0
for i in range(m):
    cleanData[0][i] = data[0][i]

for i in range(1, n):
    cleanData[i][m] = cleanData[i-1][0]
    cleanData[i][m + 1] = cleanData[i-1][1]
    for j in range(m):
        cleanData[i][j] = data[i][j]

savetxt("data3.csv", cleanData, delimiter = ',')
