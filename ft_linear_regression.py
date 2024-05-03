import csv
import matplotlib.pyplot as plt

"""
Parse a CSV file containing 2 columns, remove the header and store the
content of the first and second columns in xValues and yValues respectively.

Parameter:
- file: filename

Returns:
tuples xValues and yValues
"""

def parseCsv (file: str):
    xValues, yValues = [], []
    try:
        with open(file, 'r') as csv_file:
            reader = csv.reader(csv_file)
            next(reader)
            xValues, yValues = zip(*reader)
    except FileNotFoundError:
        print("File not found: ", file)
    except csv.Error as e:
        print(f"Error reading CSV file: {e}")
    ###
    print(xValues, yValues)
    plt.scatter(xValues, yValues)
    plt.show()
    ###

    return (xValues, yValues)


"""
Mean squared error function used as the loss function

Parameters:
- size: size of the CSV set
- predictedValues: list of 
- realValues:
"""

def meanSquaredError (size: int, predictedValues: list, realValues: tuple):
    mean = 0
    realY = realValues[1]
    predictY = predictedValues[1]
    for i in (size - 1):
        mean += (realY[i] - predictY[i]) ** 2
    mean *= 1 / size

    return (mean)

def meanSquareError_derivativeSlope(size: int, predictedValues: list, realValues: tuple):
    mean = 0
    realY = realValues[1]
    predictX = predictedValues[0]
    predictY = predictedValues[1]
    for i in (size - 1):
        mean += predictX[i] * (realY[i] - predictY[i])
    mean *= -2 / size

    return (mean)

def meanSquareError_derivativeIntercept(size: int, predictedValues: list, realValues: tuple):
    mean = 0
    realY = realValues[1]
    predictY = predictedValues[1]
    for i in (size - 1):
        mean += (realY[i] - predictY[i])
    mean *= -2 / size

    return (mean)

if __name__ == "__main__":

    epochs = 1000
    learning_rate = 0.0001

    predictedY = []
    realX, realY = parseCsv('data.csv')
    realValues = (realX, realY)
    theta0, theta1 = 0, 0   # f(x) = theta1 * x + theta0

    # for i in range(epochs):
    #     predictY = theta1 * realX
