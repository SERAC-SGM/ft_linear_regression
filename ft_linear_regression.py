import csv
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.theta0 = 0
        self.theta1 = 0
    
    def predict(self, learningRate):
        predictY = 

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

    xValues = [int(x) for x in xValues]
    yValues = [int(y) for y in yValues]

    ###
    # print(xValues, yValues)
    # plt.scatter(xValues, yValues)
    # plt.show()
    ###

    return (xValues, yValues)


"""
Mean squared error function used as the loss function

Parameters:
- size: size of the CSV set
- predictedValues: list of 
- realValues:
"""

def meanSquaredError (size: int, predictY: list, realValues: tuple):
    mean = 0
    realY = realValues[1]
    for i in range(size - 1):
        mean += (realY[i] - predictY[i]) ** 2
    mean *= 1 / size

    return (mean)

def meanSquareError_derivativeSlope(size: int, predictY: list, realValues: tuple):
    mean = 0
    realX = realValues[0]
    realY = realValues[1]
    for i in range(size - 1):
        mean += realX[i] * (realY[i] - predictY[i])
    mean *= -2 / size

    return (mean)

def meanSquareError_derivativeIntercept(size: int, predictY: list, realValues: tuple):
    mean = 0
    realY = realValues[1]
    for i in range(size - 1):
        mean += (realY[i] - predictY[i])
    mean *= -2 / size

    return (mean)

if __name__ == "__main__":

    epochs = 1000
    learningRate = 0.0001

    predictY = []
    realX, realY = parseCsv('data.csv')
    size = len(realX)
    realValues = (realX, realY)
    theta0, theta1 = 0, 0   # f(x) = theta1 * x + theta0

    for i in range(epochs):
        predictY = [theta1 * x + theta0 for x in realX]
        print("theta0: ", theta0)
        print("theta1: ", theta1)
        theta0 = theta0 - learningRate * meanSquareError_derivativeIntercept(size, predictY, realValues)
        theta1 = theta1 - learningRate * meanSquareError_derivativeSlope(size, predictY, realValues)
        print("mse: ", meanSquaredError(size, predictY, realValues))
