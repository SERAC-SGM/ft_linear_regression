import csv
import sys
import numpy as np
import matplotlib.pyplot as plt

class  DataMismatchError(Exception):
    """Exception raised when CSV data columns have unequal lengths"""
    pass


class LinearRegression:
    """
    Class containing the data and the parameters of the linear regression model
    
    Attributes:
    - X: tuple of x values
    - Y: tuple of y values
    - theta0: intercept
    - theta1: slope
    - size: size of the dataset
    """
    def __init__(self, X: tuple, Y: tuple, theta0: float = 0.0, theta1: float = 0.0):
        self.X = X
        self.Y = Y
        self.theta0 = theta0
        self.theta1 = theta1
        self.size = len(X)


def nomalizeData(data: tuple):
    return [(x - np.mean(data)) / np.std(data) for x in data]


def parseCsv (file: str):
    """
    Parse a CSV file containing two columns, remove the header and store the
    content of the first and second columns in xValues and yValues respectively.

    Parameter:
    - file: filename

    Returns:
    - tuples xValues and yValues
    """
    xValues, yValues = (), ()
    try:
        with open(file, 'r') as csv_file:
            reader = csv.reader(csv_file)
            next(reader)
            xValues, yValues = zip(*reader)
            if len(xValues) != len(yValues):
                raise DataMismatchError("Unequal number of values in X and Y columns")
    except FileNotFoundError:
        print("File not found: ", file)
        exit()
    except csv.Error as e:
        print(f"Error reading CSV file: {e}")
        exit()

    xValues = [float(x) for x in xValues]
    yValues = [float(y) for y in yValues]

    return (xValues, yValues)


def linearFunction(x: int, theta0: float, theta1: float):
    """
    Linear function calculation.

    Returns:
    - y = theta1 * x + theta0
    """
    return (theta1 * x + theta0)


def calculateValues(regressor: LinearRegression):
    """
    Apply the linear function to the dataset and return the predicted values.

    Returns:
    - list of predicted values
    """
    yPredict = [linearFunction(x, regressor.theta0, regressor.theta1) for x in regressor.X]

    return yPredict


def meanSquaredError (regressor: LinearRegression, predictY: list):
    """
    Mean squared error function used as the loss function :
    E(theta0, theta1) = 1/m * sum(i = 0 to m - 1) (theta1 * xi + theta0 - yi) ** 2

    Parameters:
    - regressor: LinearRegression object
    - predictY: list of predicted values

    Returns:
    - mean squared error
    """
    mean = 0
    for i in range(regressor.size):
        mean += (regressor.Y[i] - predictY[i]) ** 2
    mean /= regressor.size

    return (mean)


def meanSquareError_derivativeTheta0(regressor: LinearRegression, predictY: list):
    """
    Derivative of the mean squared error function with respect to theta0:
    dE/dtheta0 = 1/m * sum(i = 0 to m - 1) (theta1 * xi + theta0 - yi)

    Parameters:
    - regressor: LinearRegression object
    - predictY: list of predicted values

    Returns:
    - derivative of the mean squared error with respect to theta0
    """
    mean = 0
    for i in range(regressor.size):
        mean += (regressor.Y[i] - predictY[i])
    mean /= regressor.size

    return (mean)


def meanSquareError_derivativeTheta1(regressor: LinearRegression, predictY: list):
    """
    Derivative of the mean squared error function with respect to theta1:
    dE/dtheta1 = 1/m * sum(i = 0 to m - 1) xi * (theta1 * xi + theta0 - yi)

    Parameters:
    - regressor: LinearRegression object
    - predictY: list of predicted values

    Returns:
    - derivative of the mean squared error with respect to theta1
    """
    mean = 0
    for i in range(regressor.size):
        mean += regressor.X[i] * (regressor.Y[i] - predictY[i])
    mean /= regressor.size

    return (mean)


def gradientDescent(regressor: LinearRegression, learningRate: float):
    """
    Gradient descent algorithm to minimize the mean squared error function.

    Parameters:
    - regressor: LinearRegression object
    - learningRate: learning rate

    Returns:
    - mean squared error
    """
    predictY = calculateValues(regressor)
    regressor.theta0 += learningRate * meanSquareError_derivativeTheta0(regressor, predictY)
    regressor.theta1 += learningRate * meanSquareError_derivativeTheta1(regressor, predictY)

    return meanSquaredError(regressor, predictY)


def printData(theta0: float, theta1:float, mse: list):
    print("\n=========== Values ===========")
    print(f"theta0: {theta0}\ntheta1: {theta1}\n")
    print(f"Mean squared error: {mse[-1]}")
    print("==============================")
    return


def plotData(theta0: float, theta1: float, X: list, Y: list):
    x_values = [min(X), max(X)]
    y_values = [theta1 * x + theta0 for x in x_values]
    plt.scatter(X, Y)
    plt.title('Linear Regression')
    plt.plot(x_values, y_values, label=f'y = {theta1}x + {theta0}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()

    return

def denormalizeCoeff(regressor: LinearRegression, X: list, Y: list):
    theta1_denorm = regressor.theta1 * np.std(Y) / np.std(X)
    theta0_denorm = np.mean(Y) - theta1_denorm * np.mean(X)

    return (theta0_denorm, theta1_denorm)

if __name__ == "__main__":

    if (len(sys.argv) > 3):
        epochs = int(sys.argv[1])
        learningRate = float(sys.argv[2])
    else:
        epochs = 1000
        learningRate = 0.1

    X, Y = parseCsv('data.csv')
    Xn = nomalizeData(X)
    Yn = nomalizeData(Y)
    regressor = LinearRegression(Xn, Yn)
    mse = []

    for i in range(epochs):
        mse.append(gradientDescent(regressor, learningRate))

    theta0, theta1 = denormalizeCoeff(regressor, X, Y)
    printData(theta0, theta1, mse)
    plotData(theta0, theta1, X, Y)

    with open('theta.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['theta0', 'theta1'])
        writer.writerow([theta0, theta1])
