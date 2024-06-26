import ft_linear_regression as ft_lr

def predict(regressor: ft_lr.LinearRegression):
    number = input("Enter mileage: ")
    try:
        number = int(number)
    except ValueError:
        print("Invalid input")
        exit()

    price = ft_lr.linearFunction(number, regressor.theta0, regressor.theta1)
    print("Predicted price: ", price)

if __name__ == "__main__":
    
    theta0 = 0
    theta1 = 0
    X, Y = ft_lr.parseCsv("theta.csv")
    if (len(X) == 0):
        print("No data to process")
    else:
        theta0 = X[0]
        theta1 = Y[0]
    regressor = ft_lr.LinearRegression((),(),theta0, theta1)
    predict(regressor)
