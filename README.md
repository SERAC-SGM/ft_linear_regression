# FT_LINEAR_REGRESSION

Basic machine learning program using a linear function train with a gradient descent algorithm.

The linear function is defined as follows:

    f(x) = θ0 + θ1 * x

## Usage
### Training program
    ./ft_linear_regression.py [epochs] [learning rate]
    OR
    python3 ft_linear_regression.py [epochs] [learning rate]

If epochs and learning rate arguments are unspecified, they will be set to default values (epochs = 1000 and learning rate = 0.1).

The dataset must be included in the root directory with the name set to 'data.csv'.

Once the training is done, the values of θ0, θ1 and the mean squarred error will be prompted. A new file 'theta.csv' will be created in the root directory, containing θ0 and θ1.
