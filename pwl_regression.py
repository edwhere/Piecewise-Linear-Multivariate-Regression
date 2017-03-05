"""
This python program demonstrates the use of a threshold decomposition algorithm to build piecewise linear
regression models. This program uses the red-wine quality dataset for demonstration. The dataset is
available from the UCI data repository:
   https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv

Threshold decomposition is an operation that happens at the input. Once the data has been decomposed,
we can use standard linear regression procedures to compute the set of piecewise linear coefficients.

In the program we use the linear regression builder from sklearn to compute a linear regression model and two
piecewise linear models. We then compare the results.

The threshold decomposition algorithm was originally described in [1].

[1] E. Heredia, G. Arce, "Piecewise linear system modeling based on a continuous threshold decomposition",
    IEEE Trans. on Signal Processing, Vol 44, No. 6, June 1996.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
import thdecomp as th


# Define a function that calculates the Mean Squared Error of two vectors
def mse(Y1, Y2):
    return np.mean((Y1 - Y2) ** 2)

# Read wine quality data from file
df = pd.read_csv("winequality-red.csv", sep=';')

print "** Regression analysis for predicting wine quality **"
print "\nMaximum values for each of the variables in the dataset:"
print df.max(axis=0)
print "\nMinimum values for each of the variables in the dataset:"
print df.min(axis=0)

# Select a subset of the features. In this case the features with larger dynamic ranges
print "\nSelecting the six features with the highest dynamic range. We will not use the other features."
print "The selected features are: fixed acidity, residual sugar, free sulfur dioxide, pH, sulphates, and alcohol."
features = ['fixed acidity', 'residual sugar', 'free sulfur dioxide', 'pH', 'sulphates', 'alcohol']
Xdf = df[features]

print "These six features are used here to predict wine quality."
# Isolate the output values (the values that will be predicted using regression methods
Ydf = df['quality']

# Convert pandas data into numpy ndarrays
Xmat = Xdf.values
Yvec = Ydf.values

# Create a train/validate/test subsets
Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xmat, Yvec, test_size=0.33, random_state=4)

print "\nSplitting the data into train/test subsets using random sampling: "
print "size of train data matrix:", Xtrain.shape
print "size of test data matrix: ", Xtest.shape

# Fit a linear regression model to the "train" data set
linmod = LinearRegression()
linmod.fit(Xtrain, Ytrain)
YLpredtrain = linmod.predict(Xtrain)
YLpredtest = linmod.predict(Xtest)

print "\nRESULTS FOR LINEAR REGRESSION:"
mse_lin_train = mse(Ytrain, YLpredtrain)
print "   MSE for using the training dataset: ", mse_lin_train

mse_lin_test = mse(Ytest, YLpredtest)
print "   MSE using the test dataset: ", mse_lin_test


# Fit piecewise-linear regression models to the "train" data set
print "\nFor piecewise linear regression we choose the thresholds almost arbitrarily splitting initially"
print "the dynamic range into 2 or 3 segments:"
print "   'fixed acidity' has thresholds at 7 and 10"
print "   'residual sugar' has thresholds at 5 and 9"
print "   'free sulphur dioxide' has thresholds at 30 and 50"
print "   'pH' has a threshold at 3.5"
print "   'sulphates' has a threshold at 1"
print "   'alcohol' has thresholds at 10 and 12"


# pwl model 1
thresvec1 = [[7.0, 10], [5.0, 9.0], [30, 50], [3.5], [1.0], [10.0, 12.0]]
thresvec2 = [[7.0, 13.0], [6.0, 6.8, 11.0], [20, 35, 50], [3.5], [0.8], [10.0, 11.0, 12]]

# Decompose the train and test matrices
Xth_train = th.multivar_decomp(Xtrain, thresvec1)
Xth_test = th.multivar_decomp(Xtest, thresvec1)

# Apply linear regression to the decomposed matrices
pwlmod = LinearRegression()
pwlmod.fit(Xth_train, Ytrain)
YPpredtrain = pwlmod.predict(Xth_train)
YPpredtest = pwlmod.predict(Xth_test)

print "\nRESULTS FOR PIECEWISE LINEAR REGRESSION (INITIAL): "
mse_pwl_train = mse(Ytrain, YPpredtrain)
print "   MSE using the training dataset: ", mse_pwl_train

mse_pwl_test = mse(Ytest, YPpredtest)
print "   MSE using the test dataset: ", mse_pwl_test

print "\nWe moved the thresholds slightly to the left or right, and in some cases we added extra threshold. "
print "We checked if the error moves up or down. We ended up selecting the following thresholds: "
print thresvec2

# Decompose the train and test matrices
Xth_train2 = th.multivar_decomp(Xtrain, thresvec2)
Xth_test2 = th.multivar_decomp(Xtest, thresvec2)

# Apply linear regression to the decomposed matrices
pwlmod2 = LinearRegression()
pwlmod2.fit(Xth_train2, Ytrain)
YPpredtrain2 = pwlmod2.predict(Xth_train2)
YPpredtest2 = pwlmod2.predict(Xth_test2)

print "\nRESULTS FOR PIECEWISE LINEAR REGRESSION (AFTER ADJUSTMENTS): "
mse_pwl_train = mse(Ytrain, YPpredtrain2)
print "   MSE using the training dataset: ", mse_pwl_train

mse_pwl_test = mse(Ytest, YPpredtest2)
print "   MSE using the test dataset: ", mse_pwl_test


print "\nNotice that the MSE for piecewise linear models (0.417) is smaller than the MSE for linear models (0.452)."
print "PWL errors can be further reduced by searching for better locations and numbers of thresholds."
