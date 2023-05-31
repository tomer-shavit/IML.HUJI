from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10


    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    train_x, train_y, test_x, test_y = X[:n_samples], y[:n_samples], X[n_samples:], y[n_samples:]
    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    ridge_lamb, lasso_lamb = np.linspace(0.00001, 0.01, n_evaluations), np.linspace(0.01, 2, n_evaluations)
    ridge_scores, lasso_scores = np.zeros((n_evaluations, 2)), np.zeros((n_evaluations, 2))

    for i in range(n_evaluations):
        ridge_scores[i] = cross_validate(RidgeRegression(ridge_lamb[i]), train_x, train_y, mean_square_error)
        lasso_scores[i] = cross_validate(Lasso(lasso_lamb[i], max_iter=5000), train_x, train_y, mean_square_error)

    fig = make_subplots(1, 2, subplot_titles=["Ridge Regression", "Lasso Regression"])
    fig.add_traces([go.Scatter(x=ridge_lamb, y=ridge_scores[:, 0], name="Ridge Train Error"),
                    go.Scatter(x=ridge_lamb, y=ridge_scores[:, 1], name="Ridge Validation Error"),
                    go.Scatter(x=lasso_lamb, y=lasso_scores[:, 0], name="Lasso Train Error"),
                    go.Scatter(x=lasso_lamb, y=lasso_scores[:, 1], name="Lasso Validation Error")],
                    rows=[1, 1, 1, 1],
                    cols=[1, 1, 2, 2])
    fig.write_image(f"./ex4_pdf/lasso_ridge.pdf")


    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model

    best_ridge = ridge_lamb[np.argmin(ridge_scores[:,1])]
    best_lasso = lasso_lamb[np.argmin(lasso_scores[:,1])]
    print(f"Best ridge lambda {best_ridge}")
    print(f"Best lasso lambda {best_lasso}")
    print(f"Least-Squares: {LinearRegression().fit(train_x, train_y).loss(test_x, test_y)}")
    print(f"Ridge: {RidgeRegression(lam=best_ridge).fit(train_x, train_y).loss(test_x, test_y)}")
    print(f"Lasso: {mean_square_error(test_y, Lasso(alpha=best_lasso).fit(train_x, train_y).predict(test_x))}")


if __name__ == '__main__':
    np.random.seed(0)
    select_regularization_parameter()

