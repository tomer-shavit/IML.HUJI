from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"

def remove_nan(X: pd.DataFrame, y: Optional[pd.Series] = None):
    nan_rows_X = X.isnull().any(axis=1)
    nan_rows_y = None
    if y is not None:
        nan_rows_y = y.isnull()

    nan_rows = nan_rows_X
    if nan_rows_y is not None:
        nan_rows = nan_rows_X | nan_rows_y
    X = X[~nan_rows]
    if y is not None:
        y = y[~nan_rows]

    return X, y


def preprocess_data(X: pd.DataFrame, y: Optional[pd.Series] = None):
    """
    preprocess data
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem
    y : array-like of shape (n_samples, )
        Response vector corresponding given samples
    Returns
    -------
    Post-processed design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    X.drop(['id', 'date', 'sqft_living15', 'sqft_lot15', 'long', 'lat'], axis=1, inplace=True)
    X, y = remove_nan(X, y)

    X = pd.get_dummies(X, prefix="zipcode", columns=['zipcode'])

    no_price = X[X['price'] <= 0]
    bad_grade = X[(X['grade'] < 1) | (X['grade'] > 13)]
    bad_condition = X[(X['condition'] < 1) | (X['condition'] > 5)]
    bad_view = X[(X['view'] < 0) | (X['view'] > 4)]
    no_price_y = None
    if y is not None:
        no_price_y = y[y <= 0]

    mask_index = pd.concat([no_price, bad_grade, bad_condition, bad_view, no_price_y])

    X.drop(mask_index.index, inplace=True)
    y.drop(mask_index.index, inplace=True)

    x = X.drop('price', axis=1)

    return x, y


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    pearson_corr = np.array([])
    feature_title = np.array([])
    y_std = y.std()

    for col in X.columns:
        col_std = X[col].std()
        if col_std != 0:
            cor = (X[col].cov(y)) / (X[col].std() * y_std)
            pearson_corr = np.append(pearson_corr, cor)
            feature_title = np.append(feature_title, col)

    # Find highest and lowest pearson correlations
    data1 = go.Scatter(x=feature_title, y=pearson_corr, mode="markers")
    layout1 = go.Layout(title="Pearson Correlation of Features and the Price", xaxis={"title": "Feature Name"}
                        , yaxis={"title": "Pearson Correlation"})
    fig = go.Figure(data=[data1], layout=layout1)
    fig.show()

    # Highest is sqft_living with correlation of 0.702
    data2 = go.Scatter(x=X['sqft_living'], y=y, mode="markers")
    layout2 = go.Layout(title="Correlation between Living room size and the price",
                        xaxis={"title": "Living room size (sqft)"}, yaxis={"title": "House Price (USD)"})
    fig = go.Figure(data=[data2], layout=layout2)
    fig.show()
    pio.write_image(fig, "ex2_pdf/sqft_living and price.pdf")
    # Lowest is condition with correlation of 0.0362

    data3 = go.Scatter(x=X['condition'], y=y, mode="markers")
    layout3 = go.Layout(title="Correlation between House condition and the price",
                        xaxis={"title": "House condition (1-5)"}, yaxis={"title": "House Price (USD)"})
    fig = go.Figure(data=[data3], layout=layout3)
    fig.show()
    pio.write_image(fig, "ex2_pdf/condition and price.pdf")


if __name__ == '__main__':
    np.random.seed(0)
    df = pd.read_csv("../datasets/house_prices.csv")

    # Question 1 - split data into train and test sets
    labels = df["price"]
    train_x, train_y, test_x, test_y = split_train_test(df, labels)

    # Question 2 - Preprocessing of housing prices dataset
    train_x, train_y = preprocess_data(train_x, train_y)

    # Question 3 - Feature evaluation with respect to response
    feature_evaluation(train_x, train_y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    linear_reg = LinearRegression()
    test_x, test_y = preprocess_data(test_x, test_y)
    test_x, train_x = test_x.align(train_x, axis=1, fill_value=0)
    test_x = test_x[train_x.columns]
    mean_loss, mean_std, losses_10 = np.array([]), np.array([]), np.array([])
    for i in range(10, 101, 1):
        for j in range(10):
            sample = train_x.sample(frac=(i / 100))
            sample_labels = train_y.loc[sample.index]
            loss = linear_reg.fit(sample, sample_labels).loss(test_x, test_y.to_numpy())
            losses_10 = np.append(losses_10, loss)

        mean_loss = np.append(mean_loss, np.mean(losses_10))
        mean_std = np.append(mean_std, np.std(losses_10))
        losses_10 = np.array([])

    p_values = np.arange(10, 101, 1)

    scatter = [go.Scatter(x=p_values, y=mean_loss, mode="markers+lines", name="Mean Loss"),
               go.Scatter(x=p_values, y=(mean_loss + 2 * mean_std), mode="lines", fill="tonexty",
                          line=dict(color="lightgrey"), showlegend=False),
               go.Scatter(x=p_values, y=(mean_loss - 2 * mean_std), mode="lines", fill="tonexty",
                          line=dict(color="lightgrey"), showlegend=False)]

    layout = go.Layout(title="Mean loss as a function of %p", xaxis=dict(title="p%"),
                       yaxis=dict(title="Mean Loss"))

    fig = go.Figure(data=scatter, layout=layout)
    fig.show()
    pio.write_image(fig, "ex2_pdf/Mean loss p%.pdf")
