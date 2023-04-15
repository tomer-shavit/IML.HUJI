import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    temp_df = pd.read_csv(filename, parse_dates=[2])
    temp_df.dropna(inplace=True)
    temp_df = temp_df[temp_df["Temp"] > -50]
    temp_df["Year"] = temp_df["Year"].astype(str)
    temp_df["DayOfYear"] = temp_df["Date"].dt.dayofyear

    return temp_df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data(r"C:\Users\tomer\Uni\IML.HUJI\datasets\City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    il_df = df[df["Country"] == "Israel"]
    fig1 = px.scatter(il_df, x='DayOfYear', y="Temp", color="Year",
                      title="Temperature change based on day of the year.")
    fig1.update_layout(xaxis_title="Day of the year", yaxis_title="Avg temperature")
    # fig1.show()
    # fig1.write_image("Israel temp and day.pdf")
    month_group_il = il_df.groupby("Month").agg(np.std).reset_index()
    fig2 = px.bar(month_group_il, x="Month", y='Temp', title='Std of Temperature based on the Month')
    fig2.update_layout(xaxis_title="Month number", yaxis_title="Std of Temperature")
    # fig2.show()
    # fig2.write_image("std temp grouped by month.pdf")

    # Question 3 - Exploring differences between countries
    month_country_group = df.groupby(["Country", "Month"], as_index=False).agg(mean=("Temp", 'mean'),
                                                                               std=("Temp", np.std))

    fig3 = px.line(month_country_group, x="Month", y='mean', error_y='std', color="Country",
                   title="Avg temperature based on month with standard deviation")
    fig3.update_layout(xaxis_title="Month number", yaxis_title="Temperature Avg")
    # fig3.show()
    # fig3.write_image("Avg temp based on month.pdf")

    # Question 4 - Fitting model for different values of `k`
    loss = np.array([])
    train_x, train_y, test_x, test_y = split_train_test(il_df["DayOfYear"], il_df["Temp"])

    for i in range(1, 11):
        model = PolynomialFitting(i)
        model.fit(train_x, train_y)
        loss = np.append(loss, round(model.loss(train_x, train_y), 2))

    print(list(loss))
    fig4 = px.bar(x=list(range(1,11)), y=loss, title="Loss as a factor of polynom degree.")
    fig4.update_layout(xaxis_title="Polynom degree", yaxis_title="Error")
    # fig4.show()
    # fig4.write_image("Loss and p.pdf")

    # Question 5 - Evaluating fitted model on different countries
    loss_il = np.array([])
    model_il = PolynomialFitting(5)
    model_il.fit(il_df["DayOfYear"], il_df["Temp"])
    countries = df["Country"].unique()
    countries = countries[countries != "Israel"]
    for country in countries:
        country_filter = df[df["Country"] == country]
        loss_il = np.append(loss_il, model_il.loss(country_filter["DayOfYear"], country_filter["Temp"]))

    fig5 = px.bar(x=countries, y=loss_il, title="Error per Country", color=countries)
    fig5.update_layout(xaxis_title="Country", yaxis_title="Error")
    # fig5.show()
    # fig5.write_image("Error per Country.pdf")
