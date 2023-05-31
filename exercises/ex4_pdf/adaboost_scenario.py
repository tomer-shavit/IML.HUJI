import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    ada_b = AdaBoost(DecisionStump, n_learners).fit(train_X, train_y)
    loss_train = np.zeros(n_learners)
    loss_test = np.zeros(n_learners)
    for i in range(n_learners):
        loss_test[i] = ada_b.partial_loss(test_X, test_y, i + 1)
        loss_train[i] = ada_b.partial_loss(train_X, train_y, i + 1)

    scatter1 = go.Scatter(x=np.arange(1, n_learners + 1), y=loss_train, mode="lines", name="train")
    scatter2 = go.Scatter(x=np.arange(1, n_learners + 1), y=loss_test, mode="lines", name="test")

    layout1 = go.Layout(title="Partial loss on committee members", xaxis_title="Members in the committee",
                        yaxis_title="Loss")
    fig1 = go.Figure(data=[scatter1, scatter2], layout=layout1)
    fig1.write_image(f"./ex4_pdf/partial_loss_{noise}_noise.pdf")

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])

    fig2 = make_subplots(1, 4, subplot_titles=[f"{t} Members" for t in T])

    for i in range(len(T)):
        fig2.add_traces([decision_surface(lambda dataset: ada_b.partial_predict(dataset, T[i]), lims[0], lims[1],
                                          showscale=False),
                         go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                    marker=dict(color=test_y,
                                                symbol=np.where(test_y == 1, "circle", "x")))], rows=1, cols=i + 1, )
    fig2.write_image(f"./ex4_pdf/desision_surface_{noise}_noise.pdf")

    # Question 3: Decision surface of best performing ensemble
    best_i = np.argmin(loss_test)
    fig3 = go.Figure([decision_surface(lambda dataset: ada_b.partial_predict(dataset, best_i),
                                       lims[0], lims[1], showscale=False),
                      go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                 marker=dict(color=test_y, symbol=np.where(test_y == 1, "circle", "x")))],
                     layout=go.Layout(
                         title=f"Best committee, {best_i + 1} members, Accuracy: {1 - round(loss_test[best_i], 2)}",
                         showlegend=False))
    fig3.write_image(f"./ex4_pdf/best_committee_{noise}_noise.pdf")

    # Question 4: Decision surface with weighted samples
    dist = 20 * ada_b.D_ / np.max(ada_b.D_)
    fig4 = go.Figure([decision_surface(ada_b.predict, lims[0], lims[1], showscale=False),
                      go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                                 marker=dict(size=dist, color=train_y, symbol=np.where(train_y == 1, "circle", "x")))],
                     layout=go.Layout(
                         title=f"Adaboost with weighted samples",
                         showlegend=False))
    fig4.write_image(f"./ex4_pdf/ada_weighted_{noise}_noise.pdf")


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
