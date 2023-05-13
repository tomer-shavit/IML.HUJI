import numpy as np

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        x, y = load_dataset(f"../datasets/{f}")

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def record_loss(tron: Perceptron, sample: np.ndarray, label: int):
            losses.append(tron.loss(x, y))

        tron = Perceptron(callback=record_loss)
        tron.fit(x, y)
        # Plot figure of loss as function of fitting iteration
        iterations = np.arange(1, len(losses) + 1)
        data1 = go.Scatter(x=iterations, y=losses, mode="lines")
        layout1 = go.Layout(title="Loss as function of fitting iteration", xaxis={"title": "Iterations"}
                            , yaxis={"title": "Loss"})
        fig1 = go.Figure(data=[data1], layout=layout1)
        fig1.show()
        fig1.write_image(f"./ex3_pdf/{n}.pdf")


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        x: np.ndarray
        y: np.ndarray
        x, y = load_dataset(f"../datasets/{f}")

        # Fit models and predict over training set
        lda, gnb = LDA(), GaussianNaiveBayes()
        lda_pred = lda.fit(x, y).predict(x)
        gnb_pred = gnb.fit(x, y).predict(x)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        fig = make_subplots(rows=1, cols=2, subplot_titles=[f"Gaussian Naive Bayes - accuracy {accuracy(y, gnb_pred)}",
                                                            f"LDA - accuracy {accuracy(y, lda_pred)}"],
                            vertical_spacing=.03)

        fig.add_traces([go.Scatter(x=x[:, 0], y=x[:, 1], mode='markers',
                                   marker={"color": gnb_pred, "symbol": class_symbols[y],
                                           "colorscale": class_colors(3)}),
                        go.Scatter(x=x[:, 0], y=x[:, 1], mode='markers',
                                   marker={"color": lda_pred, "symbol": class_symbols[y],
                                           "colorscale": class_colors(3)})],
                       rows=[1, 1], cols=[1, 2])

        # Add `X` dots specifying fitted Gaussians' means
        fig.add_traces([go.Scatter(x=gnb.mu_[:, 0], y=gnb.mu_[:, 1], mode="markers",
                                   marker={"symbol": 'x', "color": "black", "size": 10}),
                        go.Scatter(x=lda.mu_[:, 0], y=lda.mu_[:, 1], mode="markers",
                                   marker={"symbol": 'x', "color": "black", "size": 10})], rows=[1, 1], cols=[1, 2])

        # Add ellipses depicting the covariances of the fitted Gaussians
        for i in range(np.unique(y).shape[0]):
            fig.add_traces([get_ellipse(gnb.mu_[i], np.diag(gnb.vars_[i])),
                            get_ellipse(lda.mu_[i], lda.cov_)], rows=[1,1], cols=[1,2])

        fig.update_layout(title_text=f"Comparing Gaussian and LDA classifiers, dataset - {f[:-4]}", showlegend=False)
        # fig.show()
        fig.write_image(f"./ex3_pdf/{f[:-4]}.pdf")


if __name__ == '__main__':
    np.random.seed(0)
    # run_perceptron()
    compare_gaussian_classifiers()
