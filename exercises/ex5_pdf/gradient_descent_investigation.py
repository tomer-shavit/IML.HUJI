import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test
from sklearn.metrics import roc_curve, auc
import plotly.graph_objects as go
from IMLearn.metrics import misclassification_error
from IMLearn.model_selection import cross_validate


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines",
                                 marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values = []
    weights = []

    def callback(model, **kwargs):
        values.append(kwargs['val'])
        weights.append(kwargs['weights'])
        return

    return callback, values, weights


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    modules = [("L1", L1), ("L2", L2)]
    for f_name, f_model in modules:
        for eta in etas:
            f = f_model(init)
            callback, values, weights = get_gd_state_recorder_callback()
            gd = GradientDescent(learning_rate=FixedLR(eta), callback=callback)
            gd.fit(f, None, None)
            fig1 = plot_descent_path(f_model, np.array(weights), title=f"Decent plot of {f_name} with eta:{eta}")
            # fig1.write_image(f"ex5_pdf/decent_plot_{f_name}_eta_{eta}.pdf")

            fig2 = go.Figure()
            fig2.update_layout(title=f"Convergence Rate: Norm {f_name}, eta:{eta}", xaxis_title="Iterations",
                               yaxis_title=f"{f_name} Norm")
            fig2.add_trace(go.Scatter(x=np.arange(len(values)), y=values, mode="markers", showlegend=False))
            # fig2.write_image(f"ex5_pdf/norm_{f_name}_eta_{eta}_iterations.pdf")
            print(f"the lowest norm in {f_name} and eta {eta} is {np.min(values)}")


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    raise NotImplementedError()

    # Plot algorithm's convergence for the different values of gamma
    raise NotImplementedError()

    # Plot descent path for gamma=0.95
    raise NotImplementedError()


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    log_model = LogisticRegression()
    log_model.fit(X_train.to_numpy(), y_train.to_numpy())
    proba_vec_test = log_model.predict_proba(X_test.to_numpy())
    fpr, tpr, thresholds = roc_curve(y_test, proba_vec_test)
    fig1 = go.Figure(
        data=[
            go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds, name="", showlegend=False, marker_size=5,
                       hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
                         xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                         yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$")))

    # fig1.write_image(f"ex5_pdf/roc_curve.pdf")

    best_alpha_id = np.argmax(tpr - fpr)
    print(f"the best alpha is {thresholds[best_alpha_id]}")

    pred_best_alpha = np.where(proba_vec_test > thresholds[best_alpha_id], 1, 0)
    best_alpha_loss = misclassification_error(y_test.to_numpy(), pred_best_alpha)
    print(f"loss for best alpha is: {best_alpha_loss}")

    # Plotting convergence rate of logistic regression over SA heart disease data

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    l1_train_scores, l2_train_scores, l1_validation_scores, l2_validation_scores = [], [], [], []
    lambdas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]

    for lam in lambdas:
        log_l1 = LogisticRegression(solver=GradientDescent(FixedLR(1e-4), max_iter=20000), penalty="l1", lam=lam,
                                    alpha=0.5)
        log_l2 = LogisticRegression(solver=GradientDescent(FixedLR(1e-4), max_iter=20000), penalty="l2", lam=lam,
                                    alpha=0.5)

        l1_train_score, l1_validation_score = cross_validate(log_l1, X_train.to_numpy(), y_train.to_numpy(),
                                                             misclassification_error)
        l2_train_score, l2_validation_score = cross_validate(log_l2, X_train.to_numpy(), y_train.to_numpy(),
                                                             misclassification_error)

        l1_train_scores.append(l1_train_score)
        l1_validation_scores.append(l1_validation_score)
        l2_train_scores.append(l2_train_score)
        l2_validation_scores.append(l2_validation_score)

    best_l1_id = np.argmin(l1_validation_scores)
    best_l2_id = np.argmin(l2_validation_scores)

    best_l1_lam = lambdas[best_l1_id]
    best_l2_lam = lambdas[best_l2_id]
    best_l1_log = LogisticRegression(solver=GradientDescent(FixedLR(1e-4), max_iter=20000), penalty="l1",
                                     lam=best_l1_lam, alpha=0.5)
    best_l2_log = LogisticRegression(solver=GradientDescent(FixedLR(1e-4), max_iter=20000), penalty="l2",
                                     lam=best_l2_lam, alpha=0.5)

    best_l1_log.fit(X_train.to_numpy(), y_train.to_numpy())
    best_l2_log.fit(X_train.to_numpy(), y_train.to_numpy())

    print(
        f"The best lambda for L1 is {best_l1_lam} and the loss is "
        f"{best_l1_log.loss(X_test.to_numpy(), y_test.to_numpy())}")
    print(
        f"The best lambda for L2 is {best_l2_lam} and the loss is "
        f"{best_l2_log.loss(X_test.to_numpy(), y_test.to_numpy())}")


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    # compare_exponential_decay_ratcktes()
    fit_logistic_regression()
