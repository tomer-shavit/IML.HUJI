from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu, var = 10, 1
    uni_gau = UnivariateGaussian()
    samples = np.random.normal(mu, var, 1000)
    uni_gau.fit(samples)
    print((uni_gau.mu_, uni_gau.var_))

    # Question 2 - Empirically showing sample mean is consistent
    models_diff = []
    for i in range(10, 1001, 10):
        models_diff.append(np.abs(uni_gau.mu_ - UnivariateGaussian().fit(samples[:i]).mu_))
    scattered_data2 = go.Scatter(x=np.arange(0, 1001, 10), y=models_diff, mode="markers")
    layout2 = go.Layout(title="Distance between estimated and true value of expectation",
                        xaxis={"title": "Sample size"}, yaxis={"title": "Difference in expectation"})
    fig2 = go.Figure(data=[scattered_data2], layout=layout2)
    fig2.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    sorted_samples = np.sort(samples)
    scattered_data3 = go.Scatter(x=sorted_samples, y=uni_gau.pdf(sorted_samples), mode="markers")
    layout3 = go.Layout(title="PDF of samples",
                        xaxis={"title": "Sample value"}, yaxis={"title": "PDF value"})
    fig3 = go.Figure(data=[scattered_data3], layout=layout3)
    fig3.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    sigma = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    multi_gau = MultivariateGaussian()
    samples = np.random.multivariate_normal(mu, sigma, 1000)
    multi_gau.fit(samples)
    print(multi_gau.mu_)
    print(multi_gau.cov_)

    # Question 5 - Likelihood evaluation

    # Question 6 - Maximum likelihood


if __name__ == '__main__':
    np.random.seed(0)
    # test_univariate_gaussian()
    test_multivariate_gaussian()
