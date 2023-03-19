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
    # pio.write_image(fig2, format='pdf', file="Question 2.pdf")

    # Question 3 - Plotting Empirical PDF of fitted model
    sorted_samples = np.sort(samples)
    scattered_data3 = go.Scatter(x=sorted_samples, y=uni_gau.pdf(sorted_samples), mode="markers")
    layout3 = go.Layout(title="PDF of samples",
                        xaxis={"title": "Sample value"}, yaxis={"title": "PDF value"})
    fig3 = go.Figure(data=[scattered_data3], layout=layout3)
    fig3.show()

    # pio.write_image(fig3, format='pdf', file="Question 3.pdf")


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
    f1_array = np.linspace(-10, 10, 200)
    f3_array = np.linspace(-10, 10, 200)

    log_like = np.zeros((f1_array.size, f3_array.size))

    for i in range(f1_array.size):
        for j in range(f3_array.size):
            mu5 = np.array([f1_array[i], 0, f3_array[j], 0])
            log_like[i, j] = multi_gau.log_likelihood(mu5, sigma, samples)

    heatmap_data5 = go.Heatmap(x=f1_array, y=f3_array, z=log_like, colorscale='Viridis')
    layout5 = go.Layout(title="Heatmap of the log-likelihood of linespace [-10,10]", xaxis={"title": "f1 values"},
                        yaxis={"title": "f3 values"})
    fig5 = go.Figure(data=[heatmap_data5], layout=layout5)
    fig5.show()
    # pio.write_image(fig5, format='pdf', file="Question 5.pdf")

    # Question 6 - Maximum likelihood
    max_index = np.argmax(log_like)
    max_row, max_col = np.unravel_index(max_index, log_like.shape)
    print("{:.3f} {:.3f}".format(f1_array[max_col], f3_array[max_col]))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
