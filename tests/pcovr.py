import numpy as np
from matplotlib import pyplot as plt

from utilities.sklearn_pcovr import PCovR
from utilities.plotting import plot_regression, plot_projection

data = np.load('./tests/CSD-test.npz')
X = data["X"]
Y = data["Y"]

for alpha in np.linspace(0,1,3):
    pcovr = PCovR(alpha = alpha,
                  n_components=2,
                  regularization=1e-6,
                  tol=1e-12)
    pcovr.fit(X, Y)

    T = pcovr.transform(X)
    Xr = pcovr.inverse_transform(T)
    Yp = pcovr.predict(X)

    fig, ax = plt.subplots(1, 3, figsize=(12,4))
    plot_projection(Y, T, fig=fig, ax=ax[0],
                    xlabel=r"$PC_1$", ylabel=r"$PC_2$",
                    cbar=False,
                    title="Projection of X into Latent Space")
    plot_regression(X, Xr, fig=fig, ax=ax[1],
                    xlabel=r"$\mathbf{X}$", ylabel=r"$\mathbf{TP}_{TX}$",
                    cbar=False,
                    title="Reconstruction of X from Latent Space")
    plot_regression(Y, Yp, fig=fig, ax=ax[2],
                    xlabel=r"$\mathbf{Y}$", ylabel=r"$\mathbf{TP}_{TY}$",
                    cbar=False,
                    title="Regression of Y from Latent Space")
    plt.show()
