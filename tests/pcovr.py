import numpy as np
from matplotlib import pyplot as plt

from utilities.sklearn_pcovr import PCovR
from utilities.plotting import plot_regression, plot_projection
from utilities.colorbars import load
load()

data = np.load('./tests/CSD-test.npz')
X = data["X"]
Y = data["Y"]

fig, axes = plt.subplots(3,3, figsize=(4,4))

for i, alpha in enumerate(np.linspace(0,1,3)):
    pcovr = PCovR(alpha = alpha,
                  n_components=2,
                  regularization=1e-6,
                  tol=1e-12)
    pcovr.fit(X, Y)

    T = pcovr.transform(X)
    Xr = pcovr.inverse_transform(T)
    Yp = pcovr.predict(X)

    plot_projection(Y, T, fig=fig, ax=axes[0][i],
                    x_label=r"$PC_1$", y_label=r"$PC_2$",
                    cbar=False, s=4, font_scaled=True,
                    cmapX='cbarHot_0.3_1.05'
                    )
    plot_regression(X, Xr, fig=fig, ax=axes[1][i],
                    x_label=r"$\mathbf{X}$", y_label=r"$\mathbf{TP}_{TX}$",
                    cbar=False, s=4, font_scaled=True,
                    cmapY='bone_r_0.2_1.0'
                    )
    plot_regression(Y, Yp, fig=fig, ax=axes[2][i],
                    x_label=r"$\mathbf{Y}$", y_label=r"$\mathbf{TP}_{TY}$",
                    cbar=False, s=4, font_scaled=True,
                    cmapY='bone_r_0.2_1.0'
                    )
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    axes[0][i].set_title(r'$\alpha =$'+f' {alpha}')

for ax in axes.flatten():
    ax.set_xticklabels([])
    ax.set_yticklabels([])


plt.show()
