from abc import ABC, abstractmethod
import numpy as np
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from statsmodels.nonparametric.kernel_regression import KernelReg


class VolSurface(BaseEstimator, ABC):
    def fit(self, X, y):
        """
        Fit the vol surface to the data.
        X: 2D array-like of shape (n_samples, 2) with delta and maturity.
        y: 1D array-like of shape (n_samples,) with implied volatilities.
        """
        self._fitted = True
        self._maturity_range = (np.min(X[:, 1]), np.max(X[:, 1]))
        self._fit(X[:, 0], X[:, 1], y)
        return self

    def predict(self, X):
        """
        Evaluate implied volatility at the given strike and maturity.
        """
        return self._predict(X[:, 0], X[:, 1])

    def predict_grid(self, delta, maturity):
        """
        Evaluate implied volatility at the given strike and maturity.
        delta: 1D array-like of shape (n_samples,) with deltas.
        maturity: 1D array-like of shape (n_samples,) with maturities.
        """
        d, m = np.meshgrid(delta, maturity, indexing="ij")
        vol = self._predict(d.ravel(), m.ravel())
        return vol.reshape(d.shape)

    @abstractmethod
    def _fit(self, delta, maturity, vol):
        """
        Fit the vol surface to the data.
        delta: 1D array-like of shape (n_samples,) with deltas.
        maturity: 1D array-like of shape (n_samples,) with maturities.
        vol: 1D array-like of shape (n_samples,) with implied volatilities.
        """
        pass

    @abstractmethod
    def _predict(self, delta, maturity) -> np.ndarray:
        """
        Evaluate implied volatility at the given strike and maturity.
        delta: 1D array-like of shape (n_samples,) with deltas.
        maturity: 1D array-like of shape (n_samples,) with maturities.
        """
        pass

    def plot(self, ax=None, resolution=10, **kwargs):
        """
        Plot the vol surface.
        """
        if not hasattr(self, "_fitted") or not self._fitted:
            raise RuntimeError("VolSurface must be fitted before calling plot().")

        maturity_min, maturity_max = self._maturity_range

        delta = np.linspace(0, 1, resolution + 1)
        maturity = np.linspace(maturity_min, maturity_max, resolution + 1)

        d, m = np.meshgrid(delta, maturity, indexing="ij")
        v = self.predict_grid(delta, maturity)

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

        ax.plot_surface(d, m, v, **kwargs)
        ax.set_xlabel("Delta")
        ax.set_ylabel("Maturity")
        ax.set_zlabel("Implied Volatility")
        ax.set_title("Volatility Surface")
        return ax


class GridInterpVolSurface(VolSurface):
    """
    A class to fit a volatility surface using grid interpolation.
    This class uses a grid-based approach to interpolate the implied volatility
    surface based on the provided delta and maturity data.
    The value of each grid point is computed as the average of the input implied volatilities.
    For inputs that is not on the grid, the class uses interpolation to estimate the implied volatility.
    """

    def __init__(self, delta_grid=None, maturity_grid=None, kx=3, ky=3):
        self.kx = kx
        self.ky = ky
        self.delta_grid = delta_grid  # if None, computed from data in fit
        self.maturity_grid = maturity_grid

    def _fit(self, delta, maturity, vol):
        delta = np.asarray(delta)
        maturity = np.asarray(maturity)
        vol = np.asarray(vol)

        if self.delta_grid is None:
            self.delta_grid = np.linspace(np.min(delta), np.max(delta), 11)
        if self.maturity_grid is None:
            self.maturity_grid = np.linspace(np.min(maturity), np.max(maturity), 11)

        grid_vol = np.zeros((len(self.delta_grid), len(self.maturity_grid)))
        counts = np.zeros_like(grid_vol, dtype=int)

        delta_idx = np.digitize(delta, self.delta_grid) - 1
        maturity_idx = np.digitize(maturity, self.maturity_grid) - 1

        # aggregate vols into grid
        # TODO optimize the for loop
        for i in range(len(vol)):
            d_idx, m_idx = delta_idx[i], maturity_idx[i]
            if 0 <= d_idx < len(self.delta_grid) and 0 <= m_idx < len(
                self.maturity_grid
            ):
                grid_vol[d_idx, m_idx] += vol[i]
                counts[d_idx, m_idx] += 1

        # take average
        with np.errstate(invalid="ignore"):
            grid_vol = grid_vol / counts
            grid_vol[counts == 0] = np.nan  # leave empty where no data

        # Optionally: fill missing with nearest, interpolate, etc.
        # For simplicity, fill with nearest non-nan value
        from scipy.ndimage import generic_filter

        mask = np.isnan(grid_vol)
        with np.errstate(invalid="ignore"):
            grid_vol = generic_filter(grid_vol, np.nanmean, size=3, mode="nearest")
        grid_vol = np.where(mask, grid_vol, grid_vol)

        self.grid_vol = grid_vol

        self._interp = RectBivariateSpline(
            self.delta_grid, self.maturity_grid, grid_vol, kx=self.kx, ky=self.ky
        )

    def _predict(self, delta, maturity) -> np.ndarray:
        return self._interp.ev(delta, maturity)


class KernelVolSurface(VolSurface):
    def __init__(self, bandwidth=None, kernel="gaussian"):
        self.bandwidth = bandwidth
        self.kernel = kernel

    def _fit(self, delta, maturity, vol):
        if self.bandwidth is None:
            n = len(delta)
            self.bandwidth = (
                np.array([np.std(delta), np.std(maturity)]) * (n ** (-1 / 5)) * 1.06
            )

        self.model = KernelReg(
            endog=vol,
            exog=np.column_stack((delta, maturity)),
            var_type="cc",
            bw=self.bandwidth,
        )

    def _predict(self, delta, maturity) -> np.ndarray:
        result, _ = self.model.fit(np.column_stack((delta, maturity)))
        return result
