from abc import ABC, abstractmethod
import numpy as np
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline, griddata
from statsmodels.nonparametric.kernel_regression import KernelReg
import torch


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
    
    def fit_grid(self, delta, maturity, vol):
        """
        Fits a grid of implied volatilities based on delta and maturity values.
        This method takes in arrays of delta, maturity, and corresponding volatility
        values, constructs a grid using these inputs, and fits the model to the data.

        Parameters:
        -----------
        delta : 1D array-like
            Array of delta values representing the moneyness of options.
        maturity : 1D array-like
            Array of maturity values representing the time to expiration of options.
        vol : 2D array-like
            Array of implied volatility values corresponding to the delta and maturity grid.
         
        Returns:
        --------
        self : object
            Returns the instance of the class after fitting the model.
        """
        
        d, m = np.meshgrid(delta, maturity, indexing="ij")
        X = np.column_stack([d.ravel(), m.ravel()])
        y = vol.ravel()
        self.fit(X, y)
        return self

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

    # def plot(self, ax=None, resolution=10, **kwargs):
    #     """
    #     Plot the vol surface.
    #     """
    #     if not hasattr(self, "_fitted") or not self._fitted:
    #         raise RuntimeError("VolSurface must be fitted before calling plot().")

    #     maturity_min, maturity_max = self._maturity_range

    #     delta = np.linspace(0, 1, resolution + 1)[1:-1]
    #     maturity = np.linspace(maturity_min, maturity_max, resolution + 1)

    #     d, m = np.meshgrid(delta, maturity, indexing="ij")
    #     v = self.predict_grid(delta, maturity)

    #     if ax is None:
    #         fig = plt.figure()
    #         ax = fig.add_subplot(111, projection="3d")

    #     ax.plot_surface(d, m, v, **kwargs)
    #     ax.set_xlabel("Delta")
    #     ax.set_ylabel("Maturity")
    #     ax.set_zlabel("Implied Volatility")
    #     ax.set_title("Volatility Surface")
    #     return ax

    def plot(self, ax=None, resolution=10, cmap="viridis", figsize=(10, 8), **kwargs):
        """
        Plot the vol surface with enhanced visualization.
        
        Parameters:
        -----------
        ax : matplotlib.axes._subplots.Axes3DSubplot, optional
            The axes to plot on. If None, a new figure and axes are created.
        resolution : int, optional
            The resolution of the grid for plotting. Higher values give finer plots.
        cmap : str, optional
            The colormap to use for the surface plot.
        figsize : tuple, optional
            The size of the figure (width, height) in inches.
        **kwargs : dict
            Additional keyword arguments passed to `plot_surface`.
        """
        if not hasattr(self, "_fitted") or not self._fitted:
            raise RuntimeError("VolSurface must be fitted before calling plot().")

        maturity_min, maturity_max = self._maturity_range

        delta = np.linspace(0, 1, resolution + 1)[1:-1]
        maturity = np.linspace(maturity_min, maturity_max, resolution + 1)

        d, m = np.meshgrid(delta, maturity, indexing="ij")
        v = self.predict_grid(delta, maturity)

        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="3d")

        # Create the surface plot with a colormap
        surf = ax.plot_surface(
            d, m, v, cmap=cmap, edgecolor="k", linewidth=0.5, **kwargs
        )

        # Add a color bar for the surface plot
        cbar = plt.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
        cbar.set_label("Implied Volatility", fontsize=12)

        # Set axis labels and title with larger font sizes
        ax.set_xlabel("Delta", fontsize=12)
        ax.set_ylabel("Maturity", fontsize=12)
        ax.set_zlabel("Implied Volatility", fontsize=12)
        ax.set_title("Volatility Surface", fontsize=14)

        # Adjust tick label sizes
        ax.tick_params(axis="both", which="major", labelsize=10)

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

        delta_idx = np.digitize(delta, (self.delta_grid[:-1] + self.delta_grid[1:]) / 2)
        maturity_idx = np.digitize(
            maturity, (self.maturity_grid[:-1] + self.maturity_grid[1:]) / 2
        )

        # aggregate vols into grid
        # TODO optimize the for loop
        for i in range(len(vol)):
            d_idx, m_idx = delta_idx[i], maturity_idx[i]
            grid_vol[d_idx, m_idx] += vol[i]
            counts[d_idx, m_idx] += 1

        # take average
        with np.errstate(invalid="ignore"):
            grid_vol = grid_vol / counts
            grid_vol[counts == 0] = np.nan  # leave empty where no data

        # Fill missing values using griddata for interpolation
        x, y = np.indices(grid_vol.shape)
        x = x.flatten()
        y = y.flatten()
        v = grid_vol.flatten()
        grid_vol[np.isnan(grid_vol)] = griddata(
            (x[~np.isnan(v)], y[~np.isnan(v)]),
            v[~np.isnan(v)],
            (x[np.isnan(v)], y[np.isnan(v)]),
            method="linear",
        )

        # fill boundary NaNs with nearest neighbor interpolation
        x, y = np.indices(grid_vol.shape)
        v = grid_vol.flatten()
        x = x.flatten()
        y = y.flatten()
        grid_vol[np.isnan(grid_vol)] = griddata(
            (x[~np.isnan(v)], y[~np.isnan(v)]),
            v[~np.isnan(v)],
            (x[np.isnan(v)], y[np.isnan(v)]),
            method="nearest",
        )

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
    
class TrainedDecoderVolSurface(VolSurface):
    def __init__(self, model_type, model, maturity_range, random_src=None, latent=None):
        self.model_type = model_type
        self.model = model
        self.decoder = model.decoder
        # self.decoder = decoder
        self.device = next(self.decoder.parameters()).device
        self._fitted = True
        self._maturity_range = maturity_range
        self.random_src = random_src
        self.latent = latent
        if not (random_src or latent):
            raise ValueError("Either random_src or latent must be provided.")
        elif random_src and latent:
            raise ValueError("Only one of random_src or latent can be provided.")
        if random_src is not None:
            self._latent = next(self.random_src)
        elif latent is not None:
            self._latent = self.latent

    def refresh(self):
        if self.random_src is None:
            return
        self._latent = next(self.random_src)

    def _fit(self, delta, maturity, vol):
        # This method is not used in this class
        pass

    def _predict(self, delta, maturity):
        rand = torch.tensor(self._latent, dtype=torch.float32, device=self.device) 
        latent = rand.repeat(len(delta), 1)
        delta = torch.tensor(delta, dtype=torch.float32, device=self.device).reshape(-1, 1)
        maturity = torch.tensor(maturity, dtype=torch.float32, device=self.device).reshape(-1, 1) / 365.0
        if self.model_type == "vae_pw_ii":
            delta_embed = self.model.dltemb_net(self.model.dltembed(delta)).squeeze(1)  # shape [72, 4]
            maturity_embed = self.model.ttmemb_net(self.model.ttmembed(maturity)).squeeze(1)  # shape [72, 4]
            z = latent + delta_embed + maturity_embed
        else:
            z = torch.cat((latent, delta, maturity), dim=1).to(self.device)
        return self.decoder(z).detach().cpu().numpy()