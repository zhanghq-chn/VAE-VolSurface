import numpy as np

import sys
import os
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.getenv("SRC_PATH"))

from src.volsurface import GridInterpVolSurface  # noqa: E402


def test_grid_interpolated_vol_surface_linear_function():
    # interpolate a simple linear function
    rng = np.random.default_rng(42)
    delta = rng.uniform(0, 1, 10000)
    maturity = rng.uniform(0, 1, 10000)
    vol = delta + maturity

    X = np.column_stack([delta, maturity])
    y = vol

    # Fit the model
    model = GridInterpVolSurface(kx=1, ky=1)
    model.fit(X, y)

    # Test predict accuracy
    pred = model.predict(np.array([[0.55, 0.8]]))[0]
    expected = 0.55 + 0.8
    assert np.isclose(pred, expected, atol=0.02)

    delta_grid = np.linspace(0.2, 0.8, 10)
    maturity_grid = np.linspace(0.5, 1.5, 8)
    grid_vol = model.predict_grid(delta_grid, maturity_grid)
    assert grid_vol.shape == (len(delta_grid), len(maturity_grid))


def test_grid_interpolated_vol_surface_with_missing_data():
    # interpolate a simple linear function with missing data
    rng = np.random.default_rng(42)
    delta = rng.uniform(0, 1, 10000)
    maturity = rng.uniform(0, 1, 10000)
    vol = delta + maturity

    mask_delta = (0.4 < delta) & (delta < 0.6)
    mask_maturity = (0.4 < maturity) & (maturity < 0.6)
    mask = (~mask_delta) & (~mask_maturity)
    delta = delta[mask]
    maturity = maturity[mask]
    vol = vol[mask]

    X = np.column_stack([delta, maturity])
    y = vol

    model = GridInterpVolSurface(kx=1, ky=1)
    model.fit(X, y)

    pred = model.predict(np.array([[0.5, 1.0]]))[0]
    expected = 0.5 + 1.0
    assert np.isclose(pred, expected, atol=0.02)


def test_grid_interpolated_vol_surface_constant_function():
    # interpolate a constant function
    rng = np.random.default_rng(42)
    delta = rng.uniform(0, 1, 10000)
    maturity = rng.uniform(0, 1, 10000)
    vol = delta + maturity

    mask_delta = (0.4 < delta) & (delta < 0.6)
    mask_maturity = (0.4 < maturity) & (maturity < 0.6)
    mask = (~mask_delta) & (~mask_maturity)
    delta = delta[mask]
    maturity = maturity[mask]
    vol = vol[mask]

    X = np.column_stack([delta, maturity])
    y = np.ones_like(vol)

    model = GridInterpVolSurface(kx=1, ky=1)
    model.fit(X, y)

    delta_grid = np.linspace(0.2, 0.8, 10)
    maturity_grid = np.linspace(0.5, 1.5, 8)
    grid_vol = model.predict_grid(delta_grid, maturity_grid)
    assert np.isclose(grid_vol, 1.0).all()


def test_grid_interpolated_vol_surface_with_nan_and_peaks():
    n = 7
    mid = n // 2
    x_axis = np.linspace(-0.5, 0.5, n)
    y_axis = np.linspace(0.1, 2.0, n)

    delta, maturity = np.meshgrid(x_axis, y_axis)
    delta = delta.flatten()
    maturity = maturity.flatten()

    vol = np.full((n, n), np.nan)
    vol[mid - 1 : mid + 2, mid - 1 : mid + 2] = 1.0
    vol[mid, mid] = 2.0

    vol = vol.flatten()
    model = GridInterpVolSurface(kx=1, ky=1, delta_grid=x_axis, maturity_grid=y_axis)
    X = np.column_stack([delta, maturity])
    y = vol
    model.fit(X, y)

    expected = np.ones((n, n))
    expected[mid, mid] = 2.0
    assert np.allclose(expected, model.predict(X).reshape((n, n)))


def test_grid_interpolated_vol_surface_with_complex_nan_pattern():
    n = 11
    mid = n // 2
    x_axis = np.linspace(-0.5, 0.5, n)
    y_axis = np.linspace(0.1, 2.0, n)

    delta, maturity = np.meshgrid(x_axis, y_axis)
    delta = delta.flatten()
    maturity = maturity.flatten()

    vol = np.full((n, n), np.nan)
    vol[mid - 4 : mid + 5, mid - 4 : mid + 5] = 1.0
    vol[mid - 3 : mid + 4, mid - 3 : mid + 4] = np.nan
    vol[mid, mid] = 2.0

    vol = vol.flatten()
    model = GridInterpVolSurface(kx=1, ky=1, delta_grid=x_axis, maturity_grid=y_axis)
    X = np.column_stack([delta, maturity])
    y = vol
    model.fit(X, y)

    pred = model.predict(X).reshape((n, n))
    assert np.allclose(np.ones(n), pred[0])
    assert np.allclose(np.linspace(1.0, 2.0, 5), pred[mid - 4 : mid + 1, mid])
    assert np.allclose(
        np.linspace(1.0, 2.0, 4), np.diag(pred[np.ix_(range(2, 6), range(2, 6))])
    )
