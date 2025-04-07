import numpy as np

import sys
import os
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.getenv("SRC_PATH"))

from src.volsurface import GridInterpVolSurface  # noqa: E402


def test_grid_interpolated_vol_surface():
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
    pred = model.predict(np.array([[0.55, 1.05]]))[0]
    expected = 0.55 + 1.05
    assert np.isclose(pred, expected, atol=0.05)

    # Test predict_grid shape
    delta_grid = np.linspace(0.2, 0.8, 10)
    maturity_grid = np.linspace(0.5, 1.5, 8)
    grid_vol = model.predict_grid(delta_grid, maturity_grid)
    assert grid_vol.shape == (len(delta_grid), len(maturity_grid))

    # interpolate a simple linear function with missing data
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

    pred = model.predict(np.array([[0.45, 0.95]]))[0]
    expected = 0.45 + 0.95
    assert np.isclose(pred, expected, atol=0.05)

    # interpolate a constant function
    y = np.ones_like(vol)

    model = GridInterpVolSurface(kx=1, ky=1)
    model.fit(X, y)

    delta_grid = np.linspace(0.2, 0.8, 10)
    maturity_grid = np.linspace(0.5, 1.5, 8)
    grid_vol = model.predict_grid(delta_grid, maturity_grid)
    assert np.isclose(grid_vol, 1.0).all()
