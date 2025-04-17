import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .logger import setup_logger
logger = setup_logger(__name__)


class VolSurfPointwiseDataset(Dataset):
    def __init__(self, pw_grid_data, pw_vol_data, surface_data, mapping_ids):
        self.pw_grid_data = pw_grid_data
        self.pw_vol_data = pw_vol_data
        self.surface_data = surface_data
        self.mapping_ids = mapping_ids

    def __len__(self):
        return len(self.pw_grid_data)

    def __getitem__(self, idx):
        pw_grid = self.pw_grid_data[idx]
        pw_vol = self.pw_vol_data[idx]
        mapping = self.mapping_ids[idx]
        surface = self.surface_data[mapping]
        return pw_grid, pw_vol, surface


def clean_data(df_raw):
    df_raw['ttm'] = (df_raw['exdate'] - df_raw['date']).dt.days

    idx_filter_bid = df_raw['best_bid'] > 0.0
    idx_filter_spread = \
        (df_raw['best_offer'] - df_raw['best_bid'] >= 0.0) & \
        (((df_raw['best_offer'] - df_raw['best_bid']) / (df_raw['best_bid'] + df_raw['best_offer'] ) * 2) <= 0.1) # this alone filters about 6%
    idx_filter_impl_vol = ~ df_raw['impl_volatility'].isna()
    idx_filter_leverage = df_raw['delta'].abs().between(*np.nanquantile(df_raw['delta'].abs(), [0.01, 0.99]))
    idx_filter_no_trade_consistency = (df_raw['volume'] > 0) == (df_raw['date'] == df_raw['last_date']) # probably corrupted data, current count of only 86
    idx_filter_ttm = df_raw['ttm'] < 1e5

    df = df_raw[
        idx_filter_bid & 
        idx_filter_spread & 
        idx_filter_impl_vol & 
        idx_filter_leverage & 
        idx_filter_no_trade_consistency &
        idx_filter_ttm
    ].copy()

    logger.info(f"Bad data - Filtered {df_raw.shape[0] - df.shape[0]} rows, Retained sample {df.shape[0] / df_raw.shape[0]:.2%}")

    # print(f"Retained sample {df.shape[0] / df_raw.shape[0]:.2%}")
    df['days_since_last'] = (df['date'] - df['last_date']).dt.days
    df['traded'] = (df['volume'] > 0)

    df['consecutive_traded'] = df.groupby('symbol')['traded'].transform(
        lambda x: (x != x.shift(1)).cumsum()
    )
    df.loc[~df['traded'], 'consecutive_traded'] = np.nan

    df['consecutive_traded_len'] = df.groupby(['symbol', 'consecutive_traded'])['traded'].transform('count')

    logger.info("Consecutive trading stats completed")

    def filter_consecutive_trading(df, consecutive_threshold):
        """
        Filter the DataFrame to include only rows where the options have been trading for at least n days consecutively.
        """
        consecutive_traded_start = df.loc[
            (df['traded']) &
            (df['consecutive_traded_len'] >= consecutive_threshold)]
        consecutive_traded_start = consecutive_traded_start.loc[
            (consecutive_traded_start.groupby('symbol').cumcount() == 0), 
            ['symbol', 'date']
        ].rename(columns={'date': 'consecutive_traded_start'})
        df_active = df.merge(
            consecutive_traded_start,
            how='left',
            on='symbol'
        )
        df_active = df_active[df_active['date'] >= df_active['consecutive_traded_start']]
        return df_active

    df_active = filter_consecutive_trading(df, consecutive_threshold=5)

    logger.info(f"Consecutive trading - Filtered {df.shape[0] - df_active.shape[0]} rows, Retained sample {df_active.shape[0] / df.shape[0]:.2%}")

    delta = df_active['delta']
    df_active['moneyness'] = np.where(delta > 0, delta, 1 + delta)

    logger.info("Moneyness calculation completed")

    return df_active