import pandas as pd


def linear_interpolation(series: pd.Series):
    return series.interpolate(method='linear')


def forward_fill(series: pd.Series):
    return series.fillna(method='ffill')
