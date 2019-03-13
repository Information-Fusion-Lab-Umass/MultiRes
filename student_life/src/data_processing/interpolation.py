import pandas as pd


def linear_interpolation(series: pd.Series):
    return series.interpolate(method='linear')


def forward_fill(series: pd.Series):
    return series.fillna(method='ffill')


def mean_fill(series: pd.Series):
    mean = series.mean(skipna=True)
    return series.fillna(mean.to_dict())


def mode_fill(series: pd.Series):
    mode = series.mode() 
    return series.fillna(mode.to_dict())
