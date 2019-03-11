import pandas as pd
import numpy as np


def day_of_week(flattened_student_data: pd.DataFrame) -> pd.DataFrame:
    """

    @param flattened_student_data: Flattened student data in form of Data Frame.
    @return: Day of week (integer value from 0-6) as a feature in separate column.
    """
    flattened_student_data.insert(loc=1, column='day_of_week', value=flattened_student_data.index.dayofweek)

    return flattened_student_data


def epoch_of_day(flattened_student_data: pd.DataFrame) -> pd.DataFrame:
    """
    @brief: Day has been defined into 4 epoch of 6 hours each.
    @param flattened_student_data: Flattened student data in form of Data Frame.
    @return: Epoch of day (integer value from 0-3) as a feature in separate column.
    """

    flattened_student_data.insert(loc=2, column='epoch_of_day', value=flattened_student_data.index.hour)
    flattened_student_data['epoch_of_day'] = flattened_student_data['epoch_of_day'].map(evaluate_epoch)
    return flattened_student_data


def evaluate_epoch(hour):
    if 0 <= hour < 6:
        return 0
    if 6 <= hour < 12:
        return 1
    if 12 <= hour < 18:
        return 2
    else:
        return 3


def time_since_last_label_min(flattened_student_data: pd.DataFrame) -> pd.DataFrame:
    null_mask = flattened_student_data['stress_level_mode'].isnull()
    flattened_student_data.insert(loc=3, column='time_since_last_label', value=flattened_student_data.index)
    flattened_student_data.loc[null_mask, 'time_since_last_label'] = np.nan
    flattened_student_data['time_since_last_label'].fillna(method='ffill', inplace=True)
    # Filling the sequences which do not have a last label and appear first in the data set.
    flattened_student_data['time_since_last_label'] = (flattened_student_data.index - flattened_student_data[
        'time_since_last_label']).astype('timedelta64[m]')

    return flattened_student_data


def time_to_next_label_min(flattened_student_data: pd.DataFrame) -> pd.DataFrame:
    null_mask = flattened_student_data['stress_level_mode'].isnull()
    flattened_student_data.insert(loc=4, column='time_to_next_label', value=flattened_student_data.index)
    flattened_student_data.loc[null_mask, 'time_to_next_label'] = np.nan
    flattened_student_data['time_to_next_label'].fillna(method='bfill', inplace=True)
    flattened_student_data['time_to_next_label'] = (flattened_student_data[
                          'time_to_next_label'] - flattened_student_data.index).astype('timedelta64[m]')

    return flattened_student_data


def evaluate_gender(flattened_student_data: pd.DataFrame) -> pd.DataFrame:
    # todo(abhinavshaw): Implement this.
    return flattened_student_data
