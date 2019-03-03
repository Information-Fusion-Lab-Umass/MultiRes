import pandas as pd


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


def evaluate_gender(flattened_student_data: pd.DataFrame) -> pd.DataFrame:
    # todo(abhinavshaw): Implement this.
    return flattened_student_data
