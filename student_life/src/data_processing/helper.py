import pandas as pd
import numpy as np
import definitions

from src.utils.aggregation_utils import mode
from src.utils import validation_utils as validate



def get_aggregation_rule(feature_inference_cols, feature_config, student_id):
    """

    @param feature_inference_cols:
    @param feature_config:
    @return: Return Aggregation Rule for the feature based on the configuration.
    """
    def value(array_like):
        return student_id

    # List of custom aggregate function.
    custom_aggregates = []
    simple_aggregates = feature_config['simple_aggregates']

    if "mode" in feature_config['custom_aggregates']:
        custom_aggregates.append(mode)

    rule = {"student_id": value}

    for col in feature_inference_cols:
        rule[col] = simple_aggregates + custom_aggregates

    return rule


def get_resampled_aggregated_data(feature_data: pd.DataFrame, feature_config, student_id)->pd.DataFrame:
    """

    @attention : Imputes missing value with -1.
    @param feature_data: Un-resampled data for the feature.
    @param feature_config: Configs for the specific feature.
    @return: Aggregated data on the resampled frequency.
    """
    validate.validate_config_key('resample_freq_min', config=feature_config)

    # Extracting columns other than student id (These are the feature inference columns)
    feature_inference_cols = list(feature_data.columns.values)
    feature_inference_cols.remove("student_id")
    # Resampling and applying aggregate rule.
    resample_freq_min = feature_config[definitions.RESAMPLE_FREQ_CONFIG_KEY]
    resampled_feature_data = feature_data.resample(rule=str(resample_freq_min) + "T")
    aggregation_rule = get_aggregation_rule(feature_inference_cols, feature_config, student_id)
    aggregated_data = resampled_feature_data.agg(aggregation_rule)
    aggregated_data.fillna(value=-1, inplace=True)
    # Flattening all the columns.
    aggregated_data.columns = ['_'.join(col).strip() if 'student_id' not in col else 'student_id'
                               for col in aggregated_data.columns.values]

    return aggregated_data


def get_flattened_student_data_from_list(student_data: pd.DataFrame, student_id)->pd.DataFrame:
    """

    @param student_data: A list of data frame with various features from the student_life data-set.
    @param student_id: Student id of the student.
    @return: flattened data-set after applying a left join.
    """
    validate.validate_student_id_in_data(*student_data)

    # Pre-processing
    feature_data_first = student_data[0]
    start_date = feature_data_first.index[0].floor("D")
    end_date = feature_data_first.index[-1].floor("D")
    flattened_df_index = pd.date_range(start_date, end_date, freq=definitions.DEFAULT_BASE_FREQ)
    flattened_df = pd.DataFrame(np.full(len(flattened_df_index), student_id),
                                index=flattened_df_index,
                                columns=["student_id"])

    for idx, feature_df in enumerate(student_data):
        feature_df_dropped_student_id = feature_df.drop("student_id", axis=1, inplace=False)
        flattened_df = flattened_df.join(feature_df_dropped_student_id, how='left', sort=True)

    return flattened_df


def replace_neg_one_with_nan(df):
    """

    @param df: DataFrame to be processed.
    @return: Replaces -1(int) or -1.0(double) all rows to np.nan.
    """
    # Converting any missing values to NaN.
    return df.replace(to_replace={-1: np.nan, -1.0: np.nan}, value=None, inplace=False)


def remove_days_with_no_stress_label(flattened_student_data: pd.DataFrame)->pd.DataFrame:
    """

    @param flattened_student_data: Flattened data of student. Must contain stress_level_mode as
                                   one of the columns.
    @return: processed data frame where sequences belonging to the same day are removed where there
            are no stress label.
    """

    validate.validate_student_id_in_data(flattened_student_data)

    stress_not_null_df = flattened_student_data[flattened_student_data['stress_level_mode'].notnull()]
    stress_not_null_indices = stress_not_null_df.index
    td = pd.Timedelta('1 days')

    for idx, time_index in enumerate(stress_not_null_indices):
        floored_time_index = time_index.floor("D")
        if idx == 0:
            time_indices_to_keep = pd.date_range(floored_time_index,
                                                 floored_time_index + td,
                                                 freq=definitions.DEFAULT_BASE_FREQ,
                                                 closed="left")
        else:
            time_indices_to_keep = time_indices_to_keep.union(
                pd.date_range(floored_time_index,
                              floored_time_index + td,
                              freq=definitions.DEFAULT_BASE_FREQ,
                              closed="left"))

    indices_to_be_dropped = flattened_student_data.index.difference(time_indices_to_keep)
    flattened_student_data_dropped = flattened_student_data.drop(indices_to_be_dropped)

    return flattened_student_data_dropped


def get_time_deltas_min(flattened_student_data: pd.DataFrame) -> pd.DataFrame:
    """

    @param flattened_student_data:
    @return: Returns time deltas of the last observed data in a DataFrame.
    """
    time_deltas = pd.DataFrame(index=flattened_student_data.index,
                               columns=flattened_student_data.columns,
                               dtype=float)
    last_observed_time = {}
    for col in flattened_student_data.columns:
        last_observed_time[col] = flattened_student_data.index[0]

    cols = flattened_student_data.columns
    rows = len(flattened_student_data)

    for i in range(0, rows):
        for col_idx, col in enumerate(cols):
            is_col_nan = np.isnan(flattened_student_data.iloc[i][col])
            if not is_col_nan:
                last_observed_time[col] = flattened_student_data.index[i]

            delta = time_deltas.index[i] - last_observed_time[col]
            # converting to minutes.
            time_deltas.iloc[i, col_idx] = delta.total_seconds() / 60

    return time_deltas
