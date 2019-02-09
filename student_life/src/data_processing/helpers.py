import pandas as pd
import numpy as np

from src.utils.aggregation_utils import mode


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


def get_resampled_aggregated_data(feature_data, feature_config, student_id):
    """
    @attention : Imputes missing value with -1.
    @param feature_data: Un-resampled data for the feature.
    @param feature_config: Configs for the specific feature.
    @return: Aggregated data on the resampled frequency.
    """
    assert "resample_freq_min" in feature_config.keys(), "Invalid config given!"

    # Extracting columns other than student id (These are the feature inference columns)
    feature_inference_cols = list(feature_data.columns.values)
    feature_inference_cols.remove("student_id")
    # Resampling and applying aggregate rule.
    resample_freq_min = feature_config["resample_freq_min"]
    resampled_feature_data = feature_data.resample(rule=str(resample_freq_min) + "T")
    aggregation_rule = get_aggregation_rule(feature_inference_cols, feature_config, student_id)
    aggregated_data = resampled_feature_data.agg(aggregation_rule)
    aggregated_data.fillna(value=-1, inplace=True)
    # Flattening all the columns.
    aggregated_data.columns = ['_'.join(col).strip() if 'student_id' not in col else 'student_id'
                               for col in aggregated_data.columns.values]

    return aggregated_data


def get_flattened_student_data_from_list(student_data, student_id):
    """

    @param student_data: A list of data frame with various features from the student_life data-set.
    @param student_id: Student id of the student.
    @return: flattened data-set after applying a left join.
    """

    for feature_df in student_data:
        assert "student_id" in feature_df .columns, "Invalid Student data, student id " \
                                                      "missing in one of the data frames."
    # Pre-processing
    feature_data_first = student_data[0]
    start_date = feature_data_first.index[0].round("1min")
    end_date = feature_data_first.index[-1].round("1min")
    flattened_df_index = pd.date_range(start_date, end_date, freq="min")
    flattened_df = pd.DataFrame(np.full(len(flattened_df_index), student_id),
                                index=flattened_df_index,
                                columns=["student_id"])

    for idx, feature_df in enumerate(student_data):
        feature_df_dropped_student_id = feature_df.drop("student_id", axis=1, inplace=False)
        flattened_df = flattened_df.join(feature_df_dropped_student_id, how='left', sort=True)

    return flattened_df
