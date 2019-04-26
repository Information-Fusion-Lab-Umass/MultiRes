def split_data_by_percentage(data_list, start_index: int = 0, percent: float = -1):
    """

    @param data_list: The data for which slice is required.
    @param start_index: all indices before this are not considered for slicing.
    @param percent: Percentage of data that contributes to the slice. If percent = -1,
           then everything from start_index to len(data) is returned.
    @return:
    """
    data_len = len(data_list)
    slice_length = round(data_len * percent / 100)

    assert 0 < percent <= 100 or percent == -1, "Percent value must be between 1 and 100 but got {}".format(percent)
    assert 0 <= start_index < data_len
    assert start_index + slice_length < data_len, "Over flow of data list. " \
                                                  "Enter smaller percent value or reduce the start_index."

    if percent == -1:
        slice_data = data_list[start_index:]
        end_index = data_len - 1
    else:
        slice_data = data_list[start_index: start_index + slice_length]
        end_index = start_index + slice_length

    return slice_data, end_index

