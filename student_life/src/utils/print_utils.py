import bin.validations as validations


def data_debug_string(data: dict, seq_limit):
    """

    @param data:
    @param seq_limit: Integer value to limit the seq.
    @return: Returns samples example of small slices of data.
    """
    validations.validate_data_dict_keys(data)
    first_key = next(iter(data['data'].keys()))
    print('first_key: ', first_key)

    for idx, datum in enumerate(data['data'][first_key]):
        if idx == 3:
            print("Label: ", datum)
        else:
            print(datum[:seq_limit])
