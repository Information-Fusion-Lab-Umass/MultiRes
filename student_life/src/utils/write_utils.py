import os
import pickle


def df_to_csv(df, file_name, path_to_folder):
    """

    @param file_name: Prefix to be added.
    @param path_to_folder: Path to folder where file has to be written.
    @return: None!
    """

    if not os.path.exists(path_to_folder):
        os.mkdir(path_to_folder)
    student_binned_data_file_path = os.path.join(path_to_folder,
                                                 file_name)
    df.to_csv(path_or_buf=student_binned_data_file_path)


def data_structure_to_pickle(data, file_path):
    if not os.path.exists(file_path):
        print("File {} doesn't exists and will be created.".format(file_path))

    with open(file_path, 'wb') as handle:
        pickle.dump(data, handle)
