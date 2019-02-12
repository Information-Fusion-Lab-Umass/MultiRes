import yaml


def read_yaml(file_path):
    """Util to read Yaml File."""

    # Reading from YML file.
    with open(file_path, "r") as ymlfile:
        yaml_file = yaml.load(ymlfile)

    return yaml_file
