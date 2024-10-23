import tomli_w
import tomllib


class TOMLParser:
    def __init__(self, config_file):
        """
        Create a Configuration object from a TOML configuration file.
        :param config_file: str, toml configuration file path.
        """
        self.__toml_dict = self._open_config_file(config_file)

    @property
    def toml_dict(self):
        """
        Return a dict of the loaded TOML configuration file.
        :return: dict.
        """
        return self.__toml_dict

    @toml_dict.setter
    def toml_dict(self, config_file):
        """
        Load a new TOML configuration file.
        :param config_file: str, toml configuration file path.
        :return: None.
        """
        self.__toml_dict = self._open_config_file(config_file)

    @staticmethod
    def _open_config_file(config_file):
        """
        A helper method to open a TOML configuration file.
        :param config_file: str, toml configuration file path.
        :return: a file object.
        """
        try:
            with open(config_file, mode="rb") as file_object:
                try:
                    return tomllib.load(file_object)
                except tomllib.TOMLDecodeError as err:
                    print(err)
        except FileNotFoundError as err:
            print(err)

    def __str__(self):
        """
        Print the contents of TOML configuration file.
        :return:
        """
        return str(self.__toml_dict)

    def save_config(self, file_path):
        """
        Save Configuration object to a TOML configuration file.
        :param file_path: str, toml configuration file path.
        :return: None.
        """
        try:
            with open(file_path, mode="wb") as file_object:
                try:
                    tomllib.loads(str(self.__toml_dict))
                except tomllib.TOMLDecodeError as err:
                    print(err)
                else:
                    tomli_w.dump(self.__toml_dict, file_object)
        except FileNotFoundError as err:
            print(err)
