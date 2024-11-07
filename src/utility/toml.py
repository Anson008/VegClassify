import tomlkit
from pathlib import Path


class TOML:
    def __init__(self, toml_file_path: Path):
        """
        Create a Configuration object from a TOML configuration file.
        :param toml_file_path: str, toml configuration file path.
        """
        self.__toml_document = self._open_config_file(toml_file_path)

    @property
    def toml_document(self) -> tomlkit.TOMLDocument:
        """
        Return a TOMLDocument of the loaded TOML configuration file.
        :return: dict.
        """
        return self.__toml_document

    @toml_document.setter
    def toml_document(self, config_file: Path):
        """
        Load a new TOML configuration file.
        :param config_file: Path, toml configuration file path.
        :return: None.
        """
        self.__toml_document = self._open_config_file(config_file)

    @staticmethod
    def _open_config_file(config_file: Path) -> tomlkit.TOMLDocument:
        """
        A helper method to open a TOML configuration file.
        :param config_file: str, toml configuration file path.
        :return: a TOMLDocument.
        """
        try:
            with config_file.open(mode="rt", encoding="utf-8") as fp:
                return tomlkit.load(fp)
        except FileNotFoundError as err:
            print(err)

    def __str__(self):
        """
        Print the contents of TOML configuration file.
        :return:
        """
        return self.__toml_document.as_string()

    def save_config_file(self, toml_file_path: Path):
        """
        Save Configuration object to a TOML configuration file.
        :param toml_file_path: str, toml configuration file path.
        :return: None.
        """
        try:
            with toml_file_path.open(mode="wt", encoding="utf-8") as fp:
                tomlkit.dump(self.__toml_document, fp)
        except FileNotFoundError as err:
            print(err)
