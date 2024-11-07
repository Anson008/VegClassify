from pathlib import Path

from smartndvi import DIR_ERROR, SUCCESS
from utility.toml import TOML
from typing import List

class WorkSpace:
    def __init__(self, config_file_path: Path):
        self._toml = TOML(config_file_path)

    def init_workspace(self) -> int:
        """
        Initialize the workspace of smartndvi.
        :return: None
        """
        try:
            self._create_directories(self.get_cache_paths())
            self._create_directories(self.get_output_paths())
            self._create_directories(self.get_model_paths())
            return SUCCESS
        except FileExistsError:
            return DIR_ERROR

    @staticmethod
    def _create_directories(paths: List[str]):
        for path in paths:
            try:
                Path(path).mkdir(parents=True, exist_ok=True)
            except FileExistsError as err:
                print(f"{err}: path '{path}' already exists and is not a directory.")
                raise err

    def get_cache_paths(self) -> List[str] | None:
        """
        Get cache paths in the config file.
        :return: list of str, a list of directories under "Cache" directory.
        """
        path_list = []
        try:
            path_list.append(self._toml.toml_document["General"]["Cache"]["ground_truth_image"])
            path_list.append(self._toml.toml_document["General"]["Cache"]["ground_truth_mask"])
            path_list.append(self._toml.toml_document["General"]["Cache"]["naip_sample_mask"])
        except KeyError as err:
            print(f"{err}: can't find cache path in config file.")
            return
        return path_list

    def get_output_paths(self) -> List[str] | None:
        """
        Get output paths in the config file.
        :return: list of str, a list of output directories.
        """
        path_list = []
        try:
            path_list.append(self._toml.toml_document["General"]["Output"]["land_cover_maps"])
            path_list.append(self._toml.toml_document["General"]["Output"]["vegetation_mask"])
            path_list.append(self._toml.toml_document["General"]["Output"]["optimal_ndvi"])
        except KeyError as err:
            print(f"{err}: can't find output path in config file.")
            return
        return path_list

    def get_model_paths(self) -> List[str] | None:
        """
        Get model paths in the config file.
        :return: list of str, a list of model directories.
        """
        path_list = []
        try:
            path_list.append(self._toml.toml_document["General"]["Model"]["config"])
            path_list.append(self._toml.toml_document["General"]["Model"]["checkpoint"])
        except KeyError as err:
            print(f"{err}: can't find output path in config file.")
            return
        return path_list

    def get_workspace_root(self):
        return self._toml.toml_document["General"]["ndvi_workspace_root"]