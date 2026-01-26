import typer
import os
from pathlib import Path
from smartndvi import SUCCESS, DIR_ERROR, FILE_ERROR, DB_WRITE_ERROR, __app_name__
from utility.toml import TOML
import tomlkit

CONFIG_DIR = Path(typer.get_app_dir(__app_name__))
CONFIG_FILE_PATH = CONFIG_DIR / "config.toml"


def init_app(output_root_path: str) -> int:
    """
    Initialize the application.
    :param output_root_path: str, specifying the root path of the output.
    :return: int, status code.
    """
    config_status = _init_config_file()
    if config_status != SUCCESS:
        return config_status

    output_dir_status = _create_output_directory(output_root_path)
    if output_dir_status != SUCCESS:
        return output_dir_status
    return SUCCESS


def _init_config_file() -> int:
    try:
        CONFIG_DIR.mkdir(exist_ok=True)
    except OSError:
        return DIR_ERROR

    try:
        CONFIG_FILE_PATH.unlink(missing_ok=True)
    except OSError:
        return FILE_ERROR

    try:
        CONFIG_FILE_PATH.touch(exist_ok=True)
    except OSError:
        return FILE_ERROR
    return SUCCESS


def _create_output_directory(out_path: str) -> int:
    toml = TOML(CONFIG_FILE_PATH)
    general = tomlkit.table()

    # Add output root directory
    general.add("ndvi_workspace_root", out_path)

    # Add cache directory
    cache = tomlkit.table()
    cache.add("cache_root", os.path.join(out_path, "cache"))
    cache.add("ground_truth_image", os.path.join(out_path, "cache\\ground_truth_image"))
    cache.add("ground_truth_mask", os.path.join(out_path, "cache\\ground_truth_mask"))
    cache.add("ground_truth_landcover", os.path.join(out_path, "cache\\ground_truth_landcover"))
    cache.add("naip_sample_mask", os.path.join(out_path, "cache\\naip_sample_mask"))

    # Add output directory
    output = tomlkit.table()
    output.add("output_root", os.path.join(out_path, "output"))
    output.add("land_cover_maps", os.path.join(out_path, "output\\land_cover_maps"))
    output.add("vegetation_mask", os.path.join(out_path, "output\\vegetation_mask"))
    output.add("optimal_ndvi", os.path.join(out_path, "output\\optimal_ndvi"))

    # Add model directory
    model = tomlkit.table()
    model.add("model_root", os.path.join(out_path, "model"))
    model.add("config", os.path.join(out_path, "model\\config"))
    model.add("checkpoint", os.path.join(out_path, "model\\checkpoint"))

    general.add("Cache", cache)
    general.add("Output", output)
    general.add("Model", model)
    toml.toml_document.add("General", general)

    try:
        toml.save_config_file(CONFIG_FILE_PATH)
    except OSError:
        return DB_WRITE_ERROR
    return SUCCESS
