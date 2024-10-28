import typer
from pathlib import Path
from smartndvi import __app_name__
from smartndvi import SUCCESS, DIR_ERROR, FILE_ERROR, DB_WRITE_ERROR, __app_name__
from utility.toml import TOML
import tomlkit

CONFIG_DIR = Path(typer.get_app_dir(__app_name__))
CONFIG_FILE_PATH = CONFIG_DIR / "config.toml"


def init_app(db_path: str) -> int:
    """
    Initialize the application.
    :param db_path: str, specifying the path of the database.
    :return: int, status code.
    """
    config_status = _init_config_file()
    if config_status != SUCCESS:
        return config_status


def _init_config_file() -> int:
    try:
        CONFIG_DIR.mkdir(exist_ok=True)
    except OSError:
        return DIR_ERROR

    try:
        CONFIG_FILE_PATH.touch(exist_ok=True)
    except OSError:
        return FILE_ERROR
    return SUCCESS


def _create_database(db_path: str) -> int:
    toml = TOML(CONFIG_FILE_PATH)
    general = tomlkit.table()
    general.add("database", db_path)
    toml.toml_document.add("General", general)
    try:
        toml.save_config_file(CONFIG_FILE_PATH)
    except OSError:
        return DB_WRITE_ERROR
    return SUCCESS
