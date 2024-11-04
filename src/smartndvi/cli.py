from typing import Optional, List
import typer
from pathlib import Path
from smartndvi import __app_name__, __version__, ERRORS, config, database, smartndvi_controller
from smartndvi.workspace import WorkSpace

DEFAULT_WORK_DIR = Path.home().joinpath("smartndvi_workspace")
app = typer.Typer()


@app.command()
def init(
        work_dir: str = typer.Option(
            str(DEFAULT_WORK_DIR),
            "--work-dir",
            "-wd",
            prompt="smartndvi workspace location?"
        ),
) -> None:
    """
    Initialize the smartndvi database.
    :param work_dir: str, the working directory of smartndvi.
    :return: None.
    """
    app_init_error = config.init_app(work_dir)
    if app_init_error:
        typer.secho(
            f"Failed to create config file with '{ERRORS[app_init_error]}'",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)

    workspace = WorkSpace(Path(config.CONFIG_FILE_PATH))
    work_dir_init_error = workspace.init_workspace()
    if work_dir_init_error:
        typer.secho(
            f"Failed to create workspace with '{ERRORS[work_dir_init_error]}'",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)
    else:
        typer.secho(
            f"The smartndvi working directory is {workspace.get_workspace_root()}",
            fg=typer.colors.GREEN
        )

    # db_init_error = database.init_database(Path(db_path))
    # if db_init_error:
    #     typer.secho(
    #         f"Failed to create database with '{ERRORS[db_init_error]}",
    #         gf=typer.colors.RED,
    #     )
    #     raise typer.Exit(1)
    # else:
    #     typer.secho(f"The smartndvi database is {db_path}", fg=typer.colors.GREEN)


def get_smartndvi_controller() -> smartndvi_controller.SmartNDVIController:
    if config.CONFIG_FILE_PATH.exists():
        db_path = database.get_database_path(config.CONFIG_FILE_PATH)
    else:
        typer.secho(
            f"Config file not found. Please run 'smartndvi init.'",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)
    if db_path.exists():
        return smartndvi_controller.SmartNDVIController(db_path)
    else:
        typer.secho(
            "Database not found. Please run 'smartndvi init'",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)


def _show_version(value: bool) -> None:
    if value:
        typer.echo(f"{__app_name__} v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
        version: Optional[bool] = typer.Option(
            None,
            "--version",
            "-v",
            help="Show the application's version and exit.",
            callback=_show_version,
            is_eager=True,
        )
) -> None:
    return

