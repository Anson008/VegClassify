from typing import Optional, List
import typer
from pathlib import Path
from smartndvi import __app_name__, __version__, ERRORS, config, database, smartndvi_controller, SUCCESS
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
    if app_init_error != SUCCESS:
        typer.secho(
            f"Failed to create config file with '{ERRORS[app_init_error]}'",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)

    workspace = WorkSpace(Path(config.CONFIG_FILE_PATH))
    work_dir_init_error = workspace.init_workspace()
    if work_dir_init_error != SUCCESS:
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

@app.command(name="optimize")
def search_optimal_ndvi_threshold(
        naip_file_path: str = typer.Argument(),
        land_cover_metrics: str = typer.Option(
            None,
            "--land-cover",
            "-lc",
            help="Generate land-cover maps after optimal NDVI threshold is found."
        ),
) -> None:
    """

    :param naip_file_path:
    :param land_cover_metrics:
    :return:
    """
    controller = get_smartndvi_controller()
    try:
        controller.optimize_ndvi_threshold(naip_file_path, land_cover_metrics)
    except OSError as err:
        print(err)
        print("NAIP file or directory not exists.")


def get_smartndvi_controller() -> smartndvi_controller.SmartNDVIController:
    if config.CONFIG_FILE_PATH.exists():
        return smartndvi_controller.SmartNDVIController(Path(config.CONFIG_FILE_PATH))
    else:
        typer.secho(
            "Config file not found. Please run 'smartndvi init [{--work-dir | -wd} {workspace path}].'",
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

