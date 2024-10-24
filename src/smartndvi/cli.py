from typing import Optional
import typer
from smartndvi import __app_name__, __version__

app = typer.Typer()


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

