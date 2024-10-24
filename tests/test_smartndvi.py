from typer.testing import CliRunner
from smartndvi import __app_name__, __version__, cli


runner = CliRunner()


def test_version():
    res = runner.invoke(cli.app, ["--version"])
    assert res.exit_code == 0
    assert f"{__app_name__} v{__version__}\n" in res.stdout