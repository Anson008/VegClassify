import json
import pytest
from datetime import datetime
from typer.testing import CliRunner
from smartndvi import __app_name__, __version__, cli, DB_READ_ERROR, SUCCESS, smartndvi_controller


runner = CliRunner()


# test_data1 = {
#                 "id": 1,
#                 "naip_filename": "naip_test1.tif",
#                 "datetime_processed": datetime(2024, 6, 16),
#                 "optimal_ndvi_threshold": {
#                     "on_kappa": 0.11,
#                     "on_accuracy": 0.10,
#                 }
#               }
#
# test_data2 = {
#                 "id": 2,
#                 "naip_filename": "naip_test2.tif",
#                 "datetime_processed": datetime(2024, 8, 26),
#                 "optimal_ndvi_threshold": {
#                     "on_kappa": 0.13,
#                     "on_accuracy": 0.12,
#                 }
#               }
#
#
# @pytest.fixture
# def mock_json_file(tmp_path):
#     ndvi_res = [{
#                     "id": 0,
#                     "naip_filename": "naip_test0.tif",
#                     "datetime_processed": datetime.now().replace(microsecond=0),
#                     "optimal_ndvi_threshold": {
#                         "on_kappa": 0.14,
#                         "on_accuracy": 0.13,
#                     }
#                  }]
#     db_file_path = tmp_path / "smartndvi_test.json"
#     with db_file_path.open("w") as db:
#         json.dump(ndvi_res, db, indent=4)
#     return db_file_path


# @pytest.mark.parametrize(
#     "naip_filename, datetime_processed, optimal_ndvi_threshold, expected",
#     [
#         pytest.param(
#             test_data1["naip_filename"],
#             test_data1["datetime_processed"],
#             test_data1["optimal_ndvi_threshold"],
#             (test_data1[""])
#         )
#     ]
# )

def test_version():
    res = runner.invoke(cli.app, ["--version"])
    assert res.exit_code == 0
    assert f"{__app_name__} v{__version__}\n" in res.stdout