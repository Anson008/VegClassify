import numpy as np
from imagery_wayback_driver import ImageryWaybackDriver
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import os
from mmseg.apis import init_model, inference_model, show_result_pyplot


def generate_train_data():
    # top_left = (41.763356, -111.860899)
    # bottom_right = (41.701922, -111.801124)
    # xy_min = [min(top_left[0], bottom_right[0]), min(top_left[1], bottom_right[1])]
    # xy_max = [max(top_left[0], bottom_right[0]), max(top_left[1], bottom_right[1])]
    n_samples = 1
    location_data = np.zeros((n_samples, 2))
    location_data[0, 0] = 40.68889
    location_data[0, 1] = -111.86859
    # print(location_data)

    # Specify Chrome driver path
    options = Options()
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.binary_location = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"

    # Start Chrome
    # driver = webdriver.Chrome(options=options)

    base_url = "https://livingatlas.arcgis.com/wayback/#active={:d}&mapCenter={:.6f}%2C{:.6f}%2C{:d}"
    release_num = 47963
    scale_factor = 18
    width = 512
    height = 1024

    out_path_base = f"./image/test_data/RN{release_num}_C{scale_factor}_{width}X{height}_Num{n_samples}"
    if not os.path.exists(out_path_base):
        os.makedirs(out_path_base)

    base_filename = "green_space_"
    suffix = ".png"

    pred_out_path = "./image/predictions"
    for i in range(location_data.shape[0]):
        driver = webdriver.Chrome(options=options)
        wayback = ImageryWaybackDriver(driver)
        url = base_url.format(release_num, location_data[i, 1], location_data[i, 0], scale_factor)
        wayback.load_url(url)
        wayback.toggle_off_version_filter()
        wayback.accept_cookies()
        filename = base_filename + str(i).zfill(len(str(n_samples))) + suffix
        save_to_file = os.path.join(out_path_base, filename)
        wayback.take_screenshot(width, height, save_to_file)

        config_path = "./configs/fcn_aux-hr48_256x512_80k_singlegreen.py"
        checkpoint_path = "./checkpoints/iter_1000.pth"

        test_img = save_to_file
        pred_out_filename = "pred_{}.png".format(filename)
        pred_img = os.path.join(pred_out_path, pred_out_filename)

        model = init_model(config_path, checkpoint_path, device="cuda:0")
        pred_res = inference_model(model, test_img)

        vis_image = show_result_pyplot(model, test_img, pred_res, out_file=pred_img, wait_time=1)

        print("Done!")


if __name__ == "__main__":
    generate_train_data()
